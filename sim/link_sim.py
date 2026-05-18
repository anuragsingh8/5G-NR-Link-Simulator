"""
sim/link_sim.py — Main Link-Level Simulation Engine
Orchestrates PHY → channel → PHY chain for one SNR point.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.sim_config   import SimConfig
from config.nr_tables    import get_mcs_params, get_tbs
from phy.ofdm            import OFDMModulator, OFDMDemodulator
from phy.channel         import AWGNChannel, RayleighChannel, CDLChannel
from phy.channel_est     import LSEstimator, LMMSEEstimator
from phy.equaliser       import ZFEqualiser, MMSEEqualiser, SICEqualiser
from phy.modulation      import Mapper, Demapper, LLRComputer
from phy.mimo            import MIMOChannel, BeamformingMatrix
from coding.ldpc         import LDPCEncoder, BPDecoder, RateMatch
from coding.harq         import HARQManager
from mac.amc             import LinkAdaptor, OLLAController
from mac.csi             import CQIComputer, RISelector
from sim.metrics         import BERCounter, BLERCounter, ThroughputCalc


@dataclass
class SimResult:
    snr_db:      float
    ber:         float
    bler:        float
    throughput:  float
    avg_mcs:     float
    n_slots:     int
    n_harq_tx:   int


class LinkSimulator:
    """
    5G NR Link-Level Simulator.

    Full chain per slot:
      TX bits → LDPC encode → rate match → QAM map → OFDM modulate
      → channel → OFDM demodulate → channel estimate → equalise
      → LLR compute → HARQ combine → LDPC decode → metrics

    Parameters
    ----------
    cfg : SimConfig dataclass
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._build_chain()

    def _build_chain(self):
        cfg = self.cfg
        mcs = get_mcs_params(cfg.mcs)
        self.Qm  = mcs['Qm']
        self.tbs = get_tbs(cfg.mcs, cfg.n_prb, n_layers=cfg.n_layers)

        # ── TX ────────────────────────────────────────────────────────────────
        self.mapper    = Mapper(self.Qm)
        self.modulator = OFDMModulator(cfg.n_fft, cfg.cp_len, cfg.n_subcarriers)
        self.encoder   = LDPCEncoder(k=min(self.tbs, 100), n=min(self.tbs + 50, 200))

        # ── Channel ───────────────────────────────────────────────────────────
        ch_map = {
            'AWGN':    AWGNChannel(cfg.snr_db),
            'Rayleigh':RayleighChannel(cfg.snr_db, cfg.n_tx, cfg.n_rx),
            'CDL-A':   CDLChannel('CDL-A', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, cfg.n_tx, cfg.n_rx),
            'CDL-B':   CDLChannel('CDL-B', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, cfg.n_tx, cfg.n_rx),
            'CDL-C':   CDLChannel('CDL-C', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, cfg.n_tx, cfg.n_rx),
        }
        self.channel = ch_map.get(cfg.channel_model, AWGNChannel(cfg.snr_db))
        self._is_awgn = cfg.channel_model == 'AWGN'

        # ── RX ────────────────────────────────────────────────────────────────
        self.demodulator  = OFDMDemodulator(cfg.n_fft, cfg.cp_len, cfg.n_subcarriers)
        self.ch_estimator = LMMSEEstimator(cfg.n_subcarriers, cfg.snr_db)
        det_map = {'ZF': ZFEqualiser, 'MMSE': MMSEEqualiser, 'SIC': SICEqualiser}
        DetCls = det_map.get(cfg.detector, MMSEEqualiser)
        self.equaliser = DetCls(cfg.n_tx, cfg.n_rx, cfg.snr_db) \
                         if cfg.detector in ('MMSE', 'SIC') \
                         else DetCls(cfg.n_tx, cfg.n_rx)

        self.llr_computer = LLRComputer(self.Qm, cfg.snr_db)
        self.demapper     = Demapper(self.Qm)

        # ── MAC ───────────────────────────────────────────────────────────────
        self.harq_mgr  = HARQManager(n_processes=8, n_max_tx=4)
        self.link_adapt = LinkAdaptor()
        self.olla       = OLLAController(target_bler=0.1)
        self.olla.attach(self.link_adapt)
        self.cqi_comp   = CQIComputer(n_subbands=1)
        self.ri_sel     = RISelector(cfg.n_tx, cfg.n_rx)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ber_ctr  = BERCounter()
        self.bler_ctr = BLERCounter()
        self.tput     = ThroughputCalc(cfg.slot_duration_us)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_bits(self, n: int) -> np.ndarray:
        return np.random.randint(0, 2, size=n, dtype=np.uint8)

    def _tx_chain(self, info_bits: np.ndarray) -> np.ndarray:
        """info bits → time-domain TX signal (single layer, single symbol for speed)."""
        cw = self.encoder.encode(info_bits[:self.encoder.k])
        e_bits = self.cfg.n_subcarriers * self.Qm
        rm = self.encoder.rate_match(cw, e_bits)
        symbols = self.mapper.map(rm)
        return self.modulator.modulate(symbols)  # (symbol_len,)

    def _rx_chain(self, rx_time: np.ndarray) -> tuple[np.ndarray, bool]:
        """time-domain RX → decoded bits + CRC pass/fail (simplified)."""
        rx_sym = self.demodulator.demodulate(rx_time)

        # Channel estimation (use received signal as pilot proxy for AWGN sim)
        H_est = self.ch_estimator.estimate(rx_sym)

        # Single-antenna: divide by channel estimate
        eq_sym = rx_sym / (H_est + 1e-10)

        # LLR
        llrs = self.llr_computer.compute(eq_sym)

        # LDPC decode
        n_llr = self.encoder.n
        decoder = BPDecoder(self.encoder.H, max_iter=20)
        decoded = decoder.decode(llrs[:n_llr])

        # CRC check proxy: compare parity bits
        crc_pass = bool(np.all(self.encoder.H @ decoded % 2 == 0))
        return decoded[:self.encoder.k], crc_pass

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(self, n_slots: int = 1000, verbose: bool = True) -> SimResult:
        """
        Run link simulation for n_slots slots.

        Parameters
        ----------
        n_slots : number of slots to simulate
        verbose : print progress every 100 slots

        Returns
        -------
        SimResult dataclass
        """
        mcs_accum = []
        n_harq_tx = 0

        for slot in range(n_slots):
            # ── Generate TX bits ──────────────────────────────────────────
            info_bits = self._random_bits(self.encoder.k)

            # ── TX chain ─────────────────────────────────────────────────
            tx_time = self._tx_chain(info_bits)

            # ── Channel ──────────────────────────────────────────────────
            if self._is_awgn:
                rx_time = self.channel.apply(tx_time)
            else:
                rx_time, _ = self.channel.apply(tx_time[np.newaxis, :])
                rx_time = rx_time.squeeze()

            # Ensure correct length for demodulator
            sym_len = self.modulator.symbol_len
            rx_time = rx_time[:sym_len]
            if len(rx_time) < sym_len:
                rx_time = np.pad(rx_time, (0, sym_len - len(rx_time)))

            # ── RX chain ─────────────────────────────────────────────────
            dec_bits, crc_ok = self._rx_chain(rx_time)

            # ── HARQ feedback ────────────────────────────────────────────
            self.olla.update(crc_ok)
            n_harq_tx += 1

            # ── Metrics ──────────────────────────────────────────────────
            min_len = min(len(info_bits), len(dec_bits))
            self.ber_ctr.update(info_bits[:min_len], dec_bits[:min_len])
            self.bler_ctr.update_from_ack(crc_ok)
            self.tput.update(self.tbs, crc_ok)
            mcs_accum.append(self.cfg.mcs)

            if verbose and (slot + 1) % 100 == 0:
                print(f"  Slot {slot+1:5d}/{n_slots} | "
                      f"BER={self.ber_ctr.ber:.3e} | "
                      f"BLER={self.bler_ctr.bler:.4f} | "
                      f"Tput={self.tput.avg_throughput_mbps:.2f} Mbps")

        return SimResult(
            snr_db=self.cfg.snr_db,
            ber=self.ber_ctr.ber,
            bler=self.bler_ctr.bler,
            throughput=self.tput.avg_throughput_mbps,
            avg_mcs=float(np.mean(mcs_accum)),
            n_slots=n_slots,
            n_harq_tx=n_harq_tx,
        )
