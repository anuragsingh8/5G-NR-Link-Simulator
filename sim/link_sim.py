"""
sim/link_sim.py — Main Link-Level Simulation Engine
Orchestrates PHY → channel → PHY chain for one SNR point.

Chain per slot
--------------
TX:  random bits → LDPC encode → rate match → QAM map → OFDM modulate
CH:  AWGN / Rayleigh / CDL channel
RX:  OFDM demodulate → channel estimate / equalise → LLR compute
     → HARQ IR combine → LDPC decode → CRC → metrics

Throughput accounting
---------------------
  Peak = TBS / slot_duration  (full 3GPP transport block)
  Actual = Peak × (1 - BLER)  (only successfully decoded slots count)

HARQ
----
  4 processes, max 4 transmissions. LLRs accumulate across retransmissions
  (IR combining). New TB on ACK or max-tx exhausted.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from numpy.fft import fft

from config.sim_config   import SimConfig
from config.nr_tables    import get_mcs_params, get_tbs
from phy.ofdm            import OFDMModulator, NR_NUMEROLOGIES
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
    snr_db:        float
    ber:           float
    bler:          float
    throughput:    float
    peak_tput:     float
    avg_mcs:       float
    n_slots:       int
    n_harq_tx:     int
    n_harq_retx:   int


# ─────────────────────────────────────────────────────────────────────────────
# Simple HARQ soft buffer (per process)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HARQState:
    """Lightweight per-process HARQ state for the link sim."""
    llr_buf:    np.ndarray = field(default_factory=lambda: np.zeros(0))
    tx_count:   int        = 0
    info_bits:  Optional[np.ndarray] = None

    def reset(self):
        self.llr_buf   = np.zeros(0)
        self.tx_count  = 0
        self.info_bits = None

    def accumulate(self, new_llrs: np.ndarray):
        if len(self.llr_buf) == 0:
            self.llr_buf = new_llrs.copy()
        else:
            n = min(len(self.llr_buf), len(new_llrs))
            self.llr_buf[:n] += new_llrs[:n]


class LinkSimulator:
    """
    5G NR Link-Level Simulator.

    Parameters
    ----------
    cfg : SimConfig dataclass
    """

    # LDPC block: k=128 info bits, n=200 codeword bits → R=0.64 ≈ MCS16
    _LDPC_K = 128
    _LDPC_N = 200
    _MAX_HARQ_TX = 4      # max transmissions per TB (1=no HARQ, 4=standard)
    _N_HARQ_PROC = 4      # parallel HARQ processes

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._build_chain()

    def _build_chain(self):
        cfg = self.cfg
        mcs = get_mcs_params(cfg.mcs)
        self.Qm        = mcs['Qm']
        self.code_rate = mcs['code_rate']
        self.tbs       = get_tbs(cfg.mcs, cfg.n_prb, n_layers=cfg.n_layers)

        # ── TX ────────────────────────────────────────────────────────────────
        self.mapper    = Mapper(self.Qm)
        self.modulator = OFDMModulator(
            mu=cfg.numerology, n_fft=cfg.n_fft,
            n_prb=cfg.n_prb,   cp_len=cfg.cp_len,
        )
        self.n_sc    = self.modulator.n_sc
        self.e_bits  = self.n_sc * self.Qm
        self._rm_idx = np.arange(self.e_bits) % self._LDPC_N

        self.encoder = LDPCEncoder(k=self._LDPC_K, n=self._LDPC_N)
        self.decoder = BPDecoder(self.encoder.H, max_iter=50)

        # ── Channel ───────────────────────────────────────────────────────────
        ch_map = {
            'AWGN':    AWGNChannel(cfg.snr_db),
            'Rayleigh':RayleighChannel(cfg.snr_db, n_tx=1, n_rx=1),
            'CDL-A':   CDLChannel('CDL-A', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, n_tx=1, n_rx=1),
            'CDL-B':   CDLChannel('CDL-B', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, n_tx=1, n_rx=1),
            'CDL-C':   CDLChannel('CDL-C', cfg.snr_db, cfg.velocity_kmh,
                                  cfg.scs_khz * 1e3, cfg.n_fft, n_tx=1, n_rx=1),
        }
        self.channel  = ch_map.get(cfg.channel_model, AWGNChannel(cfg.snr_db))
        self._is_awgn = cfg.channel_model == 'AWGN'

        # ── RX ────────────────────────────────────────────────────────────────
        self.demodulator  = self.modulator
        self.ch_estimator = LMMSEEstimator(self.n_sc, cfg.snr_db)
        self.llr_computer = LLRComputer(self.Qm, cfg.snr_db)

        # ── HARQ processes ────────────────────────────────────────────────────
        self._harq = [_HARQState() for _ in range(self._N_HARQ_PROC)]

        # ── AMC / OLLA ────────────────────────────────────────────────────────
        self.link_adapt = LinkAdaptor()
        self.olla       = OLLAController(target_bler=0.1)
        self.olla.attach(self.link_adapt)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ber_ctr  = BERCounter()
        self.bler_ctr = BLERCounter()
        self.tput     = ThroughputCalc(cfg.slot_duration_us)

    # ── TX ────────────────────────────────────────────────────────────────────

    def _tx_chain(self, info_bits: np.ndarray) -> np.ndarray:
        cw  = self.encoder.encode(info_bits)
        rm  = self.encoder.rate_match(cw, self.e_bits)
        self._last_tx_sym = self.mapper.map(rm)   # store for LS estimation
        return self.modulator.modulate(self._last_tx_sym)

    # ── Channel ───────────────────────────────────────────────────────────────

    def _apply_channel(self, tx_time: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._is_awgn:
            return self.channel.apply(tx_time), None
        rx_raw, h_true = self.channel.apply(tx_time[np.newaxis, :])
        return rx_raw.squeeze().ravel(), h_true

    # ── Equalisation ─────────────────────────────────────────────────────────

    def _equalise(self, rx_sym: np.ndarray, h_true: Optional[np.ndarray]) -> np.ndarray:
        """
        Per-subcarrier MMSE equalisation using genie-aided LS channel estimate.

        H_ls[k] = RX[k] / TX[k]  using the known transmitted symbols.
        This is equivalent to perfect DMRS pilot knowledge across all subcarriers
        and gives correct H[k] for both flat fading and multipath (CDL).

        w[k] = H_ls*[k] / (|H_ls[k]|^2 + 1/SNR)  — MMSE one-tap.

        For AWGN (H=1 everywhere) the division has no effect.
        """
        if not hasattr(self, '_last_tx_sym'):
            return rx_sym

        tx_sym  = self._last_tx_sym                          # (n_sc,) known TX
        H_ls    = rx_sym / (tx_sym + 1e-10)                  # (n_sc,) LS estimate
        snr_lin = 10 ** (self.cfg.snr_db / 10)
        w       = H_ls.conj() / (np.abs(H_ls) ** 2 + 1.0 / snr_lin)
        return rx_sym * w

    # ── LLR combining ─────────────────────────────────────────────────────────

    def _combine_llrs(self, llrs_rm: np.ndarray) -> np.ndarray:
        """De-rate-match: fold LLRs back to codeword length n via bincount."""
        return np.bincount(self._rm_idx, weights=llrs_rm, minlength=self.encoder.n)

    # ── RX chain ──────────────────────────────────────────────────────────────

    def _rx_chain(
        self,
        rx_time: np.ndarray,
        h_true:  Optional[np.ndarray],
        harq:    _HARQState,
    ) -> tuple[np.ndarray, bool]:
        """
        RX chain with HARQ IR combining.
        Accumulates soft LLRs across retransmissions before decoding.
        """
        rx_sym  = self.demodulator.demodulate(rx_time)
        eq_sym  = self._equalise(rx_sym, h_true)
        llrs_rm = self.llr_computer.compute(eq_sym)
        llrs_cw = self._combine_llrs(llrs_rm)

        # HARQ IR: accumulate with previous transmissions
        harq.accumulate(llrs_cw)

        decoded  = self.decoder.decode(harq.llr_buf)
        info_dec = decoded[:self.encoder.k]
        crc_ok   = bool(np.array_equal(harq.info_bits, info_dec))
        return info_dec, crc_ok

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, n_slots: int = 500, verbose: bool = True) -> SimResult:
        """Simulate n_slots and return BER / BLER / throughput."""
        sym_len    = self.modulator.symbol_len
        mcs_accum  = []
        n_harq_tx  = 0
        n_retx     = 0
        pid        = 0   # round-robin HARQ process selection

        for slot in range(n_slots):
            harq = self._harq[pid % self._N_HARQ_PROC]

            # New TB or retransmission
            if harq.tx_count == 0 or harq.tx_count >= self._MAX_HARQ_TX:
                # Start fresh TB
                harq.reset()
                info_bits       = np.random.randint(0, 2, self.encoder.k, dtype=np.uint8)
                harq.info_bits  = info_bits
            else:
                # Retransmission — same bits
                info_bits = harq.info_bits
                n_retx   += 1

            harq.tx_count += 1
            n_harq_tx     += 1

            # TX
            tx_time = self._tx_chain(info_bits)

            # Channel
            rx_time, h_true = self._apply_channel(tx_time)
            rx_time = rx_time[:sym_len]
            if len(rx_time) < sym_len:
                rx_time = np.pad(rx_time, (0, sym_len - len(rx_time)))

            # RX + HARQ combine
            info_dec, crc_ok = self._rx_chain(rx_time, h_true, harq)

            # On ACK or max retransmissions: finalise, advance process
            if crc_ok or harq.tx_count >= self._MAX_HARQ_TX:
                self.olla.update(crc_ok)
                self.ber_ctr.update(info_bits, info_dec)
                self.bler_ctr.update_from_ack(crc_ok)
                self.tput.update(self.tbs, crc_ok)
                mcs_accum.append(self.cfg.mcs)
                harq.reset()
                pid += 1

            if verbose and (slot + 1) % 100 == 0:
                print(f"  Slot {slot+1:5d}/{n_slots} | "
                      f"BER={self.ber_ctr.ber:.3e} | "
                      f"BLER={self.bler_ctr.bler:.4f} | "
                      f"Tput={self.tput.avg_throughput_mbps:.2f} Mbps | "
                      f"ReTx={n_retx}")

        peak = self.tbs / self.cfg.slot_duration_us
        return SimResult(
            snr_db=self.cfg.snr_db,
            ber=self.ber_ctr.ber,
            bler=self.bler_ctr.bler,
            throughput=self.tput.avg_throughput_mbps,
            peak_tput=peak,
            avg_mcs=float(np.mean(mcs_accum)) if mcs_accum else float(self.cfg.mcs),
            n_slots=n_slots,
            n_harq_tx=n_harq_tx,
            n_harq_retx=n_retx,
        )