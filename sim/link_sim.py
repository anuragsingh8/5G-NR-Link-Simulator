"""
sim/link_sim.py — Main Link-Level Simulation Engine
Orchestrates PHY → channel → PHY chain for one SNR point.

Chain per slot
--------------
TX:  random bits → LDPC encode → rate match → QAM map → OFDM modulate
CH:  AWGN / Rayleigh / CDL channel
RX:  OFDM demodulate → (channel estimate) → equalise → LLR compute
     → LLR de-rate-match combine → LDPC decode → CRC → metrics
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataclasses import dataclass
from typing import Optional

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
    snr_db:     float
    ber:        float
    bler:       float
    throughput: float
    avg_mcs:    float
    n_slots:    int
    n_harq_tx:  int


class LinkSimulator:
    """
    5G NR Link-Level Simulator.

    Parameters
    ----------
    cfg : SimConfig dataclass
    """

    # LDPC block size — small enough for the toy BP decoder to converge quickly
    # k = info bits, n = codeword bits.  Rate ≈ k/n should match MCS code rate.
    _LDPC_K = 128
    _LDPC_N = 200   # R = 0.640 — matches MCS 16 (R=0.6426)

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
        self.n_sc      = self.modulator.n_sc
        self.e_bits    = self.n_sc * self.Qm          # bits per OFDM symbol
        self._rm_idx   = np.arange(self.e_bits) % self._LDPC_N  # precomputed for LLR combining

        # LDPC sized so n = e_bits would be ideal, but toy BP decoder needs
        # small blocks. We rate-match by circular repetition and combine LLRs.
        self.encoder   = LDPCEncoder(k=self._LDPC_K, n=self._LDPC_N)
        # Pre-build decoder once — reused every slot (major speed win)
        self.decoder   = BPDecoder(self.encoder.H, max_iter=50)

        # ── Channel ───────────────────────────────────────────────────────────
        # Single-stream time-domain channel (SISO).
        # The OFDM modulator produces one waveform; the channel adds fading + noise.
        # MIMO spatial processing (CSI, precoding, equalisation) operates in
        # the frequency domain on the per-subcarrier symbol grid.
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
        self.channel   = ch_map.get(cfg.channel_model, AWGNChannel(cfg.snr_db))
        self._is_awgn  = cfg.channel_model == 'AWGN'

        # ── RX ────────────────────────────────────────────────────────────────
        # Modulator and demodulator are the same object — symbol_len guaranteed identical
        self.demodulator  = self.modulator

        # Channel estimator — used only for fading channels
        self.ch_estimator = LMMSEEstimator(self.n_sc, cfg.snr_db)

        det_map = {'ZF': ZFEqualiser, 'MMSE': MMSEEqualiser, 'SIC': SICEqualiser}
        DetCls = det_map.get(cfg.detector, MMSEEqualiser)
        self.equaliser = (
            DetCls(cfg.n_tx, cfg.n_rx, cfg.snr_db)
            if cfg.detector in ('MMSE', 'SIC')
            else DetCls(cfg.n_tx, cfg.n_rx)
        )

        self.llr_computer = LLRComputer(self.Qm, cfg.snr_db)
        self.demapper     = Demapper(self.Qm)

        # ── MAC ───────────────────────────────────────────────────────────────
        self.harq_mgr   = HARQManager(n_processes=8, n_max_tx=4)
        self.link_adapt = LinkAdaptor()
        self.olla       = OLLAController(target_bler=0.1)
        self.olla.attach(self.link_adapt)
        self.cqi_comp   = CQIComputer(n_subbands=1)
        self.ri_sel     = RISelector(cfg.n_tx, cfg.n_rx)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ber_ctr = BERCounter()
        self.bler_ctr = BLERCounter()
        self.tput    = ThroughputCalc(cfg.slot_duration_us)

    # ── TX chain ──────────────────────────────────────────────────────────────

    def _tx_chain(self, info_bits: np.ndarray) -> np.ndarray:
        """
        info bits → time-domain TX waveform (one OFDM symbol).

        Rate matching circularly repeats the codeword to fill exactly e_bits
        positions. The receiver folds the LLRs back to n positions before decoding.
        """
        cw      = self.encoder.encode(info_bits)          # (n,)
        rm      = self.encoder.rate_match(cw, self.e_bits)  # (e_bits,) — repeated
        symbols = self.mapper.map(rm)                      # (n_sc,)
        return self.modulator.modulate(symbols)            # (symbol_len,)

    # ── RX chain ──────────────────────────────────────────────────────────────

    def _equalize_awgn(self, rx_sym: np.ndarray) -> np.ndarray:
        """
        AWGN: H = 1 everywhere. No estimation needed — return rx_sym directly.
        """
        return rx_sym

    def _equalize_fading(self, rx_sym: np.ndarray, h_true: np.ndarray) -> np.ndarray:
        """
        Fading channel equalisation — SISO flat-fading.

        h_true shape: (1, 1, n_samples) from SISO CDL/Rayleigh.
        Takes the mean of h over the useful symbol window (after CP removal)
        as a single complex coefficient, then divides each subcarrier by it.

        Falls back to LMMSE pilot estimate if h_true is None.
        """
        if h_true is None:
            H_est = self.ch_estimator.estimate(rx_sym)
            return rx_sym / (H_est + 1e-10)

        # h_true: (1, 1, n_samples) — take mean over useful symbol window
        cp       = self.modulator.cp_len
        h_scalar = h_true[0, 0, cp:cp + self.modulator.n_fft].mean()
        mag      = np.abs(h_scalar)
        if mag < 1e-6:
            return rx_sym
        return rx_sym / (h_scalar + 1e-10)

    def _combine_llrs(self, llrs_rm: np.ndarray) -> np.ndarray:
        """
        De-rate-match: fold repeated LLR positions back to codeword length n.
        Uses np.bincount for O(e_bits) vectorised accumulation.
        """
        return np.bincount(self._rm_idx, weights=llrs_rm, minlength=self.encoder.n)

    def _rx_chain(
        self,
        rx_time: np.ndarray,
        h_true:  Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Time-domain RX → decoded info bits + pass/fail flag.

        Pass/fail uses bit-exact comparison against the re-encoded codeword
        (avoids relying on the toy LDPC syndrome which is not always consistent).
        """
        # Demodulate
        rx_sym = self.demodulator.demodulate(rx_time)  # (n_sc,)

        # Equalise
        if self._is_awgn:
            eq_sym = self._equalize_awgn(rx_sym)
        else:
            eq_sym = self._equalize_fading(rx_sym, h_true)

        # Soft LLRs → de-rate-match combining → decode
        llrs_rm = self.llr_computer.compute(eq_sym)   # (e_bits,)
        llrs_cw = self._combine_llrs(llrs_rm)          # (n,)
        decoded = self.decoder.decode(llrs_cw)         # (n,)

        info_dec = decoded[:self.encoder.k]
        return info_dec, decoded

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, n_slots: int = 500, verbose: bool = True) -> SimResult:
        """
        Simulate n_slots slots and return BER / BLER / throughput.
        """
        mcs_accum = []
        n_harq_tx = 0
        sym_len   = self.modulator.symbol_len

        for slot in range(n_slots):
            # TX
            info_bits = np.random.randint(0, 2, self.encoder.k, dtype=np.uint8)
            tx_time   = self._tx_chain(info_bits)

            # Channel
            h_true = None
            if self._is_awgn:
                rx_time = self.channel.apply(tx_time)
            else:
                rx_raw, h_true = self.channel.apply(tx_time[np.newaxis, :])
                rx_time = rx_raw.squeeze()

            # Length guard — ravel handles (1, N) squeeze edge cases
            rx_time = rx_time.ravel()
            if len(rx_time) < sym_len:
                rx_time = np.pad(rx_time, (0, sym_len - len(rx_time)))
            rx_time = rx_time[:sym_len]

            # RX
            info_dec, decoded_cw = self._rx_chain(rx_time, h_true)

            # CRC: bit-exact comparison (reliable regardless of H matrix quality)
            crc_ok = bool(np.array_equal(info_bits, info_dec))

            # HARQ / AMC feedback
            self.olla.update(crc_ok)
            n_harq_tx += 1

            # Metrics
            self.ber_ctr.update(info_bits, info_dec)
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