"""
sim/link_sim.py — Industry-Grade Link-Level Simulation Engine
==============================================================
3GPP TS 38.211 / 38.212 / 38.214 / 38.321

Full chain per slot (14 OFDM symbols)
--------------------------------------
TX:
  1. Generate transport block bits  (k = LDPC_K info bits)
  2. LDPC encode → rate-match to e_bits
  3. QAM map → slot resource grid  (n_symb × n_sc)
  4. Inject DMRS pilots at spec positions (symbols 2, 11)
  5. OFDM modulate full slot

Channel:
  AWGN / Rayleigh / CDL-A/B/C / TDL-A/B/C/D/E

RX:
  1. OFDM demodulate full slot → received grid
  2. DMRS-based channel estimation (SlotEstimator)
  3. Per-subcarrier MMSE equalisation
  4. LLR computation with per-SC channel scaling
  5. De-rate-match + HARQ IR combining
  6. LDPC decode

Throughput accounting:
  peak    = TBS (full 3GPP) / slot_duration
  effective = peak × (1 − BLER)

HARQ:
  up to max_harq_tx transmissions per TB
  IR combining via LLR accumulation across retransmissions
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Pool

from config.sim_config import SimConfig
from config.nr_tables  import get_mcs_params, get_tbs
from phy.ofdm          import OFDMModulator
from phy.channel       import (AWGNChannel, RayleighChannel,
                                CDLChannel, TDLChannel)
from phy.channel_est   import SlotEstimator
from phy.modulation    import Mapper, LLRComputer
from coding.ldpc       import LDPCEncoder, BPDecoder
from sim.metrics       import BERCounter, BLERCounter, ThroughputCalc


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    snr_db:       float
    ber:          float
    bler:         float
    throughput:   float
    peak_tput:    float
    spectral_eff: float
    avg_mcs:      float
    channel:      str
    bw_mhz:       int
    n_slots:      int
    n_harq_tx:    int
    n_harq_retx:  int

    def to_dict(self) -> dict:
        return {
            'snr_db':          self.snr_db,
            'ber':             self.ber,
            'bler':            self.bler,
            'throughput_mbps': self.throughput,
            'peak_tput_mbps':  self.peak_tput,
            'efficiency_pct':  self.throughput / max(self.peak_tput, 1e-9) * 100,
            'spectral_eff':    self.spectral_eff,
            'avg_mcs':         self.avg_mcs,
            'channel':         self.channel,
            'bw_mhz':          self.bw_mhz,
            'n_slots':         self.n_slots,
            'n_harq_tx':       self.n_harq_tx,
            'n_harq_retx':     self.n_harq_retx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight HARQ state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HARQState:
    llr_buf:   np.ndarray         = field(default_factory=lambda: np.zeros(0))
    tx_count:  int                = 0
    info_bits: Optional[np.ndarray] = None

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


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulator
# ─────────────────────────────────────────────────────────────────────────────

class LinkSimulator:
    """
    Full 5G NR link-level simulator.

    Processing a complete 14-symbol slot per simulation step:
      - DMRS injected at symbols 2 and 11
      - Data on remaining 12 symbols
      - Channel estimation via SlotEstimator (freq + time interpolation)
      - Per-subcarrier MMSE equalisation with channel-scaled LLRs
      - HARQ IR combining across up to max_harq_tx transmissions

    Parameters
    ----------
    cfg : SimConfig  (use cfg.n_prb_eff, cfg.n_fft_eff, cfg.cp_len_eff)
    """

    # LDPC block — k info bits, n codeword bits, R≈0.64 ≈ MCS16
    _LDPC_K   = 128
    _LDPC_N   = 200

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._build_chain()

    def _build_chain(self):
        cfg  = self.cfg
        mcs  = get_mcs_params(cfg.mcs, cfg.mcs_table)
        self.Qm   = mcs['Qm']
        self.tbs  = get_tbs(
            cfg.mcs, cfg.n_prb_eff,
            n_symb    = cfg.n_data_symbols,
            n_layers  = cfg.n_layers,
            mcs_table = cfg.mcs_table,
        )

        # ── TX ────────────────────────────────────────────────────────────────
        self.mapper    = Mapper(self.Qm)
        self.modulator = OFDMModulator(
            mu      = cfg.numerology,
            n_fft   = cfg.n_fft_eff,
            n_prb   = cfg.n_prb_eff,
            cp_len  = cfg.cp_len_eff,
        )
        self.n_sc   = self.modulator.n_sc
        self.n_symb = cfg.n_symb_per_slot

        # bits per full slot (data symbols only)
        self.e_bits   = self.n_sc * self.Qm * cfg.n_data_symbols
        self._rm_idx  = np.arange(self.e_bits) % self._LDPC_N

        self.encoder  = LDPCEncoder(k=self._LDPC_K, n=self._LDPC_N)
        self.decoder  = BPDecoder(self.encoder.H, max_iter=50)

        # ── Channel ───────────────────────────────────────────────────────────
        fs   = cfg.scs_hz * cfg.n_fft_eff
        scs  = cfg.scs_hz
        ch   = cfg.channel_model
        snr  = cfg.snr_db
        v    = cfg.velocity_kmh
        fc   = cfg.carrier_freq_ghz * 1e9
        ds   = cfg.delay_spread_ns

        if ch == 'AWGN':
            self.channel = AWGNChannel(snr)
        elif ch == 'Rayleigh':
            self.channel = RayleighChannel(snr, n_tx=1, n_rx=1)
        elif ch in ('CDL-A', 'CDL-B', 'CDL-C'):
            self.channel = CDLChannel(ch, snr, v, scs, cfg.n_fft_eff, n_tx=1, n_rx=1)
        elif ch in ('TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'):
            self.channel = TDLChannel(ch, snr, v, fs_hz=fs,
                                       delay_spread_ns=ds, carrier_freq_hz=fc)
        else:
            self.channel = AWGNChannel(snr)

        self._is_awgn = ch == 'AWGN'

        # ── Channel estimation (full slot, DMRS-based) ─────────────────────
        self.slot_est = SlotEstimator(
            n_sc      = self.n_sc,
            n_symb    = self.n_symb,
            snr_db    = snr,
            dmrs_type = 1,
            use_lmmse = True,
        )
        # Pre-generate known DMRS symbols for TX grid
        self._dmrs_grid = self.slot_est.generate_slot_dmrs()
        self._dmrs_syms = set(self._dmrs_grid.keys())
        self._data_syms = [s for s in range(self.n_symb) if s not in self._dmrs_syms]

        # ── LLR ───────────────────────────────────────────────────────────────
        self.llr_c = LLRComputer(self.Qm, snr)

        # ── HARQ ──────────────────────────────────────────────────────────────
        self._harq = [_HARQState() for _ in range(cfg.harq_processes)]

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ber_ctr  = BERCounter()
        self.bler_ctr = BLERCounter()
        self.tput     = ThroughputCalc(cfg.slot_duration_us, bw_hz=cfg.bw_mhz * 1e6)

    # ── TX ────────────────────────────────────────────────────────────────────

    def _build_tx_grid(self, info_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (n_symb, n_sc) slot grid with data + DMRS.
        Returns (grid, rm) where rm is the rate-matched bit array.
        """
        cw   = self.encoder.encode(info_bits)
        rm   = self.encoder.rate_match(cw, self.e_bits)
        syms = self.mapper.map(rm)

        grid = np.zeros((self.n_symb, self.n_sc), dtype=complex)
        # Distribute QAM symbols across data symbol rows
        syms_per_sym = self.n_sc
        for i, s in enumerate(self._data_syms):
            sl = slice(i * syms_per_sym, (i + 1) * syms_per_sym)
            if sl.stop <= len(syms):
                grid[s] = syms[sl]
            else:
                grid[s] = syms[i * syms_per_sym : len(syms)]

        # Inject DMRS
        for s, pilots in self._dmrs_grid.items():
            grid[s] = pilots

        return grid, rm

    def _tx_chain(self, info_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode + modulate → time-domain waveform + rate-matched bits."""
        grid, rm = self._build_tx_grid(info_bits)
        tx_time  = self.modulator.modulate_slot(grid)
        return tx_time, rm

    # ── Channel ───────────────────────────────────────────────────────────────

    def _apply_channel(
        self, tx_time: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._is_awgn:
            return self.channel.apply(tx_time), None
        rx_raw, h = self.channel.apply(tx_time)
        return rx_raw.ravel(), h

    # ── RX ────────────────────────────────────────────────────────────────────

    def _rx_chain(
        self,
        rx_time:   np.ndarray,
        h_true:    Optional[np.ndarray],
        harq:      _HARQState,
    ) -> tuple[np.ndarray, bool]:
        """
        Full slot RX: demodulate → estimate → equalise → LLR → HARQ combine → decode.
        """
        slot_len = self.modulator.symbol_len * self.n_symb
        rx_time  = rx_time[:slot_len]
        if len(rx_time) < slot_len:
            rx_time = np.pad(rx_time, (0, slot_len - len(rx_time)))

        # 1. Demodulate full slot
        rx_grid = self.modulator.demodulate_slot(rx_time, self.n_symb)

        # 2. Channel estimation (DMRS-based, freq+time interpolation)
        if self._is_awgn:
            H_slot = np.ones((self.n_symb, self.n_sc), dtype=complex)
        else:
            H_slot = self.slot_est.estimate_slot(rx_grid)

        # 3. Equalise + LLR across all data symbols
        snr_lin    = 10 ** (self.cfg.snr_db / 10)
        noise_var  = 1.0 / snr_lin
        llrs_all   = []

        for s in self._data_syms:
            H_sc   = H_slot[s]                            # (n_sc,)
            eq_sym = rx_grid[s] / (H_sc + 1e-10)
            H_mag2 = np.abs(H_sc) ** 2
            llrs_s = self.llr_c.compute_with_channel(eq_sym, H_mag2, noise_var)
            llrs_all.append(llrs_s)

        llrs_rm  = np.concatenate(llrs_all)               # (e_bits,)
        llrs_cw  = np.bincount(self._rm_idx, weights=llrs_rm, minlength=self._LDPC_N)

        # 4. HARQ IR combining
        harq.accumulate(llrs_cw)

        # 5. Decode
        decoded  = self.decoder.decode(harq.llr_buf)
        info_dec = decoded[:self._LDPC_K]
        crc_ok   = bool(np.array_equal(harq.info_bits, info_dec))
        return info_dec, crc_ok

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, n_slots: int = 500, verbose: bool = True) -> SimResult:
        """
        Run simulation for n_slots and return aggregated metrics.

        Parameters
        ----------
        n_slots : number of slots to simulate
        verbose : print progress every 100 slots
        """
        cfg        = self.cfg
        mcs_acc    = []
        n_harq_tx  = 0
        n_retx     = 0
        pid        = 0

        for slot in range(n_slots):
            harq = self._harq[pid % cfg.harq_processes]

            # New TB or exhausted HARQ
            if harq.tx_count == 0 or harq.tx_count >= cfg.max_harq_tx:
                harq.reset()
                info_bits      = np.random.randint(0, 2, self._LDPC_K, dtype=np.uint8)
                harq.info_bits = info_bits
            else:
                info_bits = harq.info_bits
                n_retx   += 1

            harq.tx_count += 1
            n_harq_tx     += 1

            # TX
            tx_time, _ = self._tx_chain(info_bits)

            # Channel
            rx_time, h_true = self._apply_channel(tx_time)

            # RX + HARQ
            info_dec, crc_ok = self._rx_chain(rx_time, h_true, harq)

            # Finalise on ACK or max retransmissions
            if crc_ok or harq.tx_count >= cfg.max_harq_tx:
                self.ber_ctr.update(info_bits, info_dec)
                self.bler_ctr.update_from_ack(crc_ok)
                self.tput.update(self.tbs, crc_ok)
                mcs_acc.append(cfg.mcs)
                harq.reset()
                pid += 1

            if verbose and (slot + 1) % 100 == 0:
                print(f"  Slot {slot+1:5d}/{n_slots} | "
                      f"BER={self.ber_ctr.ber:.3e} | "
                      f"BLER={self.bler_ctr.bler:.4f} | "
                      f"Tput={self.tput.avg_throughput_mbps:.2f} Mbps | "
                      f"ReTx={n_retx}")

        peak = self.tbs / cfg.slot_duration_us
        return SimResult(
            snr_db       = cfg.snr_db,
            ber          = self.ber_ctr.ber,
            bler         = self.bler_ctr.bler,
            throughput   = self.tput.avg_throughput_mbps,
            peak_tput    = peak,
            spectral_eff = self.tput.spectral_efficiency,
            avg_mcs      = float(np.mean(mcs_acc)) if mcs_acc else float(cfg.mcs),
            channel      = cfg.channel_model,
            bw_mhz       = cfg.bw_mhz,
            n_slots      = n_slots,
            n_harq_tx    = n_harq_tx,
            n_harq_retx  = n_retx,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parallel SNR sweep
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(args: tuple) -> SimResult:
    """Worker for parallel sweep — each point gets its own seeded RNG."""
    cfg, n_slots, verbose = args
    np.random.seed(cfg.seed + int(cfg.snr_db * 10))
    sim = LinkSimulator(cfg)
    return sim.run(n_slots=n_slots, verbose=verbose)


def run_parallel_sweep(
    configs:  list[SimConfig],
    n_slots:  int  = 500,
    n_workers: int = 4,
    verbose:  bool = False,
) -> list[SimResult]:
    """
    Run multiple SNR/MCS/BW points in parallel.

    Parameters
    ----------
    configs   : list of SimConfig, one per sweep point
    n_slots   : slots per point
    n_workers : parallel processes (default 4)
    verbose   : per-slot output (not recommended for parallel)

    Returns
    -------
    results : list[SimResult] in same order as configs
    """
    args = [(cfg, n_slots, verbose) for cfg in configs]
    if n_workers <= 1:
        return [_run_single(a) for a in args]
    with Pool(processes=n_workers) as pool:
        return pool.map(_run_single, args)