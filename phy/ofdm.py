"""
phy/ofdm.py — OFDM Modulator / Demodulator
============================================
3GPP TS 38.211 Sections 4.3, 5.3, 7.4

All five NR numerologies (μ = 0..4) are supported via the Numerology dataclass
and NR_NUMEROLOGIES table. The modulator validates CP length against a
caller-supplied maximum delay spread so misconfigured numerology/channel
combinations are caught at construction time rather than silently producing
wrong BER curves.

Numerology table (TS 38.211 Table 4.3.2-1)
-------------------------------------------
μ   SCS(kHz)  CP_normal(μs)  CP_extended(μs)  Slots/frame
0   15         4.6875         —                10
1   30         2.3438         —                20
2   60         1.1719         4.1667           40
3   120        0.5859         —                80
4   240        0.2930         —                160

Classes
-------
OFDMModulator  — modulate() / demodulate() / modulate_slot() / demodulate_slot()
                 CP insert/remove, DC-centred subcarrier mapping, guard-band zeroing,
                 DMRS pilot injection, info() / print_info()
NRFrameStructure — slot/mini-slot map, DMRS symbol positions, PRB grid shape
"""

from __future__ import annotations
import numpy as np
from numpy.fft import fft, ifft
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# NR Numerology table  (TS 38.211 Table 4.3.2-1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Numerology:
    mu:                 int
    scs_khz:            float   # subcarrier spacing kHz
    cp_normal_us:       float   # normal CP duration μs
    cp_extended_us:     float   # extended CP μs (μ=2 only, else 0)
    slots_per_frame:    int
    slots_per_subframe: int
    symb_per_slot:      int = 14

    @property
    def scs_hz(self) -> float:
        return self.scs_khz * 1e3

    @property
    def symbol_duration_us(self) -> float:
        """Useful symbol duration (1/SCS) in μs."""
        return 1e6 / self.scs_hz

    @property
    def slot_duration_us(self) -> float:
        """One slot = 1 ms / slots_per_subframe."""
        return 1e3 / self.slots_per_subframe


NR_NUMEROLOGIES: dict[int, Numerology] = {
    0: Numerology(0,  15,   4.6875, 0.0,    10,  1),
    1: Numerology(1,  30,   2.3438, 0.0,    20,  2),
    2: Numerology(2,  60,   1.1719, 4.1667, 40,  4),
    3: Numerology(3, 120,   0.5859, 0.0,    80,  8),
    4: Numerology(4, 240,   0.2930, 0.0,   160, 16),
}


# ─────────────────────────────────────────────────────────────────────────────
# DMRS helpers  (TS 38.211 Section 7.4.1.1 — Type-1, single port, comb-2)
# ─────────────────────────────────────────────────────────────────────────────

def dmrs_pilot_sequence(n_sc: int, seed: int = 42) -> np.ndarray:
    """
    Generate QPSK DMRS pilots for n_sc subcarriers.
    Pilots on even subcarriers (comb-2): indices 0, 2, 4, ...
    Odd subcarriers are zeroed (null subcarriers in DMRS symbol).

    Returns (n_sc,) complex array.
    """
    rng       = np.random.default_rng(seed)
    n_pilots  = n_sc // 2
    bits      = rng.integers(0, 2, size=2 * n_pilots)
    pilot_vals = (1 - 2 * bits[0::2] + 1j * (1 - 2 * bits[1::2])) / np.sqrt(2)
    grid        = np.zeros(n_sc, dtype=complex)
    grid[0::2]  = pilot_vals
    return grid


def dmrs_symbol_indices(mu: int = 1) -> list[int]:
    """
    OFDM symbol indices carrying DMRS within one slot.
    Type-A mapping, single-symbol, position 0 (TS 38.211 Table 7.4.1.1.2-3).
    Symbols 2 and 11 for all numerologies under normal CP.
    """
    return [2, 11]


# ─────────────────────────────────────────────────────────────────────────────
# NR Frame Structure helper
# ─────────────────────────────────────────────────────────────────────────────

class NRFrameStructure:
    """
    3GPP NR frame / subframe / slot / mini-slot structure.

    Parameters
    ----------
    mu    : numerology index 0-4
    n_prb : number of allocated PRBs
    """

    MINI_SLOT_LENGTHS = [2, 4, 7]   # TS 38.211 Table 11.1-1

    def __init__(self, mu: int = 1, n_prb: int = 52):
        if mu not in NR_NUMEROLOGIES:
            raise ValueError(f"μ must be 0-4, got {mu}")
        self.num   = NR_NUMEROLOGIES[mu]
        self.n_prb = n_prb
        self.n_sc  = n_prb * 12

    @property
    def slots_per_frame(self) -> int:
        return self.num.slots_per_frame

    @property
    def symb_per_frame(self) -> int:
        return self.slots_per_frame * self.num.symb_per_slot

    @property
    def prb_grid_shape(self) -> tuple[int, int]:
        """(n_symbols_per_slot, n_subcarriers) per slot."""
        return (self.num.symb_per_slot, self.n_sc)

    def dmrs_positions(self) -> list[int]:
        """Symbol indices carrying DMRS in one slot."""
        return dmrs_symbol_indices(self.num.mu)

    def data_symbol_indices(self) -> list[int]:
        """Symbol indices carrying data (all non-DMRS symbols)."""
        dmrs = set(self.dmrs_positions())
        return [i for i in range(self.num.symb_per_slot) if i not in dmrs]

    def print_slot_map(self):
        """Pretty-print per-symbol resource grid map for one slot."""
        dmrs = set(self.dmrs_positions())
        print(f"\nNR Slot Map  μ={self.num.mu}  SCS={self.num.scs_khz} kHz  "
              f"{self.n_prb} PRBs ({self.n_sc} SCs)")
        print(f"{'Sym':>4} | {'Type':^8} | {'Content':^30}")
        print("─" * 50)
        for s in range(self.num.symb_per_slot):
            sym_type = "DMRS" if s in dmrs else "DATA"
            content  = "▓▓ pilots (comb-2) ▓▓" if s in dmrs else "░░ data subcarriers ░░"
            print(f"  {s:2d} | {sym_type:^8} | {content}")
        print(f"\nSlot duration : {self.num.slot_duration_us:.3f} μs")
        print(f"CP (normal)   : {self.num.cp_normal_us:.4f} μs")
        if self.num.cp_extended_us:
            print(f"CP (extended) : {self.num.cp_extended_us:.4f} μs  (μ=2 only)")


# ─────────────────────────────────────────────────────────────────────────────
# OFDM Modulator / Demodulator
# ─────────────────────────────────────────────────────────────────────────────

class OFDMModulator:
    """
    OFDM Modulator and Demodulator — TS 38.211 compliant.

    Handles both modulation and demodulation in one class so the subcarrier
    index vectors, CP length, and FFT size are never out of sync between TX and RX.

    Constructor accepts either:
      (a) mu + n_prb  — derives cp_len from the NR numerology table, or
      (b) cp_len directly — for legacy / non-NR use (mu defaults to 1)

    Parameters
    ----------
    mu                  : NR numerology 0-4 (sets SCS and CP duration)
    n_fft               : FFT size, must be power of 2
    n_prb               : allocated PRBs (n_sc = n_prb * 12)
    cp_len              : override CP length in samples (ignores numerology CP if set)
    n_subcarriers       : override active subcarrier count (ignores n_prb if set)
    use_extended_cp     : use extended CP (μ=2 only, 4.17 μs)
    max_delay_spread_us : if > 0, raises ValueError when CP < this value
    """

    def __init__(
        self,
        mu:                  int   = 1,
        n_fft:               int   = 2048,
        n_prb:               int   = 52,
        cp_len:              Optional[int]   = None,
        n_subcarriers:       Optional[int]   = None,
        use_extended_cp:     bool  = False,
        max_delay_spread_us: float = 0.0,
    ):
        if mu not in NR_NUMEROLOGIES:
            raise ValueError(f"μ must be 0-4, got {mu}")
        assert (n_fft & (n_fft - 1)) == 0, "n_fft must be a power of 2"

        self.mu    = mu
        self.num   = NR_NUMEROLOGIES[mu]
        self.n_fft = n_fft
        self.n_prb = n_prb

        # Active subcarrier count
        self.n_sc = n_subcarriers if n_subcarriers is not None else n_prb * 12
        # Legacy alias used by ran_link_sim stack
        self.n_subcarriers = self.n_sc
        assert self.n_sc <= n_fft, f"n_sc={self.n_sc} exceeds n_fft={n_fft}"

        # Sample rate
        self.fs = self.num.scs_hz * n_fft    # Hz
        self.Ts = 1.0 / self.fs              # s/sample

        # CP length
        if cp_len is not None:
            self.cp_len = cp_len
        else:
            if use_extended_cp:
                if mu != 2:
                    raise ValueError("Extended CP is only defined for μ=2")
                cp_us = self.num.cp_extended_us
            else:
                cp_us = self.num.cp_normal_us
            self.cp_len = int(round(cp_us * 1e-6 * self.fs))

        self.cp_us      = self.cp_len / self.fs * 1e6   # actual CP in μs
        self.symbol_len = n_fft + self.cp_len

        # CP guard check
        if max_delay_spread_us > 0 and self.cp_us < max_delay_spread_us:
            raise ValueError(
                f"CP ({self.cp_us:.4f} μs) < max_delay_spread ({max_delay_spread_us} μs). "
                f"Increase n_fft, lower μ, or use extended CP."
            )

        # Precompute DC-centred subcarrier index vectors
        half = self.n_sc // 2
        self._sc_idx_pos = np.arange(1,          half + 1)   # bins 1..half
        self._sc_idx_neg = np.arange(n_fft - half, n_fft)    # bins N-half..N-1

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _map_subcarriers(self, symbols: np.ndarray) -> np.ndarray:
        """
        Map n_sc QAM symbols onto n_fft frequency bins.
        DC bin (index 0) is zeroed; guard bands are zeroed implicitly.
        """
        grid = np.zeros(self.n_fft, dtype=complex)
        half = self.n_sc // 2
        grid[self._sc_idx_pos] = symbols[half:]
        grid[self._sc_idx_neg] = symbols[:half]
        return grid

    def _unmap_subcarriers(self, freq: np.ndarray) -> np.ndarray:
        """Extract n_sc symbols from n_fft FFT bins (inverse of _map_subcarriers)."""
        half    = self.n_sc // 2
        symbols = np.empty(self.n_sc, dtype=complex)
        symbols[half:] = freq[self._sc_idx_pos]
        symbols[:half] = freq[self._sc_idx_neg]
        return symbols

    def _add_cp(self, time: np.ndarray) -> np.ndarray:
        """Prepend cyclic prefix: last cp_len samples → front."""
        return np.concatenate([time[-self.cp_len:], time])

    def _remove_cp(self, rx: np.ndarray) -> np.ndarray:
        """Strip cyclic prefix."""
        return rx[self.cp_len:]

    # ── TX path ───────────────────────────────────────────────────────────────

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Modulate one OFDM symbol.

        Parameters
        ----------
        symbols : (n_sc,) complex QAM symbols

        Returns
        -------
        tx : (n_fft + cp_len,) time-domain samples
        """
        assert symbols.shape == (self.n_sc,), \
            f"Expected ({self.n_sc},), got {symbols.shape}"
        freq = self._map_subcarriers(symbols)
        time = ifft(freq) * np.sqrt(self.n_fft)
        return self._add_cp(time)

    def modulate_slot(self, grid: np.ndarray) -> np.ndarray:
        """
        Modulate a full slot resource grid.

        Parameters
        ----------
        grid : (n_symb, n_sc) complex array

        Returns
        -------
        tx : (n_symb * symbol_len,) time-domain signal
        """
        return np.concatenate([self.modulate(grid[i]) for i in range(grid.shape[0])])

    def inject_dmrs(self, grid: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Overwrite DMRS symbol rows in a slot grid with pilot sequences.

        Parameters
        ----------
        grid : (n_symb, n_sc) complex array  — modified in place
        seed : DMRS sequence seed

        Returns
        -------
        grid : same array with DMRS rows filled
        """
        pilots = dmrs_pilot_sequence(self.n_sc, seed)
        for sym_idx in dmrs_symbol_indices(self.mu):
            if sym_idx < grid.shape[0]:
                grid[sym_idx] = pilots
        return grid

    # ── RX path ───────────────────────────────────────────────────────────────

    def demodulate(self, rx: np.ndarray) -> np.ndarray:
        """
        Demodulate one received OFDM symbol (includes CP removal).

        Parameters
        ----------
        rx : (n_fft + cp_len,) received samples

        Returns
        -------
        symbols : (n_sc,) complex symbols
        """
        assert len(rx) == self.symbol_len, \
            f"Expected {self.symbol_len} samples, got {len(rx)}"
        time = self._remove_cp(rx)
        freq = fft(time) / np.sqrt(self.n_fft)
        return self._unmap_subcarriers(freq)

    def demodulate_slot(self, rx: np.ndarray, n_symb: int = 14) -> np.ndarray:
        """
        Demodulate a full received slot.

        Parameters
        ----------
        rx     : (n_symb * symbol_len,) received samples
        n_symb : number of OFDM symbols in the slot

        Returns
        -------
        grid : (n_symb, n_sc) complex array
        """
        grid = np.zeros((n_symb, self.n_sc), dtype=complex)
        for i in range(n_symb):
            s       = i * self.symbol_len
            grid[i] = self.demodulate(rx[s: s + self.symbol_len])
        return grid

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Key parameters as a flat dictionary."""
        return {
            "mu":               self.mu,
            "scs_khz":          self.num.scs_khz,
            "n_fft":            self.n_fft,
            "n_prb":            self.n_prb,
            "n_sc":             self.n_sc,
            "cp_len_samples":   self.cp_len,
            "cp_us":            round(self.cp_us, 4),
            "fs_MHz":           round(self.fs / 1e6, 3),
            "symbol_len":       self.symbol_len,
            "slot_duration_us": round(self.num.slot_duration_us, 3),
        }

    def print_info(self):
        d = self.info()
        print(f"\n{'─'*48}")
        print(f"  OFDM Config  μ={d['mu']}  ({d['scs_khz']} kHz SCS)")
        print(f"{'─'*48}")
        for k, v in d.items():
            print(f"  {k:<22}: {v}")
        print(f"{'─'*48}")


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible OFDMDemodulator alias
# Kept so ran_link_sim imports that reference OFDMDemodulator still work.
# ─────────────────────────────────────────────────────────────────────────────

class OFDMDemodulator(OFDMModulator):
    """
    Thin alias for OFDMModulator — demodulate() and demodulate_slot() are
    already on OFDMModulator. This class exists purely for backward compatibility
    with any code that imports OFDMDemodulator separately.
    """

    def __init__(self, n_fft: int = 2048, cp_len: int = 144, n_subcarriers: int = 624):
        # Legacy positional interface used by the original ran_link_sim ofdm.py
        super().__init__(
            mu=1,
            n_fft=n_fft,
            cp_len=cp_len,
            n_subcarriers=n_subcarriers,
        )