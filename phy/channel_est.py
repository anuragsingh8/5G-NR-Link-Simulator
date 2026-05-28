"""
phy/channel_est.py — Channel Estimators
=========================================
3GPP TS 38.211 Section 7.4 (DMRS)

Estimators
----------
  LSEstimator    — Least Squares at DMRS pilots, linear freq interpolation
  LMMSEEstimator — Linear MMSE, exponential spatial correlation model
  SlotEstimator  — Full slot estimator: DMRS-based, freq + time interpolation

DMRS patterns (TS 38.211 §7.4.1.1)
  Type 1 comb-2: pilots on even subcarriers (default)
  Type 2 comb-4: pilots on subcarriers 0,2,6,8 within each PRB

Pilot symbols within a slot (Type A mapping):
  Single-symbol DMRS: symbols 2 and 11 (l0=2, additional=11)
  Double-symbol DMRS: symbols 2,3 and 11,12
"""

from __future__ import annotations
import numpy as np
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# DMRS sequence generation  (TS 38.211 §7.4.1.1)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dmrs_sequence(n_sc: int, seed: int = 42) -> np.ndarray:
    """
    Generate Gold-code DMRS sequence of length n_sc.
    Simplified: uses QPSK mapping of a PN sequence.
    Full spec uses Gold code initialised from slot/symbol/port indices.
    Returns (n_sc,) complex array, |d|=1.
    """
    rng  = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=2 * n_sc)
    return (1 - 2 * bits[:n_sc] + 1j * (1 - 2 * bits[n_sc:])) / np.sqrt(2)


def dmrs_pilot_subcarriers(
    n_sc:      int,
    dmrs_type: int = 1,
) -> np.ndarray:
    """
    Return subcarrier indices carrying DMRS pilots.

    Type 1 (comb-2): every even SC within the allocation → n_sc/2 pilots
    Type 2 (comb-4): SCs 0,2,6,8 per PRB → n_sc*4/12 pilots
    """
    if dmrs_type == 1:
        return np.arange(0, n_sc, 2)
    else:  # Type 2
        pilots = []
        for prb in range(n_sc // 12):
            base = prb * 12
            pilots.extend([base, base+2, base+6, base+8])
        return np.array(pilots)


def dmrs_symbol_positions(
    n_symb:      int  = 14,
    mapping:     Literal['typeA', 'typeB'] = 'typeA',
    additional:  bool = True,
) -> list[int]:
    """
    OFDM symbol indices within a slot that carry DMRS.

    Type A (data from symbol 0): first DMRS at symbol 2
    Type B (data from symbol 0): first DMRS at symbol 0

    Parameters
    ----------
    additional : include additional DMRS position (symbol 11 for typeA)
    """
    if mapping == 'typeA':
        pos = [2]
        if additional and n_symb == 14:
            pos.append(11)
    else:
        pos = [0]
        if additional:
            pos.append(4)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freq_interpolate(
    pilot_idx: np.ndarray,
    h_pilots:  np.ndarray,
    n_sc:      int,
) -> np.ndarray:
    """Linear interpolation from pilot positions to all n_sc subcarriers."""
    all_idx = np.arange(n_sc)
    return (
        np.interp(all_idx, pilot_idx, h_pilots.real)
        + 1j * np.interp(all_idx, pilot_idx, h_pilots.imag)
    )


def _time_interpolate(
    h_at_dmrs: np.ndarray,
    dmrs_syms: list[int],
    n_symb:    int,
) -> np.ndarray:
    """
    Linear interpolation of H across all n_symb symbols.

    Parameters
    ----------
    h_at_dmrs : (n_dmrs, n_sc) channel estimates at DMRS symbols
    dmrs_syms : symbol indices of DMRS (sorted)
    n_symb    : total symbols in slot

    Returns
    -------
    H_slot : (n_symb, n_sc) full-slot channel estimate
    """
    n_sc = h_at_dmrs.shape[1]
    H    = np.zeros((n_symb, n_sc), dtype=complex)

    if len(dmrs_syms) == 1:
        # Single DMRS: constant across slot
        H[:] = h_at_dmrs[0]
        return H

    sym_arr = np.array(dmrs_syms)
    for k in range(n_sc):
        H[:, k] = (
            np.interp(np.arange(n_symb), sym_arr, h_at_dmrs[:, k].real)
            + 1j * np.interp(np.arange(n_symb), sym_arr, h_at_dmrs[:, k].imag)
        )
    return H


# ─────────────────────────────────────────────────────────────────────────────
# LS Estimator
# ─────────────────────────────────────────────────────────────────────────────

class LSEstimator:
    """
    Least-Squares channel estimator from a single DMRS symbol.
    H_ls[k] = RX[k] / DMRS[k] at pilot positions, then linear interpolation.

    Parameters
    ----------
    n_subcarriers : active subcarriers
    dmrs_type     : 1 (comb-2) or 2 (comb-4)
    seed          : DMRS sequence seed
    """

    def __init__(
        self,
        n_subcarriers: int  = 624,
        dmrs_type:     int  = 1,
        seed:          int  = 42,
    ):
        self.n_sc      = n_subcarriers
        self.pilot_idx = dmrs_pilot_subcarriers(n_subcarriers, dmrs_type)
        seq            = generate_dmrs_sequence(n_subcarriers, seed)
        self.pilots    = seq[self.pilot_idx]

    def estimate(self, rx_symbol: np.ndarray) -> np.ndarray:
        """
        Estimate H from one received DMRS OFDM symbol.

        Parameters
        ----------
        rx_symbol : (n_sc,) received symbols — DMRS symbol row from slot grid

        Returns
        -------
        H_est : (n_sc,) interpolated channel estimate
        """
        h_p = rx_symbol[self.pilot_idx] / (self.pilots + 1e-12)
        return _freq_interpolate(self.pilot_idx, h_p, self.n_sc)


# ─────────────────────────────────────────────────────────────────────────────
# LMMSE Estimator
# ─────────────────────────────────────────────────────────────────────────────

class LMMSEEstimator:
    """
    Linear MMSE channel estimator.

    H_lmmse = R_hh (R_hh + σ² I)^{-1} H_ls

    R_hh modelled as exponentially decaying across pilot separation.

    Parameters
    ----------
    n_subcarriers : active subcarriers
    snr_db        : regularisation SNR
    coherence_bw  : coherence bandwidth in subcarrier spacings
    dmrs_type     : 1 or 2
    seed          : DMRS seed
    """

    def __init__(
        self,
        n_subcarriers: int   = 624,
        snr_db:        float = 15.0,
        coherence_bw:  float = 50.0,
        dmrs_type:     int   = 1,
        seed:          int   = 42,
    ):
        self.n_sc      = n_subcarriers
        self.pilot_idx = dmrs_pilot_subcarriers(n_subcarriers, dmrs_type)
        seq            = generate_dmrs_sequence(n_subcarriers, seed)
        self.pilots    = seq[self.pilot_idx]
        self.n_pilots  = len(self.pilot_idx)

        d        = np.abs(np.subtract.outer(self.pilot_idx, self.pilot_idx))
        R_hh     = np.exp(-d / coherence_bw).astype(complex)
        snr_lin  = 10 ** (snr_db / 10)
        self._W  = R_hh @ np.linalg.inv(R_hh + np.eye(self.n_pilots) / snr_lin)

    def estimate(self, rx_symbol: np.ndarray) -> np.ndarray:
        """
        LMMSE estimate from one received DMRS OFDM symbol.

        Parameters
        ----------
        rx_symbol : (n_sc,) received symbol

        Returns
        -------
        H_est : (n_sc,) complex channel estimate
        """
        h_ls    = rx_symbol[self.pilot_idx] / (self.pilots + 1e-12)
        h_lmmse = self._W @ h_ls
        return _freq_interpolate(self.pilot_idx, h_lmmse, self.n_sc)


# ─────────────────────────────────────────────────────────────────────────────
# Full Slot Estimator (DMRS-based, freq + time interpolation)
# ─────────────────────────────────────────────────────────────────────────────

class SlotEstimator:
    """
    Full-slot channel estimator.

    Processes a complete (n_symb × n_sc) received slot grid:
      1. Extract received symbols at DMRS positions
      2. Divide by known DMRS sequence → LS estimate at pilot SCs
      3. LMMSE smoothing across pilot subcarriers
      4. Interpolate across all subcarriers (frequency domain)
      5. Interpolate across all symbols (time domain)

    This is the production estimation chain for OFDM with DMRS.

    Parameters
    ----------
    n_sc          : active subcarriers
    n_symb        : symbols per slot (14 for normal CP)
    snr_db        : regularisation SNR for LMMSE
    dmrs_type     : 1 (comb-2) or 2 (comb-4)
    dmrs_mapping  : 'typeA' or 'typeB'
    use_lmmse     : True = LMMSE, False = plain LS
    seed          : DMRS sequence seed
    """

    def __init__(
        self,
        n_sc:         int   = 624,
        n_symb:       int   = 14,
        snr_db:       float = 15.0,
        dmrs_type:    int   = 1,
        dmrs_mapping: str   = 'typeA',
        use_lmmse:    bool  = True,
        seed:         int   = 42,
    ):
        self.n_sc        = n_sc
        self.n_symb      = n_symb
        self.dmrs_syms   = dmrs_symbol_positions(n_symb, dmrs_mapping)
        self.pilot_idx   = dmrs_pilot_subcarriers(n_sc, dmrs_type)
        self.n_pilots    = len(self.pilot_idx)

        # DMRS sequences per symbol (same sequence, different seeds could model ports)
        self.dmrs_seq    = {}
        for s in self.dmrs_syms:
            seq = generate_dmrs_sequence(n_sc, seed + s)
            self.dmrs_seq[s] = seq[self.pilot_idx]

        # LMMSE weight matrix
        if use_lmmse:
            d       = np.abs(np.subtract.outer(self.pilot_idx, self.pilot_idx))
            R       = np.exp(-d / max(50.0, n_sc // 12)).astype(complex)
            snr_lin = 10 ** (snr_db / 10)
            self._W = R @ np.linalg.inv(R + np.eye(self.n_pilots) / snr_lin)
        else:
            self._W = np.eye(self.n_pilots)

    def generate_slot_dmrs(self) -> dict[int, np.ndarray]:
        """
        Return known DMRS symbols for TX grid injection.
        {symbol_idx: (n_sc,) array, zeros at non-pilot SCs}
        """
        out = {}
        for s, pilots in self.dmrs_seq.items():
            row = np.zeros(self.n_sc, dtype=complex)
            row[self.pilot_idx] = pilots
            out[s] = row
        return out

    def estimate_slot(self, rx_grid: np.ndarray) -> np.ndarray:
        """
        Estimate H across full slot.

        Parameters
        ----------
        rx_grid : (n_symb, n_sc) received slot grid

        Returns
        -------
        H_slot : (n_symb, n_sc) complex channel estimate per symbol per SC
        """
        h_at_dmrs = np.zeros((len(self.dmrs_syms), self.n_sc), dtype=complex)

        for i, sym in enumerate(self.dmrs_syms):
            rx_pilots  = rx_grid[sym, self.pilot_idx]
            h_ls       = rx_pilots / (self.dmrs_seq[sym] + 1e-12)
            h_smooth   = self._W @ h_ls   # LMMSE or identity
            h_at_dmrs[i] = _freq_interpolate(self.pilot_idx, h_smooth, self.n_sc)

        return _time_interpolate(h_at_dmrs, self.dmrs_syms, self.n_symb)