"""
phy/modulation.py — Bit-to-Symbol Mapper, Demapper, LLR Computer
==================================================================
Supports: BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM with Gray coding
3GPP TS 38.211 Section 5.1

Modulation order (Qm):
  1 = BPSK   (special case: single-axis)
  2 = QPSK
  4 = 16-QAM
  6 = 64-QAM
  8 = 256-QAM  ← new
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Gray-coded constellation tables
# ─────────────────────────────────────────────────────────────────────────────

def _bpsk_table() -> np.ndarray:
    """BPSK: 1 bit/symbol, normalised E[|s|²]=1"""
    return np.array([-1.0 + 0j, 1.0 + 0j])


def _qpsk_table() -> np.ndarray:
    """QPSK: 2 bits/symbol, normalised"""
    return np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)


def _qam16_table() -> np.ndarray:
    """16-QAM: 4 bits/symbol, Gray-coded, normalised"""
    pts = np.array([-3, -1, 1, 3], dtype=float)
    table = np.array([r + 1j*i for r in pts for i in pts])
    return table / np.sqrt(10)


def _qam64_table() -> np.ndarray:
    """64-QAM: 6 bits/symbol, Gray-coded, normalised"""
    pts = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
    table = np.array([r + 1j*i for r in pts for i in pts])
    return table / np.sqrt(42)


def _qam256_table() -> np.ndarray:
    """256-QAM: 8 bits/symbol, Gray-coded, normalised"""
    pts = np.array([-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], dtype=float)
    table = np.array([r + 1j*i for r in pts for i in pts])
    return table / np.sqrt(170)


_CONSTELLATIONS: dict[int, np.ndarray] = {
    1: _bpsk_table(),
    2: _qpsk_table(),
    4: _qam16_table(),
    6: _qam64_table(),
    8: _qam256_table(),
}

_MOD_NAMES: dict[int, str] = {
    1: 'BPSK', 2: 'QPSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'
}


def constellation(qm: int) -> np.ndarray:
    """Return normalised Gray-coded constellation for modulation order Qm."""
    if qm not in _CONSTELLATIONS:
        raise ValueError(f"Unsupported Qm={qm}. Use 1 (BPSK), 2, 4, 6, or 8.")
    return _CONSTELLATIONS[qm]


def mod_name(qm: int) -> str:
    return _MOD_NAMES.get(qm, f'{2**qm}-QAM')


# ─────────────────────────────────────────────────────────────────────────────
# Bit → index helper
# ─────────────────────────────────────────────────────────────────────────────

def _bits_to_index(bits: np.ndarray, m: int) -> np.ndarray:
    """Convert (N*m,) bit array to (N,) symbol indices."""
    b = bits.reshape(-1, m).astype(np.int32)
    powers = 2 ** np.arange(m - 1, -1, -1, dtype=np.int32)
    return (b * powers).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────────────────────────────────────

class Mapper:
    """
    Map bit stream → complex QAM symbols.

    Parameters
    ----------
    modulation_order : Qm — 1 (BPSK), 2 (QPSK), 4 (16-QAM), 6 (64-QAM), 8 (256-QAM)
    """

    def __init__(self, modulation_order: int = 2):
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported Qm={modulation_order}. Use 1, 2, 4, 6, or 8."
            )
        self.Qm            = modulation_order
        self.constellation = _CONSTELLATIONS[modulation_order]
        self.bits_per_sym  = max(1, modulation_order)

    def map(self, bits: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        bits : (N,) binary array; N must be divisible by Qm

        Returns
        -------
        symbols : (N // Qm,) complex array
        """
        m = self.bits_per_sym
        assert len(bits) % m == 0, \
            f"Bit length {len(bits)} not divisible by Qm={m}"
        if m == 1:
            # BPSK: bit 0 → -1, bit 1 → +1
            return self.constellation[bits.astype(np.int32) & 1]
        idx = _bits_to_index(bits, m)
        return self.constellation[idx % len(self.constellation)]


# ─────────────────────────────────────────────────────────────────────────────
# Demapper (hard decision)
# ─────────────────────────────────────────────────────────────────────────────

class Demapper:
    """
    Hard-decision demapper: nearest constellation point → bits.
    """

    def __init__(self, modulation_order: int = 2):
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(f"Unsupported Qm={modulation_order}.")
        self.Qm            = modulation_order
        self.constellation = _CONSTELLATIONS[modulation_order]

    def demap(self, symbols: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        symbols : (N,) complex received symbols

        Returns
        -------
        bits : (N * Qm,) hard-decision bits
        """
        m    = max(1, self.Qm)
        dist = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :]) ** 2
        idx  = np.argmin(dist, axis=1)
        if m == 1:
            return idx.astype(np.uint8)
        bits_mat = np.unpackbits(
            idx.astype(np.uint8), bitorder='big'
        ).reshape(-1, 8)[:, -m:]
        return bits_mat.ravel().astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# LLR Computer (Max-Log-MAP)
# ─────────────────────────────────────────────────────────────────────────────

class LLRComputer:
    """
    Soft LLR computation via Max-Log-MAP approximation.

    LLR(b_k) = min_{s: b_k=1} |y-s|²/σ²  −  min_{s: b_k=0} |y-s|²/σ²

    Supports adaptive sigma: per-call sigma2 override allows the equaliser
    to supply post-equalisation noise variance for accurate soft values.

    Parameters
    ----------
    modulation_order : Qm
    snr_db           : default operating SNR (used when sigma2 not supplied)
    """

    def __init__(self, modulation_order: int = 2, snr_db: float = 15.0):
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(f"Unsupported Qm={modulation_order}.")
        self.Qm            = modulation_order
        self.constellation = _CONSTELLATIONS[modulation_order]
        self.n_points      = len(self.constellation)
        self.sigma2        = 10 ** (-snr_db / 10)

        m = max(1, modulation_order)
        # Precompute Gray bit labels: (n_points, m)
        self._labels = np.array([
            list(map(int, format(i, f'0{m}b')))
            for i in range(self.n_points)
        ])
        # Precompute index sets for each bit position
        self._idx0 = [np.where(self._labels[:, k] == 0)[0] for k in range(m)]
        self._idx1 = [np.where(self._labels[:, k] == 1)[0] for k in range(m)]

    def compute(
        self,
        symbols: np.ndarray,
        sigma2:  float | None = None,
    ) -> np.ndarray:
        """
        Compute soft LLRs for received symbols.

        Parameters
        ----------
        symbols : (N,) complex received symbols
        sigma2  : noise variance per real dimension (overrides default)

        Returns
        -------
        llrs : (N * Qm,) float array  (positive = bit 0 likely)
        """
        s2   = sigma2 if sigma2 is not None else self.sigma2
        m    = max(1, self.Qm)
        N    = len(symbols)
        llrs = np.zeros(N * m)

        # Squared Euclidean distances: (N, n_points)
        dist = np.abs(
            symbols[:, np.newaxis] - self.constellation[np.newaxis, :]
        ) ** 2

        for k in range(m):
            min0 = dist[:, self._idx0[k]].min(axis=1)
            min1 = dist[:, self._idx1[k]].min(axis=1)
            llrs[k::m] = (min1 - min0) / s2

        return llrs

    def compute_with_channel(
        self,
        symbols:  np.ndarray,
        H_mag_sq: np.ndarray,
        noise_var: float,
    ) -> np.ndarray:
        """
        LLR computation with known per-subcarrier channel gain.
        Scales noise variance by channel power: σ²_eff = noise_var / |H|²

        Parameters
        ----------
        symbols   : (N,) equalised complex symbols
        H_mag_sq  : (N,) per-subcarrier |H|² (channel power)
        noise_var : noise variance σ²

        Returns
        -------
        llrs : (N * Qm,) float array
        """
        m    = max(1, self.Qm)
        N    = len(symbols)
        llrs = np.zeros(N * m)
        dist = np.abs(
            symbols[:, np.newaxis] - self.constellation[np.newaxis, :]
        ) ** 2
        # Per-subcarrier effective sigma²
        s2_per_sc = noise_var / (H_mag_sq + 1e-12)   # (N,)

        for k in range(m):
            min0 = dist[:, self._idx0[k]].min(axis=1)
            min1 = dist[:, self._idx1[k]].min(axis=1)
            llrs[k::m] = (min1 - min0) / s2_per_sc
        return llrs