"""
phy/modulation.py — Bit-to-Symbol Mapper, Demapper, and LLR Computation

This module implements digital modulation and demodulation utilities used in
wireless communication systems such as LTE and 5G NR.

Supported modulation schemes:
  • QPSK
  • 16-QAM
  • 64-QAM

Features:
  • Bit-to-symbol mapping
  • Hard-decision demapping
  • Soft LLR generation using Max-Log-MAP approximation

Reference:
  3GPP TS 38.211 Section 5.1
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Gray-coded constellation tables
# ─────────────────────────────────────────────────────────────────────────────
#
# These functions generate normalized constellation points for different
# modulation orders.
#
# Gray coding is used so that neighboring constellation points differ by only
# one bit. This minimizes bit errors caused by small noise perturbations.
#
# The constellation average power is normalized to 1 so that:
#
#   E[|s|²] = 1
#
# This simplifies SNR calculations and keeps modulation schemes comparable.
# ─────────────────────────────────────────────────────────────────────────────


def _qpsk_table() -> np.ndarray:
    """
    Generate normalized QPSK constellation.

    QPSK:
      • 2 bits per symbol
      • 4 constellation points

    Returns
    -------
    np.ndarray
        Complex QPSK constellation points.
    """

    # Standard QPSK constellation:
    #
    #   (+1,+1)
    #   (-1,+1)
    #   (+1,-1)
    #   (-1,-1)
    #
    # Division by sqrt(2) normalizes average symbol energy to 1.
    return np.array([
        1 + 1j,
        -1 + 1j,
        1 - 1j,
        -1 - 1j
    ]) / np.sqrt(2)


def _qam16_table() -> np.ndarray:
    """
    Generate normalized 16-QAM constellation.

    16-QAM:
      • 4 bits per symbol
      • 16 constellation points

    Returns
    -------
    np.ndarray
        Complex 16-QAM constellation.
    """

    # Possible PAM amplitude levels for each dimension.
    pts = np.array([-3, -1, 1, 3], dtype=float)

    # Build 2D QAM constellation using Cartesian product:
    #
    #   real_part + j * imag_part
    #
    # Example:
    #   (-3 - 3j)
    #   (-3 - 1j)
    #   ...
    table = np.array([
        c_r + 1j * c_i
        for c_r in pts
        for c_i in pts
    ])

    # Normalize average symbol power.
    #
    # For 16-QAM:
    #   average energy = 10
    return table / np.sqrt(10)


def _qam64_table() -> np.ndarray:
    """
    Generate normalized 64-QAM constellation.

    64-QAM:
      • 6 bits per symbol
      • 64 constellation points

    Returns
    -------
    np.ndarray
        Complex 64-QAM constellation.
    """

    # PAM levels for 64-QAM.
    pts = np.array(
        [-7, -5, -3, -1, 1, 3, 5, 7],
        dtype=float
    )

    # Generate all complex constellation points.
    table = np.array([
        c_r + 1j * c_i
        for c_r in pts
        for c_i in pts
    ])

    # Normalize average energy.
    #
    # For 64-QAM:
    #   average energy = 42
    return table / np.sqrt(42)


# Dictionary mapping modulation order → constellation table.
#
# Key meanings:
#   2 → QPSK
#   4 → 16-QAM
#   6 → 64-QAM
_CONSTELLATIONS: dict[int, np.ndarray] = {
    2: _qpsk_table(),
    4: _qam16_table(),
    6: _qam64_table(),
}


# Bits carried per symbol for each modulation type.
_BITS_PER_SYMBOL: dict[int, int] = {
    2: 2,
    4: 4,
    6: 6,
}


def _bits_to_index(bits: np.ndarray, m: int) -> np.ndarray:
    """
    Convert groups of bits into constellation indices.

    Example:
      bits = [1,0,1,1]
      m = 2

      Groups:
        [1,0] → 2
        [1,1] → 3

    Parameters
    ----------
    bits : np.ndarray
        Flat binary bit array.

    m : int
        Bits per symbol.

    Returns
    -------
    np.ndarray
        Integer constellation indices.
    """

    # Reshape bit stream into rows of m bits.
    #
    # Example:
    #   [1,0,1,1]
    #
    # becomes:
    #   [[1,0],
    #    [1,1]]
    bits = bits.reshape(-1, m)

    # Allocate integer index array.
    idx = np.zeros(len(bits), dtype=int)

    # Convert binary rows into decimal indices.
    #
    # Example:
    #   [1,0] → 2
    #
    # Uses MSB-first ordering.
    for i in range(m):

        # Add weighted bit contribution:
        #
        #   bit × 2^(position)
        idx += bits[:, i] * (2 ** (m - 1 - i))

    return idx


class Mapper:
    """
    Digital modulation mapper.

    Converts binary bits into complex modulation symbols using:
      • QPSK
      • 16-QAM
      • 64-QAM

    Parameters
    ----------
    modulation_order : int
        Bits per symbol:
          2 → QPSK
          4 → 16-QAM
          6 → 64-QAM
    """

    def __init__(self, modulation_order: int = 2):
        """
        Initialize modulation mapper.
        """

        # Validate modulation order before continuing.
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported Qm={modulation_order}. "
                f"Use 2, 4, or 6."
            )

        # Store modulation order.
        self.Qm = modulation_order

        # Load corresponding constellation table.
        self.constellation = _CONSTELLATIONS[modulation_order]

    def map(self, bits: np.ndarray) -> np.ndarray:
        """
        Convert binary bits into complex modulation symbols.

        Parameters
        ----------
        bits : np.ndarray
            Flat binary array.

            Length must be divisible by Qm.

        Returns
        -------
        np.ndarray
            Complex modulation symbols.
        """

        # Ensure the bit stream length aligns with the modulation order.
        #
        # Example:
        #   QPSK requires multiples of 2 bits.
        assert len(bits) % self.Qm == 0, (
            f"Bit length {len(bits)} not divisible by Qm={self.Qm}"
        )

        # Convert groups of bits into constellation indices.
        idx = _bits_to_index(bits, self.Qm)

        # Look up corresponding constellation symbols.
        return self.constellation[idx]


class Demapper:
    """
    Hard-decision demapper.

    Converts received complex symbols back into binary bits by selecting the
    nearest constellation point.

    This is called "hard decision" because:
      • only one bit decision is produced,
      • confidence information is discarded.
    """

    def __init__(self, modulation_order: int = 2):
        """
        Initialize demapper.
        """

        # Validate modulation order.
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported Qm={modulation_order}."
            )

        # Store modulation order.
        self.Qm = modulation_order

        # Load constellation table.
        self.constellation = _CONSTELLATIONS[modulation_order]

        # Number of constellation points:
        #
        #   2^Qm
        self.n_points = 2 ** modulation_order

    def demap(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision demapping.

        Parameters
        ----------
        symbols : np.ndarray
            Complex received symbols.

        Returns
        -------
        np.ndarray
            Recovered hard-decision bits.
        """

        # Compute Euclidean distance between every received symbol and every
        # constellation point.
        #
        # Shape:
        #   (N_symbols, constellation_size)
        dist = np.abs(
            symbols[:, np.newaxis]
            - self.constellation[np.newaxis, :]
        ) ** 2

        # Select nearest constellation point for each received symbol.
        idx = np.argmin(dist, axis=1)

        # Convert integer indices into binary representation.
        #
        # unpackbits produces 8 bits per uint8, so only the last Qm bits are
        # retained.
        bits_matrix = np.unpackbits(
            idx.astype(np.uint8),
            bitorder='big'
        ).reshape(-1, 8)[:, -self.Qm:]

        # Flatten bit matrix into 1D bit stream.
        return bits_matrix.ravel()


class LLRComputer:
    """
    Soft-decision LLR computation using Max-Log-MAP approximation.

    LLR = Log-Likelihood Ratio.

    LLR indicates how confident the receiver is about each bit.

    Interpretation:
      • Positive LLR:
            Bit likely equals 0

      • Negative LLR:
            Bit likely equals 1

      • Large magnitude:
            High confidence

      • Small magnitude:
            Low confidence

    Max-Log-MAP approximation:

        LLR(b_k) =
            max_{s:b_k=0} (-|y-s|²/σ²)
            -
            max_{s:b_k=1} (-|y-s|²/σ²)

    Parameters
    ----------
    modulation_order : int
        Bits per symbol.

    snr_db : float
        Operating SNR used for noise variance estimation.
    """

    def __init__(
        self,
        modulation_order: int = 2,
        snr_db: float = 15.0,
    ):
        """
        Initialize LLR computer.
        """

        # Validate modulation order.
        if modulation_order not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported Qm={modulation_order}."
            )

        # Store modulation order.
        self.Qm = modulation_order

        # Load modulation constellation.
        self.constellation = _CONSTELLATIONS[modulation_order]

        # Total constellation size:
        #
        #   M = 2^Qm
        self.n_points = 2 ** modulation_order

        # Approximate noise variance from SNR.
        self.sigma2 = 10 ** (-snr_db / 10)

        # Precompute bit labels for every constellation point.
        #
        # Example for QPSK:
        #
        #   0 → 00
        #   1 → 01
        #   2 → 10
        #   3 → 11
        #
        # Shape:
        #   (constellation_size, Qm)
        self._labels = np.array([
            list(map(
                int,
                format(i, f'0{modulation_order}b')
            ))
            for i in range(self.n_points)
        ])

    def compute(
        self,
        symbols: np.ndarray,
        sigma2: float | None = None,
    ) -> np.ndarray:
        """
        Compute soft LLR values for received symbols.

        Parameters
        ----------
        symbols : np.ndarray
            Complex received symbols.

        sigma2 : float | None
            Optional noise variance override.

        Returns
        -------
        np.ndarray
            Soft LLR values.

            Positive:
                bit likely = 0

            Negative:
                bit likely = 1
        """

        # Use provided noise variance if available.
        s2 = sigma2 if sigma2 is not None else self.sigma2

        # Number of received symbols.
        N = len(symbols)

        # Allocate output LLR array.
        #
        # One LLR value per transmitted bit.
        llrs = np.zeros(N * self.Qm)

        # Compute Euclidean distance between received symbols and all
        # constellation points.
        #
        # Shape:
        #   (N, constellation_size)
        dist = np.abs(
            symbols[:, np.newaxis]
            - self.constellation[np.newaxis, :]
        ) ** 2

        # Compute LLR independently for each bit position.
        for k in range(self.Qm):

            # Indices of constellation points where bit k = 0.
            idx0 = np.where(
                self._labels[:, k] == 0
            )[0]

            # Indices where bit k = 1.
            idx1 = np.where(
                self._labels[:, k] == 1
            )[0]

            # Best metric among symbols corresponding to bit=0.
            #
            # Max-Log approximation:
            #   max(-distance/sigma²)
            #
            # Equivalent to:
            #   minimum Euclidean distance
            min0 = dist[:, idx0].min(axis=1)

            # Best metric among symbols corresponding to bit=1.
            min1 = dist[:, idx1].min(axis=1)

            # Compute LLR:
            #
            #   positive → bit likely 0
            #   negative → bit likely 1
            #
            # Larger magnitude means higher confidence.
            llrs[k::self.Qm] = (min1 - min0) / s2

        # Return soft-decision bit metrics.
        return llrs