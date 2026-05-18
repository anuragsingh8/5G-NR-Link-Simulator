"""
mac/csi.py — Channel State Information Utilities

This module implements CSI (Channel State Information) related processing used
in LTE and 5G NR systems.

Implemented components:
  • CQIComputer
      Converts measured SINR into CQI reports.

  • PMICodebook
      Selects beamforming / precoding matrix indicators.

  • RISelector
      Chooses optimal MIMO transmission rank.

Reference:
  3GPP TS 38.214 Section 5.2
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SINR → CQI Mapping Table
# ─────────────────────────────────────────────────────────────────────────────
#
# This table approximates:
#   3GPP TS 38.214 Table 5.2.2.1-3
#
# Each entry contains:
#   (CQI index, minimum required SINR in dB)
#
# Interpretation:
#   • Higher CQI requires better SINR.
#   • CQI determines:
#         - modulation order,
#         - coding rate,
#         - throughput,
#         - robustness.
#
# CQI 0 means:
#   "Channel quality too poor for reliable transmission."
# ─────────────────────────────────────────────────────────────────────────────

_CQI_SINR_THRESHOLDS = [
    (1, -6.7),
    (2, -4.7),
    (3, -2.3),
    (4, 0.2),
    (5, 2.4),
    (6, 4.3),
    (7, 5.9),
    (8, 8.1),
    (9, 10.3),
    (10, 11.7),
    (11, 14.1),
    (12, 16.3),
    (13, 18.7),
    (14, 21.0),
    (15, 22.7),
]


class CQIComputer:
    """
    CQI computation utility.

    CQI = Channel Quality Indicator.

    This class converts measured SINR values into CQI reports used by the
    scheduler and link adaptation algorithms.

    Supported modes:
      • wideband CQI
      • subband CQI
      • effective SINR estimation

    Parameters
    ----------
    n_subbands : int
        Number of CSI subbands.

        n_subbands = 1:
            Wideband CQI only.

        n_subbands > 1:
            Frequency-selective subband reporting.
    """

    def __init__(self, n_subbands: int = 1):
        """
        Initialize CQI computer.
        """

        # Store number of CSI subbands.
        self.n_subbands = n_subbands

        # Store SINR threshold table.
        self._thresholds = _CQI_SINR_THRESHOLDS

    def sinr_to_cqi(self, sinr_db: float) -> int:
        """
        Convert SINR into CQI index.

        Parameters
        ----------
        sinr_db : float
            Measured SINR in dB.

        Returns
        -------
        int
            CQI index in range 0–15.
        """

        # Default CQI:
        #   0 = channel out of range / unusable
        cqi = 0

        # Traverse thresholds in ascending order.
        #
        # The highest threshold satisfied becomes the selected CQI.
        for c, thresh in self._thresholds:

            # If SINR exceeds this threshold, update CQI.
            if sinr_db >= thresh:
                cqi = c

        # Return final CQI.
        return cqi

    def compute_wideband_cqi(
        self,
        sinr_per_sc: np.ndarray,
    ) -> int:
        """
        Compute wideband CQI from per-subcarrier SINR values.

        Parameters
        ----------
        sinr_per_sc : np.ndarray
            SINR values per subcarrier in dB.

        Returns
        -------
        int
            Wideband CQI index.
        """

        # Compute average SINR across all subcarriers.
        mean_sinr_db = float(np.mean(sinr_per_sc))

        # Convert average SINR into CQI.
        return self.sinr_to_cqi(mean_sinr_db)

    def compute_subband_cqi(
        self,
        sinr_per_sc: np.ndarray,
    ) -> np.ndarray:
        """
        Compute CQI independently for each frequency subband.

        Parameters
        ----------
        sinr_per_sc : np.ndarray
            Per-subcarrier SINR values in dB.

        Returns
        -------
        np.ndarray
            CQI value for each subband.
        """

        # Split subcarriers into approximately equal subbands.
        subbands = np.array_split(
            sinr_per_sc,
            self.n_subbands,
        )

        # Compute average SINR and CQI for each subband separately.
        return np.array([
            self.sinr_to_cqi(float(np.mean(sb)))
            for sb in subbands
        ])

    def effective_sinr(self, sinr_per_sc: np.ndarray) -> float:
        """
        Compute effective SINR using EESM.

        EESM:
            Exponential Effective SINR Mapping

        EESM compresses frequency-selective SINR values into one equivalent
        SINR value that predicts BLER performance.

        Formula:

            SINR_eff =
                -β log(1/N Σ exp(-SINR_k / β))

        Parameters
        ----------
        sinr_per_sc : np.ndarray
            SINR values per subcarrier in dB.

        Returns
        -------
        float
            Effective SINR in dB.
        """

        # Beta parameter controls mapping sensitivity.
        #
        # Different modulation/coding schemes typically use different beta
        # calibration values.
        beta = 1.0

        # Convert SINR values from dB to linear scale.
        sinr_lin = 10 ** (sinr_per_sc / 10)

        # Compute exponential averaging.
        #
        # Poor SINR values dominate the result, which better predicts coding
        # performance compared to simple averaging.
        eff = -beta * np.log(
            np.mean(
                np.exp(-sinr_lin / beta)
            )
        )

        # Convert effective SINR back to dB.
        #
        # max(..., 1e-10) prevents log10(0).
        return float(
            10 * np.log10(max(eff, 1e-10))
        )


class PMICodebook:
    """
    PMI codebook wrapper.

    PMI = Precoding Matrix Indicator.

    This class wraps the beamforming codebook implementation and provides
    CSI-oriented PMI selection functionality.

    Supported features:
      • beamforming matrix retrieval,
      • PMI selection,
      • post-precoding SINR estimation.
    """

    def __init__(
        self,
        n_tx: int = 4,
        n_layers: int = 1,
    ):
        """
        Initialize PMI codebook.
        """

        # Import locally to avoid circular import dependency.
        from phy.mimo import BeamformingMatrix

        # Create underlying beamforming codebook.
        self._bf = BeamformingMatrix(
            n_tx,
            n_layers,
        )

        # Store antenna count.
        self.n_tx = n_tx

        # Store spatial rank / number of layers.
        self.n_layers = n_layers

    @property
    def codebook_size(self) -> int:
        """
        Return number of available PMI entries.
        """

        return self._bf.codebook_size

    def get_precoder(self, pmi: int) -> np.ndarray:
        """
        Retrieve precoding matrix for a PMI index.

        Parameters
        ----------
        pmi : int
            Precoding Matrix Indicator.

        Returns
        -------
        np.ndarray
            Beamforming matrix.
        """

        return self._bf.get_precoder(pmi)

    def select_best_pmi(self, H: np.ndarray) -> int:
        """
        Select best PMI for the current channel.

        Parameters
        ----------
        H : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        int
            Selected PMI index.
        """

        # Delegate PMI search to beamforming codebook implementation.
        return self._bf.select_pmi(H)

    def compute_pmi_sinr(
        self,
        H: np.ndarray,
        pmi: int,
        snr_db: float,
    ) -> float:
        """
        Compute post-precoding SINR for a selected PMI.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix.

        pmi : int
            Precoding Matrix Indicator.

        snr_db : float
            Operating SNR in dB.

        Returns
        -------
        float
            Average post-precoding SINR in dB.
        """

        # Retrieve selected precoding matrix.
        W = self.get_precoder(pmi)

        # Compute effective precoded channel:
        #
        #   H_eff = H * W
        #
        # Shape:
        #   (n_rx, n_layers)
        HW = H @ W

        # Singular values represent effective spatial channel strength after
        # beamforming.
        sv = np.linalg.svd(HW, compute_uv=False)

        # Convert SNR to linear scale.
        snr_lin = 10 ** (snr_db / 10)

        # Approximate average post-beamforming SINR.
        #
        # Stronger singular values imply stronger spatial streams.
        sinr_lin = (
            snr_lin
            * np.mean(sv ** 2)
            / self.n_tx
        )

        # Convert back to dB.
        return float(
            10 * np.log10(max(sinr_lin, 1e-10))
        )


class RISelector:
    """
    Rank Indicator selector.

    RI = Rank Indicator.

    RI determines how many spatial layers should be transmitted in MIMO.

    Strategy:
      • evaluate feasible spatial ranks,
      • ensure every layer exceeds minimum SINR,
      • choose rank with highest capacity.

    Higher rank:
      • higher throughput,
      • but requires stronger channel conditions.
    """

    def __init__(
        self,
        n_tx: int = 4,
        n_rx: int = 2,
        min_sinr_db: float = 3.0,
    ):
        """
        Initialize RI selector.
        """

        # Store transmit antenna count.
        self.n_tx = n_tx

        # Store receive antenna count.
        self.n_rx = n_rx

        # Minimum acceptable per-layer SINR.
        self.min_sinr_db = min_sinr_db

        # Maximum possible rank:
        #
        # limited by smaller antenna dimension.
        self.max_rank = min(n_tx, n_rx)

    def select_rank(
        self,
        H: np.ndarray,
        snr_db: float,
    ) -> int:
        """
        Select optimal spatial rank.

        Parameters
        ----------
        H : np.ndarray
            MIMO channel matrix.

        snr_db : float
            Operating SNR.

        Returns
        -------
        int
            Selected rank indicator.
        """

        # Singular values describe available spatial streams.
        #
        # Returned in descending order:
        #   strongest stream first.
        sv = np.linalg.svd(H, compute_uv=False)

        # Convert SNR to linear scale.
        snr_lin = 10 ** (snr_db / 10)

        # Convert minimum SINR threshold to linear scale.
        min_sinr_lin = 10 ** (self.min_sinr_db / 10)

        # Default to rank-1 transmission.
        best_ri = 1

        # Best capacity observed so far.
        best_cap = 0.0

        # Evaluate every possible rank.
        for r in range(1, self.max_rank + 1):

            # Use strongest r singular values.
            sv_r = sv[:r]

            # Compute per-layer SINR assuming equal power allocation.
            #
            # Each stream receives:
            #   total_power / n_tx
            sinr_r = (
                snr_lin
                * sv_r ** 2
                / self.n_tx
            )

            # If any layer becomes too weak, stop increasing rank.
            if np.any(sinr_r < min_sinr_lin):
                break

            # Compute approximate MIMO capacity:
            #
            #   Σ log2(1 + SINR_i)
            cap = float(
                np.sum(
                    np.log2(1 + sinr_r)
                )
            )

            # Keep rank with highest capacity.
            if cap > best_cap:
                best_cap = cap
                best_ri = r

        # Return selected spatial rank.
        return best_ri

    def select_with_pmi(
        self,
        H: np.ndarray,
        snr_db: float,
    ) -> tuple[int, int]:
        """
        Joint RI and PMI selection.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix.

        snr_db : float
            Operating SNR.

        Returns
        -------
        tuple[int, int]
            (RI, PMI)
        """

        # First determine optimal transmission rank.
        ri = self.select_rank(H, snr_db)

        # Build PMI codebook for selected rank.
        cb = PMICodebook(
            self.n_tx,
            ri,
        )

        # Select best beamforming matrix for this rank.
        pmi = cb.select_best_pmi(H)

        # Return both CSI feedback quantities.
        return ri, pmi