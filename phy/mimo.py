"""
phy/mimo.py — MIMO Channel & Beamforming

This module provides utilities for:
  • MIMO channel matrix generation
  • Spatial correlation modelling
  • Channel quality metrics
  • Beamforming / precoding codebooks

Implemented components:
  • MIMOChannel
      Generates correlated MIMO channel matrices using the Kronecker model.

  • BeamformingMatrix
      Builds DFT-based beamforming codebooks similar to 3GPP Type-I
      single-panel precoding.

The implementations here are lightweight educational/simulation versions
inspired by:
  • 3GPP TS 38.214
  • MIMO information theory
"""

from __future__ import annotations

import numpy as np


class MIMOChannel:
    """
    MIMO channel matrix generator.

    This class creates spatially correlated MIMO channel matrices using the
    Kronecker correlation model:

        H = R_rx^(1/2) * H_iid * R_tx^(1/2)

    where:
        H_iid :
            Independent identically distributed Rayleigh fading matrix.

        R_tx :
            Transmit antenna correlation matrix.

        R_rx :
            Receive antenna correlation matrix.

    The model simulates realistic antenna correlation that occurs when:
      • antennas are physically close together,
      • propagation paths are limited,
      • scattering environments are non-ideal.

    Parameters
    ----------
    n_tx : int
        Number of transmit antennas.

    n_rx : int
        Number of receive antennas.

    correlation : float
        Spatial antenna correlation coefficient ρ in the range [0, 1].

        ρ = 0:
            Antennas are fully independent.

        ρ → 1:
            Antennas become highly correlated.
    """

    def __init__(
        self,
        n_tx: int = 4,
        n_rx: int = 2,
        correlation: float = 0.0,
    ):
        """
        Initialize the MIMO channel model.
        """

        # Store number of transmit antennas.
        self.n_tx = n_tx

        # Store number of receive antennas.
        self.n_rx = n_rx

        # Spatial correlation coefficient.
        self.rho = correlation

        # Build exponential transmit correlation matrix.
        #
        # Shape:
        #   (n_tx, n_tx)
        #
        # Example:
        #   rho = 0.5
        #
        #   [[1.0, 0.5, 0.25, ...],
        #    [0.5, 1.0, 0.5, ...],
        #    ...]
        self._R_tx = self._exp_corr(n_tx, correlation)

        # Build receive correlation matrix.
        self._R_rx = self._exp_corr(n_rx, correlation)

        # Compute Cholesky decomposition of the correlation matrices.
        #
        # These matrix square roots are used to transform an IID fading matrix
        # into a spatially correlated channel matrix.
        #
        # Small diagonal regularisation (1e-12) improves numerical stability.
        self._R_tx_sqrt = np.linalg.cholesky(
            self._R_tx + 1e-12 * np.eye(n_tx)
        )

        self._R_rx_sqrt = np.linalg.cholesky(
            self._R_rx + 1e-12 * np.eye(n_rx)
        )

    @staticmethod
    def _exp_corr(n: int, rho: float) -> np.ndarray:
        """
        Generate exponential spatial correlation matrix.

        Mathematical form:

            R[i,j] = ρ^|i-j|

        Nearby antennas become more correlated than distant antennas.

        Parameters
        ----------
        n : int
            Number of antennas.

        rho : float
            Correlation coefficient.

        Returns
        -------
        np.ndarray
            Correlation matrix with shape (n, n).
        """

        # Antenna indices:
        #   [0, 1, 2, ..., n-1]
        idx = np.arange(n)

        # subtract.outer(idx, idx) creates a matrix:
        #
        #   [[ 0, -1, -2, ...],
        #    [ 1,  0, -1, ...],
        #    ...]
        #
        # Taking absolute value gives antenna separation distance.
        #
        # Then:
        #   rho^distance
        #
        # models exponentially decaying spatial correlation.
        return rho ** np.abs(
            np.subtract.outer(idx, idx)
        ).astype(float)

    def generate(self, n_realizations: int = 1) -> np.ndarray:
        """
        Generate correlated MIMO channel matrix/matrices.

        Parameters
        ----------
        n_realizations : int
            Number of independent channel realizations.

        Returns
        -------
        np.ndarray
            Channel matrices.

            Shape:
                (n_realizations, n_rx, n_tx)

            If n_realizations == 1:
                returns shape (n_rx, n_tx).
        """

        # Generate IID Rayleigh fading matrix:
        #
        # Entries follow:
        #   CN(0,1)
        #
        # Shape:
        #   (batch, n_rx, n_tx)
        H_iid = (
            np.random.randn(
                n_realizations,
                self.n_rx,
                self.n_tx,
            )
            + 1j * np.random.randn(
                n_realizations,
                self.n_rx,
                self.n_tx,
            )
        ) / np.sqrt(2)

        # Apply Kronecker correlation model:
        #
        #   H = R_rx^(1/2) * H_iid * R_tx^(1/2)
        #
        # einsum efficiently performs batched matrix multiplication.
        H = np.einsum(
            'ij,bjk,lk->bil',
            self._R_rx_sqrt,
            H_iid,
            self._R_tx_sqrt.conj(),
        )

        # Remove singleton batch dimension if only one realization exists.
        return H.squeeze()

    def condition_number(self, H: np.ndarray) -> float:
        """
        Compute condition number of the MIMO channel matrix.

        The condition number measures how well-conditioned the channel is.

        Interpretation:
          • Small condition number:
                Strong spatial separability.
                Good MIMO performance.

          • Large condition number:
                Layers become difficult to separate.
                Channel inversion becomes unstable.

        Parameters
        ----------
        H : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        float
            Channel condition number.
        """

        # Singular values of the channel matrix.
        sv = np.linalg.svd(H, compute_uv=False)

        # Condition number:
        #
        #   largest singular value / smallest singular value
        #
        # Small epsilon prevents division by zero.
        return float(sv[0] / (sv[-1] + 1e-12))

    def capacity(self, H: np.ndarray, snr_db: float) -> float:
        """
        Compute Shannon MIMO channel capacity.

        Mathematical formula:

            C = log2 det(I + (SNR/n_tx) H Hᴴ)

        Units:
            bits/s/Hz

        Parameters
        ----------
        H : np.ndarray
            Channel matrix with shape (n_rx, n_tx).

        snr_db : float
            Operating SNR in dB.

        Returns
        -------
        float
            Estimated channel capacity.
        """

        # Convert SNR from dB to linear scale.
        snr_lin = 10 ** (snr_db / 10)

        # Number of receive antennas.
        n = H.shape[0]

        # Compute:
        #
        #   H Hᴴ
        #
        # which captures spatial channel power.
        HHh = H @ H.conj().T

        # Shannon MIMO capacity formula.
        #
        # det() measures how many parallel spatial streams can be supported.
        C = np.real(
            np.log2(
                np.linalg.det(
                    np.eye(n)
                    + (snr_lin / self.n_tx) * HHh
                )
            )
        )

        # Numerical safety:
        # capacity cannot be negative.
        return float(max(C, 0.0))


class BeamformingMatrix:
    """
    DFT-based beamforming / precoding codebook.

    This class implements simplified codebook-based precoding inspired by
    3GPP Type-I single-panel beamforming.

    Supported modes:
      • 1 layer:
            Single beam selected from DFT matrix columns.

      • 2 layers:
            Orthogonal beam pairs.

      • 4 layers:
            Full DFT matrix (only supported for 4 TX antennas).

    Beamforming improves:
      • signal strength,
      • spatial multiplexing,
      • interference suppression,
      • link robustness.
    """

    def __init__(self, n_tx: int = 4, n_layers: int = 1):
        """
        Initialize beamforming codebook.

        Parameters
        ----------
        n_tx : int
            Number of transmit antennas.

        n_layers : int
            Number of spatial layers / transmission rank.
        """

        # Store antenna count.
        self.n_tx = n_tx

        # Store number of layers.
        self.n_layers = n_layers

        # Build beamforming codebook during initialization.
        self._codebook = self._build_codebook()

    def _build_codebook(self) -> list[np.ndarray]:
        """
        Construct DFT-based beamforming codebook.

        Returns
        -------
        list[np.ndarray]
            List of candidate precoding matrices.
        """

        # Number of antennas.
        n = self.n_tx

        # Build normalized Discrete Fourier Transform matrix.
        #
        # DFT beams correspond to different spatial steering directions.
        F = np.fft.fft(np.eye(n)) / np.sqrt(n)

        # Single-layer transmission:
        #
        # Each beam is one DFT column.
        if self.n_layers == 1:
            return [F[:, [i]] for i in range(n)]

        # Two-layer transmission:
        #
        # Use orthogonal beam pairs.
        elif self.n_layers == 2:

            # Pair each beam with another beam separated by half the array.
            pairs = [
                (i, (i + n // 2) % n)
                for i in range(n)
            ]

            # Normalize pair power.
            return [
                F[:, list(p)] / np.sqrt(2)
                for p in pairs
            ]

        # Four-layer transmission for 4x4 MIMO.
        elif self.n_layers == 4 and n == 4:
            return [F]

        # Fallback:
        #
        # Use truncated identity matrix if requested configuration is not
        # explicitly supported.
        else:
            return [np.eye(n, self.n_layers)]

    def get_precoder(self, pmi: int) -> np.ndarray:
        """
        Retrieve beamforming matrix for a PMI index.

        PMI = Precoding Matrix Indicator.

        Parameters
        ----------
        pmi : int
            Codebook index.

        Returns
        -------
        np.ndarray
            Precoding matrix with shape:
            (n_tx, n_layers)
        """

        # Validate PMI range before indexing codebook.
        if pmi >= len(self._codebook):
            raise ValueError(
                f"PMI {pmi} out of range "
                f"[0, {len(self._codebook)-1}]"
            )

        # Return selected precoding matrix.
        return self._codebook[pmi]

    @property
    def codebook_size(self) -> int:
        """
        Return number of available beamforming matrices.
        """

        return len(self._codebook)

    def select_pmi(self, H: np.ndarray) -> int:
        """
        Select best PMI based on capacity maximization.

        The method evaluates every precoding matrix and selects the beamformer
        that produces the largest estimated spatial capacity.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix with shape (n_rx, n_tx).

        Returns
        -------
        int
            Best PMI index.
        """

        # Best PMI found so far.
        best_pmi = 0

        # Best estimated capacity.
        best_cap = -np.inf

        # Test every candidate beamforming matrix.
        for pmi, W in enumerate(self._codebook):

            # Effective channel after precoding:
            #
            #   H_eff = H * W
            #
            # Shape:
            #   (n_rx, n_layers)
            HW = H @ W

            # Singular values represent effective spatial stream strengths.
            sv = np.linalg.svd(HW, compute_uv=False)

            # Proxy spatial capacity metric:
            #
            #   Σ log2(1 + sv²)
            #
            # Larger values imply better beamforming performance.
            cap = float(
                np.sum(
                    np.log2(1 + sv ** 2)
                )
            )

            # Keep best-performing PMI.
            if cap > best_cap:
                best_cap = cap
                best_pmi = pmi

        # Return selected beam index.
        return best_pmi