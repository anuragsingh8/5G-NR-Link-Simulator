"""
phy/channel_est.py — Channel Estimators

This module contains simple pilot-based channel estimators:

  • LSEstimator
      Least Squares estimator using DMRS pilot tones.

  • LMMSEEstimator
      Linear Minimum Mean Square Error estimator using a simple
      frequency-domain channel correlation model.

Reference:
  3GPP TS 38.211 Section 7.4 — DMRS
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DMRS pilot pattern
# ─────────────────────────────────────────────────────────────────────────────
# This is a simplified Type-1, single-port DMRS pattern.
#
# Assumption:
#   • DMRS pilots occupy even-numbered subcarriers.
#   • Pilot OFDM symbols are assumed to occur at symbol indices 2 and 11
#     within a slot, although this file estimates one received pilot symbol
#     at a time.
#
# Note:
#   This is not a full 3GPP-complete DMRS implementation. It is a lightweight
#   pilot pattern suitable for simulation and estimator testing.
# ─────────────────────────────────────────────────────────────────────────────


def generate_dmrs_pilots(n_subcarriers: int, seed: int = 42) -> np.ndarray:
    """
    Generate a deterministic QPSK DMRS pilot sequence.

    Parameters
    ----------
    n_subcarriers : int
        Total number of active OFDM subcarriers.

    seed : int
        Random seed used to make the generated pilot sequence repeatable.

    Returns
    -------
    np.ndarray
        Complex QPSK pilot symbols with shape ``(n_subcarriers // 2,)``.
        Each pilot has unit average magnitude.
    """

    # Create a NumPy random number generator with a fixed seed so that the
    # same pilot sequence is produced every time for the same input settings.
    rng = np.random.default_rng(seed)

    # Generate two random bits per QPSK symbol.
    # Since pilots are placed on half of the subcarriers, the number of QPSK
    # pilot symbols is n_subcarriers // 2.
    bits = rng.integers(0, 2, size=2 * (n_subcarriers // 2))

    # Map bit pairs to QPSK constellation symbols:
    #
    #   bit 0 controls the real part
    #   bit 1 controls the imaginary part
    #
    # Mapping:
    #   0 -> +1
    #   1 -> -1
    #
    # Division by sqrt(2) normalizes the QPSK symbol power to 1.
    pilots = (
        1 - 2 * bits[0::2]
        + 1j * (1 - 2 * bits[1::2])
    ) / np.sqrt(2)

    # Return one pilot symbol for every even-numbered subcarrier.
    return pilots


def get_pilot_indices(n_subcarriers: int) -> np.ndarray:
    """
    Return the subcarrier indices occupied by DMRS pilots.

    Parameters
    ----------
    n_subcarriers : int
        Total number of active OFDM subcarriers.

    Returns
    -------
    np.ndarray
        Even-numbered subcarrier indices:
        ``[0, 2, 4, ..., n_subcarriers - 2]`` when n_subcarriers is even.
    """

    # Pilots are placed on every other subcarrier, starting at index 0.
    return np.arange(0, n_subcarriers, 2)


class LSEstimator:
    """
    Least Squares channel estimator.

    The LS estimate is computed at pilot locations using:

        H_ls[k] = Y[k] / X[k]

    where:
        Y[k] is the received pilot value,
        X[k] is the known transmitted pilot value,
        H_ls[k] is the channel estimate at pilot subcarrier k.

    After estimating the channel on pilot subcarriers, the estimator performs
    linear interpolation to obtain channel estimates for all subcarriers.

    Parameters
    ----------
    n_subcarriers : int
        Total number of active OFDM subcarriers.
    """

    def __init__(self, n_subcarriers: int = 624):
        """
        Initialize the LS estimator.

        Parameters
        ----------
        n_subcarriers : int
            Total number of active OFDM subcarriers.
        """

        # Store the number of active subcarriers used by the OFDM system.
        self.n_subcarriers = n_subcarriers

        # Compute the subcarrier locations that carry DMRS pilots.
        self.pilot_idx = get_pilot_indices(n_subcarriers)

        # Generate the known transmitted DMRS pilot symbols.
        self.pilots = generate_dmrs_pilots(n_subcarriers)

        # Store all subcarrier indices. These are the target points where the
        # final interpolated channel estimate will be evaluated.
        self.data_idx = np.arange(n_subcarriers)

    def estimate(self, rx_pilot_symbol: np.ndarray) -> np.ndarray:
        """
        Estimate the channel from one received OFDM pilot symbol.

        Parameters
        ----------
        rx_pilot_symbol : np.ndarray
            Received OFDM symbol with shape ``(n_subcarriers,)``.
            Pilot values are read from the known DMRS pilot positions.

        Returns
        -------
        np.ndarray
            Complex channel estimate with shape ``(n_subcarriers,)``.
            Values at non-pilot subcarriers are obtained by interpolation.
        """

        # Extract the received values at pilot subcarriers and divide them by
        # the known transmitted pilots. This gives the LS channel estimate at
        # pilot positions only.
        h_pilot = rx_pilot_symbol[self.pilot_idx] / self.pilots

        # Interpolate the real part of the pilot channel estimates from pilot
        # positions to every active subcarrier.
        h_real = np.interp(
            self.data_idx,
            self.pilot_idx,
            h_pilot.real,
        )

        # Interpolate the imaginary part separately because np.interp operates
        # on real-valued data.
        h_imag = np.interp(
            self.data_idx,
            self.pilot_idx,
            h_pilot.imag,
        )

        # Recombine the interpolated real and imaginary parts into a complex
        # channel estimate.
        h_est = h_real + 1j * h_imag

        # Return the full-band channel estimate.
        return h_est


class LMMSEEstimator:
    """
    Linear Minimum Mean Square Error channel estimator.

    The estimator first computes the LS estimate at pilot locations, then
    applies an LMMSE filtering matrix:

        H_lmmse = R_hh (R_hh + (1 / SNR) I)^(-1) H_ls

    where:
        R_hh is the pilot-domain channel correlation matrix,
        SNR is the linear signal-to-noise ratio,
        I is the identity matrix,
        H_ls is the LS channel estimate at pilot positions.

    The pilot-domain LMMSE estimate is then linearly interpolated to all
    subcarriers.

    Parameters
    ----------
    n_subcarriers : int
        Number of active OFDM subcarriers.

    snr_db : float
        Operating SNR in dB. Used to regularize the LMMSE filter.

    coherence_bw : float
        Coherence bandwidth measured in subcarrier spacings. Larger values
        imply a smoother frequency-domain channel.
    """

    def __init__(
        self,
        n_subcarriers: int = 624,
        snr_db: float = 15.0,
        coherence_bw: float = 50.0,
    ):
        """
        Initialize the LMMSE estimator.

        Parameters
        ----------
        n_subcarriers : int
            Number of active OFDM subcarriers.

        snr_db : float
            Signal-to-noise ratio in decibels.

        coherence_bw : float
            Controls the exponential decay of frequency correlation.
        """

        # Store OFDM size.
        self.n_subcarriers = n_subcarriers

        # Store SNR in dB for reference and reproducibility.
        self.snr_db = snr_db

        # Compute pilot subcarrier positions.
        self.pilot_idx = get_pilot_indices(n_subcarriers)

        # Generate the known transmitted DMRS pilots.
        self.pilots = generate_dmrs_pilots(n_subcarriers)

        # Store all subcarrier indices for the final interpolation step.
        self.data_idx = np.arange(n_subcarriers)

        # Count how many pilot tones are used.
        self.n_pilots = len(self.pilot_idx)

        # Compute the pairwise distance between pilot subcarriers.
        # subtract.outer(a, a) forms a matrix where element (i, j) is:
        #
        #   pilot_idx[i] - pilot_idx[j]
        #
        # Taking the absolute value gives frequency separation between pilots.
        d = np.abs(np.subtract.outer(self.pilot_idx, self.pilot_idx))

        # Build a simple exponential channel correlation matrix.
        #
        # Nearby pilot subcarriers are assumed to have highly correlated
        # channel responses. Correlation decays as frequency distance grows.
        self._R_hh = np.exp(-d / coherence_bw).astype(complex)

        # Convert SNR from dB to linear scale:
        #
        #   SNR_linear = 10^(SNR_dB / 10)
        snr_lin = 10 ** (snr_db / 10)

        # Build the LMMSE filter matrix:
        #
        #   W = R_hh @ inv(R_hh + noise_variance * I)
        #
        # Here, noise_variance is approximated as 1 / SNR_linear.
        # The identity term prevents overfitting noisy LS estimates.
        self._W = self._R_hh @ np.linalg.inv(
            self._R_hh + np.eye(self.n_pilots) / snr_lin
        )

    def estimate(self, rx_pilot_symbol: np.ndarray) -> np.ndarray:
        """
        Estimate the channel from one received OFDM pilot symbol.

        Parameters
        ----------
        rx_pilot_symbol : np.ndarray
            Received OFDM symbol with shape ``(n_subcarriers,)``.

        Returns
        -------
        np.ndarray
            Complex LMMSE channel estimate with shape ``(n_subcarriers,)``.
            The estimate is filtered at pilot positions and interpolated across
            the full frequency band.
        """

        # Compute the raw LS estimate at pilot subcarriers.
        h_ls = rx_pilot_symbol[self.pilot_idx] / self.pilots

        # Apply the precomputed LMMSE filter matrix to suppress noise using
        # the assumed channel correlation structure.
        h_lmmse = self._W @ h_ls

        # Interpolate the real part of the LMMSE pilot estimates.
        h_real = np.interp(
            self.data_idx,
            self.pilot_idx,
            h_lmmse.real,
        )

        # Interpolate the imaginary part of the LMMSE pilot estimates.
        h_imag = np.interp(
            self.data_idx,
            self.pilot_idx,
            h_lmmse.imag,
        )

        # Combine real and imaginary parts into the final complex estimate.
        h_est = h_real + 1j * h_imag

        # Return the full-subcarrier channel estimate.
        return h_est