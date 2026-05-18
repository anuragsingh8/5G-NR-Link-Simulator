"""
phy/channel.py — Wireless Channel Models

This module defines three wireless channel models used for simulation:

  • AWGNChannel
      Adds complex Additive White Gaussian Noise to the transmitted signal.

  • RayleighChannel
      Applies flat Rayleigh fading independently over time samples.

  • CDLChannel
      Implements a simplified Clustered Delay Line channel model using
      CDL-A / CDL-B / CDL-C delay-power profiles inspired by 3GPP TS 38.901.

The goal of this file is to simulate realistic wireless impairments such as:
  • thermal noise,
  • fading,
  • multipath delay spread,
  • Doppler shift due to mobility,
  • multi-antenna MIMO propagation.
"""

from __future__ import annotations

import numpy as np
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# CDL delay / power profiles
# ─────────────────────────────────────────────────────────────────────────────
# These profiles approximate 3GPP CDL-A, CDL-B, and CDL-C channel models.
#
# Each tuple contains:
#   (excess_delay_ns, power_dB)
#
# excess_delay_ns:
#   Delay of a reflected path relative to the first arriving path.
#
# power_dB:
#   Relative average power of that path in decibels.
#
# These paths represent multiple reflected versions of the same transmitted
# signal arriving at the receiver through different propagation routes.
# ─────────────────────────────────────────────────────────────────────────────

_CDL_PROFILES: dict[str, list[tuple[float, float]]] = {
    "CDL-A": [
        (0.0, -13.4),
        (9.35, 0.0),
        (19.2, -2.2),
        (26.1, -4.0),
        (40.0, -6.0),
        (66.4, -8.2),
        (80.7, -9.9),
        (109.0, -10.5),
        (200.0, -7.5),
        (410.0, -15.9),
        (750.0, -20.1),
    ],
    "CDL-B": [
        (0.0, 0.0),
        (10.0, -2.2),
        (20.0, -4.0),
        (30.0, -3.2),
        (50.0, -9.8),
        (80.0, -1.2),
        (110.0, -3.4),
        (140.0, -5.2),
        (180.0, -7.6),
        (230.0, -3.0),
        (340.0, -8.9),
        (440.0, -11.6),
    ],
    "CDL-C": [
        (0.0, -4.4),
        (65.0, -1.2),
        (150.0, -3.5),
        (228.0, 0.0),
        (280.0, -9.9),
        (330.0, -16.9),
        (400.0, -15.2),
        (490.0, -10.8),
        (750.0, -11.1),
    ],
}


class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    This channel does not apply fading or multipath effects.
    It only adds complex Gaussian noise to the transmitted signal.

    This is useful as a baseline channel because it tests receiver performance
    when the only impairment is noise.
    """

    def __init__(self, snr_db: float = 15.0):
        """
        Store the target signal-to-noise ratio.

        Parameters
        ----------
        snr_db : float
            Desired SNR in decibels.
        """

        # Save SNR in dB. This value controls how strong the noise will be.
        self.snr_db = snr_db

    def apply(self, tx: np.ndarray) -> np.ndarray:
        """
        Add complex AWGN to the transmitted signal.

        Parameters
        ----------
        tx : np.ndarray
            Transmitted complex baseband signal.

        Returns
        -------
        np.ndarray
            Received signal after adding complex Gaussian noise.
        """

        # Convert SNR from dB to linear scale.
        snr_lin = 10 ** (self.snr_db / 10)

        # Estimate average transmit signal power.
        sig_power = np.mean(np.abs(tx) ** 2)

        # Complex noise has real and imaginary parts.
        # The factor of 2 splits the total noise power equally between them.
        noise_std = np.sqrt(sig_power / (2 * snr_lin))

        # Generate complex white Gaussian noise with the same shape as tx.
        noise = noise_std * (
            np.random.randn(*tx.shape) + 1j * np.random.randn(*tx.shape)
        )

        # Add noise to the transmitted signal.
        return tx + noise


class RayleighChannel:
    """
    Flat Rayleigh fading channel.

    In this model, every transmitted sample is multiplied by a complex
    Gaussian fading coefficient.

    "Flat fading" means the same channel effect is applied across frequency,
    so this model does not include multipath delay spread.
    """

    def __init__(self, snr_db: float = 15.0, n_tx: int = 1, n_rx: int = 1):
        """
        Initialize the Rayleigh channel.

        Parameters
        ----------
        snr_db : float
            Desired SNR in decibels.

        n_tx : int
            Number of transmit antennas.

        n_rx : int
            Number of receive antennas.
        """

        # Store target SNR.
        self.snr_db = snr_db

        # Store number of transmit antennas.
        self.n_tx = n_tx

        # Store number of receive antennas.
        self.n_rx = n_rx

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply flat Rayleigh fading and AWGN.

        Parameters
        ----------
        tx : np.ndarray
            Transmitted signal with shape ``(n_tx, n_samples)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            rx:
                Received noisy signal with shape ``(n_rx, n_samples)``.

            h:
                True fading channel with shape
                ``(n_rx, n_tx, n_samples)``.
        """

        # Number of time-domain samples in the transmitted signal.
        n_samples = tx.shape[-1]

        # Generate complex Gaussian Rayleigh fading coefficients.
        #
        # Shape:
        #   n_rx × n_tx × n_samples
        #
        # Division by sqrt(2) normalizes the complex Gaussian channel so that
        # its average power is approximately 1.
        h = (
            np.random.randn(self.n_rx, self.n_tx, n_samples)
            + 1j * np.random.randn(self.n_rx, self.n_tx, n_samples)
        ) / np.sqrt(2)

        # If tx is one-dimensional, treat it as a single-transmit-antenna input.
        tx_in = tx if tx.ndim == 2 else tx[np.newaxis]

        # Apply the MIMO channel.
        #
        # For each receive antenna and sample:
        #   rx[i, k] = sum_j h[i, j, k] * tx[j, k]
        rx = np.einsum("ijk,jk->ik", h, tx_in)

        # Convert SNR from dB to linear scale.
        snr_lin = 10 ** (self.snr_db / 10)

        # Estimate received signal power after fading.
        sig_power = np.mean(np.abs(rx) ** 2)

        # Compute noise standard deviation for complex AWGN.
        noise_std = np.sqrt(sig_power / (2 * snr_lin))

        # Generate complex Gaussian noise.
        noise = noise_std * (
            np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape)
        )

        # Return noisy received signal and the true channel.
        return rx + noise, h


class CDLChannel:
    """
    Clustered Delay Line channel.

    This model simulates a multipath wireless channel where the received signal
    is a sum of delayed, faded, and Doppler-shifted signal copies.

    It supports simplified CDL-A, CDL-B, and CDL-C profiles.
    """

    # Speed of light in meters per second.
    SPEED_OF_LIGHT = 3e8

    # Carrier frequency used to compute Doppler shift.
    # 3.5 GHz is a common 5G FR1 mid-band carrier.
    CARRIER_FREQ_HZ = 3.5e9

    def __init__(
        self,
        model: Literal["CDL-A", "CDL-B", "CDL-C"] = "CDL-A",
        snr_db: float = 15.0,
        velocity_kmh: float = 30.0,
        scs_hz: float = 30e3,
        n_fft: int = 2048,
        n_tx: int = 1,
        n_rx: int = 1,
    ):
        """
        Initialize the CDL channel.

        Parameters
        ----------
        model : Literal["CDL-A", "CDL-B", "CDL-C"]
            CDL profile name.

        snr_db : float
            Target SNR in decibels.

        velocity_kmh : float
            User equipment velocity in kilometers per hour.

        scs_hz : float
            OFDM subcarrier spacing in Hz.

        n_fft : int
            FFT size used to derive the sample rate.

        n_tx : int
            Number of transmit antennas.

        n_rx : int
            Number of receive antennas.
        """

        # Validate selected CDL model before continuing.
        if model not in _CDL_PROFILES:
            raise ValueError(
                f"Unknown CDL model '{model}'. Choose CDL-A, CDL-B, or CDL-C."
            )

        # Store chosen CDL model.
        self.model = model

        # Store target SNR.
        self.snr_db = snr_db

        # Store antenna configuration.
        self.n_tx = n_tx
        self.n_rx = n_rx

        # OFDM sample rate is approximated as:
        #   sample_rate = subcarrier_spacing × FFT_size
        self._sample_rate = scs_hz * n_fft

        # Convert velocity from km/h to m/s, then compute maximum Doppler shift:
        #
        #   f_d = v * f_c / c
        #
        # where:
        #   v   = receiver speed,
        #   f_c = carrier frequency,
        #   c   = speed of light.
        self._f_d = (
            (velocity_kmh / 3.6)
            * self.CARRIER_FREQ_HZ
            / self.SPEED_OF_LIGHT
        )

        # Load delay-power profile for the selected CDL model.
        profile = _CDL_PROFILES[model]

        # Separate the delay and power values into two lists.
        delays_ns, powers_db = zip(*profile)

        # Convert path delays from nanoseconds to sample units.
        self._delays_samp = np.array(
            [delay_ns * 1e-9 * self._sample_rate for delay_ns in delays_ns]
        )

        # Convert power from dB to linear scale.
        powers_lin = 10 ** (np.array(powers_db) / 10)

        # Normalize path powers so total average channel power is 1.
        self._powers = powers_lin / powers_lin.sum()

    def _doppler_phase(self, n_samples: int, n_paths: int) -> np.ndarray:
        """
        Generate Doppler phase rotation for each multipath component.

        Parameters
        ----------
        n_samples : int
            Number of time-domain samples.

        n_paths : int
            Number of multipath components.

        Returns
        -------
        np.ndarray
            Doppler phase matrix with shape ``(n_paths, n_samples)``.
        """

        # Create time vector in seconds.
        t = np.arange(n_samples) / self._sample_rate

        # Assign a random angle of arrival for each path.
        aoa = np.random.uniform(-np.pi, np.pi, (n_paths, 1))

        # Compute Doppler phase rotation.
        #
        # cos(aoa) projects the user velocity onto the arrival direction.
        return 2 * np.pi * self._f_d * np.cos(aoa) * t

    def _generate_taps(self, n_samples: int) -> np.ndarray:
        """
        Generate time-varying multipath channel taps.

        Parameters
        ----------
        n_samples : int
            Number of signal samples.

        Returns
        -------
        np.ndarray
            Channel taps with shape:
            ``(n_rx, n_tx, n_paths, n_samples)``.
        """

        # Number of multipath components in the selected CDL profile.
        n_paths = len(self._powers)

        # Generate Doppler phase for every path and time sample.
        phase = self._doppler_phase(n_samples, n_paths)

        # Generate complex Gaussian fading for each RX-TX-path-time component.
        g = (
            np.random.randn(self.n_rx, self.n_tx, n_paths, n_samples)
            + 1j * np.random.randn(self.n_rx, self.n_tx, n_paths, n_samples)
        ) / np.sqrt(2)

        # Convert Doppler phase into complex rotating phasors.
        doppler = np.exp(1j * phase)

        # Apply path power scaling and Doppler rotation to the fading taps.
        h = (
            g
            * np.sqrt(self._powers)[np.newaxis, np.newaxis, :, np.newaxis]
            * doppler
        )

        # Return full time-varying multipath channel.
        return h

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the CDL multipath channel to a time-domain signal.

        Parameters
        ----------
        tx : np.ndarray
            Transmitted signal with shape ``(n_tx, n_samples)`` or
            ``(n_samples,)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            rx:
                Received noisy signal with shape ``(n_rx, n_samples)``.

            h_eff:
                Effective flat channel obtained by summing all paths.
        """

        # If the input is one-dimensional, treat it as a single-antenna signal.
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]

        # Number of time samples.
        n_samples = tx.shape[1]

        # Generate time-varying multipath taps.
        h = self._generate_taps(n_samples)

        # Allocate received signal buffer.
        rx = np.zeros((self.n_rx, n_samples), dtype=complex)

        # Apply each delayed multipath component.
        for p, delay in enumerate(self._delays_samp):

            # Convert fractional delay in samples to nearest integer delay.
            d = int(round(delay))

            # If delay is zero, apply the path directly.
            if d == 0:
                rx += np.einsum("ijk,jk->ik", h[:, :, p, :], tx)

            # If delay is nonzero, delay the transmitted signal before applying
            # the path fading coefficient.
            else:
                # Create delayed version of tx by padding zeros at the start.
                tx_delayed = np.concatenate(
                    [
                        np.zeros((self.n_tx, d), dtype=complex),
                        tx[:, :-d] if d < n_samples else np.zeros_like(tx),
                    ],
                    axis=1,
                )

                # Add contribution from this delayed path.
                rx += np.einsum("ijk,jk->ik", h[:, :, p, :], tx_delayed)

        # Sum all multipath taps to produce a simplified effective channel.
        # This is useful for estimators that expect one channel value per sample.
        h_eff = h.sum(axis=2)

        # Convert SNR from dB to linear scale.
        snr_lin = 10 ** (self.snr_db / 10)

        # Estimate average received signal power.
        sig_power = np.mean(np.abs(rx) ** 2)

        # Compute standard deviation for complex AWGN.
        noise_std = np.sqrt(sig_power / (2 * snr_lin))

        # Generate complex Gaussian receiver noise.
        noise = noise_std * (
            np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape)
        )

        # Return noisy received signal and effective channel.
        return rx + noise, h_eff