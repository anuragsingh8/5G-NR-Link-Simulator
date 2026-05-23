"""
phy/channel.py — Wireless Channel Models
========================================

This module contains channel models used for OFDM, LTE, and 5G NR
link-level simulations.

The purpose of this file is to model how a transmitted complex baseband signal
is distorted before reaching the receiver.

A generic received signal can be written as:

    y(t) = h(t) * x(t) + n(t)

where:
    x(t) : transmitted signal
    h(t) : wireless channel response
    n(t) : additive noise
    y(t) : received signal

Implemented Channel Algorithms
------------------------------

1. AWGNChannel — Additive White Gaussian Noise
------------------------------------------------
Algorithm:
    rx = tx + noise

This is the simplest channel model. It does not include fading, Doppler,
multipath, delay spread, or antenna effects.

It only adds complex Gaussian noise:

    noise = sigma * (N(0,1) + jN(0,1))

Used for:
    • baseline BER testing
    • validating modulation and demodulation
    • comparing simulation BER against theoretical AWGN curves

Modes:
    • snr_db mode:
        Noise is scaled using measured signal power.

    • ebn0_db mode:
        Noise is scaled using Eb/N0 and bits per symbol.


2. RayleighChannel — MIMO Flat Rayleigh Fading
----------------------------------------------
Algorithm:
    h ~ CN(0,1)
    rx[i, k] = Σ_j h[i, j, k] * tx[j, k] + noise

This models flat fading for MIMO systems.

Flat fading means:
    • no delay spread
    • no multipath convolution
    • all frequencies experience the same fading at a given time sample

Rayleigh fading represents non-line-of-sight propagation where the received
signal is formed by many scattered paths.

Used for:
    • MIMO fading tests
    • receiver equaliser validation
    • antenna diversity experiments


3. FlatRayleighChannel — Single-Antenna Block Rayleigh
------------------------------------------------------
Algorithm:
    h ~ CN(0,1)
    rx = h * tx + noise

This is a simpler single-antenna Rayleigh channel.

The channel coefficient may be:
    • constant across one OFDM symbol
    • independent per sample if per_sample=True

Used for:
    • BER sweeps
    • theoretical Rayleigh BER comparison
    • simple fading-channel validation


4. EPAChannel — Extended Pedestrian A Multipath Channel
-------------------------------------------------------
Algorithm:
    rx[n] = Σ_p h_p[n] * tx[n - d_p] + noise

where:
    p   : path/tap index
    d_p : delay of path p in samples
    h_p : complex fading coefficient of path p

EPA is a 7-tap multipath channel profile from 3GPP TS 36.104.

Each tap has:
    • fixed delay
    • relative power
    • Rayleigh fading
    • Doppler phase rotation

Used for:
    • OFDM cyclic-prefix validation
    • pedestrian mobility channel testing
    • multipath BER simulations

Important:
    The cyclic prefix must be longer than the maximum channel delay spread.

For EPA:
    max delay = 410 ns

If CP duration is shorter than delay spread:
    • inter-symbol interference increases
    • BER becomes worse
    • OFDM orthogonality breaks


5. CDLChannel — Clustered Delay Line Channel
--------------------------------------------
Algorithm:
    rx_rx[n] = Σ_tx Σ_path h[rx, tx, path, n] * tx_tx[n - d_path] + noise

This is a simplified 5G NR clustered delay line model.

Supported profiles:
    • CDL-A
    • CDL-B
    • CDL-C

Each profile contains:
    • multiple delay clusters
    • relative path powers
    • MIMO TX/RX channel taps
    • Doppler evolution over time

Used for:
    • 5G-style multipath simulations
    • MIMO channel testing
    • Doppler and mobility studies
    • realistic link-level experiments

CDL is more realistic than EPA because it supports:
    • MIMO
    • many clusters
    • larger delay spreads
    • time-varying Doppler per path


SNR / Eb/N0 Conventions
-----------------------

Different simulation modes use different noise references.

SNR mode:
    Used when noise is scaled relative to measured signal power.

    snr_linear = 10^(snr_db / 10)
    noise_power = signal_power / snr_linear

Eb/N0 mode:
    Used for BER curves and modulation validation.

    Eb/N0 = energy per bit / noise spectral density

    sigma = sqrt(1 / (2 * Eb/N0_linear * bits_per_symbol))

Use snr_db when:
    • running full link simulation
    • signal power may change dynamically
    • MIMO/channel gain changes power

Use ebn0_db when:
    • comparing BER against theory
    • running modulation-only simulations
    • constellation power is normalized


Theoretical BER Utilities
-------------------------

This file also provides reference BER formulas:

    ber_bpsk_awgn()
        Theoretical BPSK BER over AWGN.

    ber_qpsk_awgn()
        Same as BPSK for Gray-coded QPSK.

    ber_qam_awgn()
        Approximate BER for square Gray-coded QAM.

    ber_rayleigh_bpsk()
        Exact BER for BPSK over flat Rayleigh fading.

    ber_rayleigh_qpsk()
        Same as BPSK for Gray-coded QPSK over Rayleigh fading.

These are useful for checking whether simulation results are correct.

References
----------
    • 3GPP TS 36.104 Table B.2-1
    • 3GPP TS 38.901 Table 7.7.2
    • Digital Communications theory for AWGN/Rayleigh BER expressions
"""

from __future__ import annotations

import numpy as np
from typing import Literal
from scipy.special import erfc


# ─────────────────────────────────────────────────────────────────────────────
# EPA delay profile
# ─────────────────────────────────────────────────────────────────────────────
# EPA = Extended Pedestrian A.
#
# Each tuple represents one multipath component:
#   (delay in nanoseconds, relative power in dB)
#
# The receiver observes the sum of all delayed paths.
# Larger delay spread makes the channel more frequency-selective.
# ─────────────────────────────────────────────────────────────────────────────

EPA_TAPS = [
    (0,    0.0),     # strongest reference path
    (30,  -1.0),    # weak delayed reflected copy
    (70,  -2.0),
    (90,  -3.0),
    (110, -8.0),
    (190, -17.2),
    (410, -20.8),   # latest EPA tap
]

# Maximum EPA delay spread.
# OFDM cyclic prefix should be longer than this delay.
EPA_MAX_DELAY_NS = 410.0


# ─────────────────────────────────────────────────────────────────────────────
# CDL delay / power profiles
# ─────────────────────────────────────────────────────────────────────────────
# CDL = Clustered Delay Line.
#
# These profiles approximate 3GPP CDL-A/B/C multipath models.
# Each entry is:
#   (excess_delay_ns, relative_power_dB)
# ─────────────────────────────────────────────────────────────────────────────

_CDL_PROFILES: dict[str, list[tuple[float, float]]] = {
    "CDL-A": [
        (0.0, -13.4), (9.35, 0.0), (19.2, -2.2), (26.1, -4.0),
        (40.0, -6.0), (66.4, -8.2), (80.7, -9.9), (109.0, -10.5),
        (200.0, -7.5), (410.0, -15.9), (750.0, -20.1),
    ],
    "CDL-B": [
        (0.0, 0.0), (10.0, -2.2), (20.0, -4.0), (30.0, -3.2),
        (50.0, -9.8), (80.0, -1.2), (110.0, -3.4), (140.0, -5.2),
        (180.0, -7.6), (230.0, -3.0), (340.0, -8.9), (440.0, -11.6),
    ],
    "CDL-C": [
        (0.0, -4.4), (65.0, -1.2), (150.0, -3.5), (228.0, 0.0),
        (280.0, -9.9), (330.0, -16.9), (400.0, -15.2), (490.0, -10.8),
        (750.0, -11.1),
    ],
}


def _ebn0_to_noise_sigma(ebn0_db: float, bits_per_symbol: int) -> float:
    """
    Convert Eb/N0 in dB to complex AWGN standard deviation.

    Eb/N0 is commonly used for BER simulations.
    The factor of 2 splits noise power between I and Q components.
    """

    ebn0_lin = 10 ** (ebn0_db / 10)

    return float(
        np.sqrt(1.0 / (2.0 * ebn0_lin * bits_per_symbol))
    )


class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    This channel adds only noise:
      • no fading
      • no Doppler
      • no multipath
      • no delay spread

    It supports two modes:
      1. SNR mode for link simulation
      2. Eb/N0 mode for BER validation
    """

    name = "AWGN"

    def __init__(
        self,
        snr_db: float = 15.0,
        bits_per_symbol: int = None,
        ebn0_db: float = None,
    ):
        # If ebn0_db is provided, use Eb/N0-based noise scaling.
        self._use_ebn0 = ebn0_db is not None

        if self._use_ebn0:
            # Eb/N0 mode requires modulation order.
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol is required when using ebn0_db")

            self.ebn0_db = ebn0_db
            self.bits_per_symbol = bits_per_symbol

            # Precompute noise sigma for normalized constellations.
            self._sigma = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)

            # Equivalent SNR for logging/debugging.
            self.snr_db = ebn0_db + 10 * np.log10(bits_per_symbol)

        else:
            # SNR mode measures signal power dynamically.
            self.snr_db = snr_db
            self.ebn0_db = None
            self.bits_per_symbol = bits_per_symbol
            self._sigma = None

    @property
    def sigma(self) -> float:
        """Return fixed noise standard deviation in Eb/N0 mode."""

        if self._sigma is None:
            raise AttributeError("sigma is only defined in ebn0_db mode")

        return self._sigma

    def apply(self, tx: np.ndarray) -> np.ndarray:
        """Add complex Gaussian noise to transmitted samples."""

        if self._use_ebn0:
            # Fixed noise level for BER curves.
            noise = self._sigma * (
                np.random.randn(*tx.shape)
                + 1j * np.random.randn(*tx.shape)
            )

        else:
            # Convert SNR from dB to linear.
            snr_lin = 10 ** (self.snr_db / 10)

            # Measure actual signal power.
            sig_power = np.mean(np.abs(tx) ** 2)

            # Complex noise standard deviation.
            noise_std = np.sqrt(sig_power / (2 * snr_lin))

            noise = noise_std * (
                np.random.randn(*tx.shape)
                + 1j * np.random.randn(*tx.shape)
            )

        return tx + noise


class RayleighChannel:
    """
    MIMO flat Rayleigh fading channel.

    Flat fading means the channel has no delay spread.
    Each sample is multiplied by a complex Gaussian channel coefficient.

    Channel shape:
      h[rx, tx, sample]
    """

    name = "Rayleigh"

    def __init__(self, snr_db: float = 15.0, n_tx: int = 1, n_rx: int = 1):
        self.snr_db = snr_db
        self.n_tx = n_tx
        self.n_rx = n_rx

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply MIMO Rayleigh fading and AWGN."""

        # Convert single stream to shape (1, n_samples).
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]

        n_samples = tx.shape[1]

        # Generate complex Gaussian fading coefficients.
        h = (
            np.random.randn(self.n_rx, self.n_tx, n_samples)
            + 1j * np.random.randn(self.n_rx, self.n_tx, n_samples)
        ) / np.sqrt(2)

        # Apply MIMO channel:
        # rx[i,k] = sum_j h[i,j,k] * tx[j,k]
        rx = np.einsum("ijk,jk->ik", h, tx)

        # Add signal-power-referenced AWGN.
        snr_lin = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))

        noise = noise_std * (
            np.random.randn(*rx.shape)
            + 1j * np.random.randn(*rx.shape)
        )

        return rx + noise, h


class FlatRayleighChannel:
    """
    Single-antenna flat Rayleigh channel.

    Used for BER simulations where one scalar fading coefficient is applied
    to the whole OFDM symbol.
    """

    name = "Flat Rayleigh"

    def __init__(self, ebn0_db: float = 10.0, bits_per_symbol: int = 2):
        self.ebn0_db = ebn0_db
        self.bits_per_symbol = bits_per_symbol

        # Noise standard deviation from Eb/N0.
        self._sigma = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)

        # Current fading coefficient.
        self._h: complex = 1.0 + 0j

    def new_realization(self):
        """Generate a new Rayleigh fading coefficient."""

        self._h = (
            np.random.randn()
            + 1j * np.random.randn()
        ) / np.sqrt(2)

    @property
    def h(self) -> complex:
        """Return current fading coefficient."""
        return self._h

    @property
    def sigma(self) -> float:
        """Return Eb/N0-based noise sigma."""
        return self._sigma

    def apply(self, tx: np.ndarray, per_sample: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Apply flat Rayleigh fading and AWGN."""

        if per_sample:
            # Fast fading: independent fading per sample.
            h = (
                np.random.randn(len(tx))
                + 1j * np.random.randn(len(tx))
            ) / np.sqrt(2)
        else:
            # Block fading: one channel coefficient for full symbol.
            self.new_realization()
            h = self._h

        noise = self._sigma * (
            np.random.randn(len(tx))
            + 1j * np.random.randn(len(tx))
        )

        rx = h * tx + noise

        return rx, h


class EPAChannel:
    """
    Extended Pedestrian A multipath channel.

    EPA is a 7-tap channel model. Each tap has:
      • delay
      • relative power
      • complex fading
      • Doppler rotation
    """

    name = "EPA"
    MAX_DELAY_NS = EPA_MAX_DELAY_NS

    def __init__(
        self,
        ebn0_db: float = 10.0,
        bits_per_symbol: int = 2,
        fs_hz: float = 30.72e6,
        velocity_kmh: float = 3.0,
        carrier_freq_hz: float = 2.0e9,
    ):
        self.ebn0_db = ebn0_db
        self.bits_per_symbol = bits_per_symbol
        self.fs = fs_hz

        self._sigma = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)

        # Convert delays and powers into arrays.
        delays_ns = np.array([tap[0] for tap in EPA_TAPS], dtype=float)
        powers_db = np.array([tap[1] for tap in EPA_TAPS], dtype=float)

        # Convert delay from nanoseconds to sample indices.
        self._delays_samp = np.round(
            delays_ns * 1e-9 * fs_hz
        ).astype(int)

        # Convert powers from dB to linear.
        powers_lin = 10 ** (powers_db / 10.0)

        # Convert powers to normalized tap amplitudes.
        self._tap_gains = np.sqrt(powers_lin / powers_lin.sum())

        # Doppler frequency.
        speed_mps = velocity_kmh / 3.6
        self._fd = speed_mps * carrier_freq_hz / 3e8

    @property
    def max_delay_samples(self) -> int:
        """Maximum EPA tap delay in samples."""
        return int(self._delays_samp.max())

    @property
    def max_delay_us(self) -> float:
        """Maximum EPA tap delay in microseconds."""
        return float(self._delays_samp.max() / self.fs * 1e6)

    def _doppler_coeff(self, n_samples: int) -> np.ndarray:
        """Generate Jakes-style Doppler phasors for each EPA tap."""

        n_taps = len(self._tap_gains)

        # Random angle of arrival per tap.
        aoa = np.random.uniform(-np.pi, np.pi, (n_taps, 1))

        # Time vector.
        t = np.arange(n_samples) / self.fs

        # Doppler phase rotation.
        return np.exp(
            1j * 2 * np.pi * self._fd * np.cos(aoa) * t
        )

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply EPA multipath channel and AWGN."""

        N_in = len(tx)

        # Output signal buffer.
        rx = np.zeros(N_in, dtype=complex)

        # Time-varying Doppler coefficients.
        doppler = self._doppler_coeff(N_in)

        # Store first-sample tap impulse response.
        h_impulse = np.zeros(len(self._tap_gains), dtype=complex)

        for p, (d, g) in enumerate(zip(self._delays_samp, self._tap_gains)):
            # Random complex tap gain.
            h_tap = (
                np.random.randn()
                + 1j * np.random.randn()
            ) / np.sqrt(2) * g

            h_impulse[p] = h_tap

            # Apply Doppler over time.
            h_t = h_tap * doppler[p]

            if d == 0:
                rx += h_t * tx

            elif d < N_in:
                # Delayed version of tx.
                rx[d:] += h_t[d:] * tx[:N_in - d]

        noise = self._sigma * (
            np.random.randn(N_in)
            + 1j * np.random.randn(N_in)
        )

        return rx + noise, h_impulse


class CDLChannel:
    """
    Clustered Delay Line channel.

    Supports CDL-A, CDL-B, and CDL-C profiles.
    This model includes MIMO, multipath delay, Doppler, and AWGN.
    """

    SPEED_OF_LIGHT = 3e8
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
        if model not in _CDL_PROFILES:
            raise ValueError(
                f"Unknown CDL model '{model}'. Choose CDL-A, CDL-B, or CDL-C."
            )

        self.model = model
        self.snr_db = snr_db
        self.n_tx = n_tx
        self.n_rx = n_rx

        # OFDM sample rate approximation.
        self._sample_rate = scs_hz * n_fft

        # Maximum Doppler shift.
        self._f_d = (
            (velocity_kmh / 3.6)
            * self.CARRIER_FREQ_HZ
            / self.SPEED_OF_LIGHT
        )

        # Load delay-power profile.
        profile = _CDL_PROFILES[model]
        delays_ns, powers_db = zip(*profile)

        # Convert path delays to sample units.
        self._delays_samp = np.array([
            d * 1e-9 * self._sample_rate
            for d in delays_ns
        ])

        # Normalize path powers.
        powers_lin = 10 ** (np.array(powers_db) / 10)
        self._powers = powers_lin / powers_lin.sum()

    def _doppler_phase(self, n_samples: int, n_paths: int) -> np.ndarray:
        """Generate Doppler phase for each path and sample."""

        t = np.arange(n_samples) / self._sample_rate
        aoa = np.random.uniform(-np.pi, np.pi, (n_paths, 1))

        return 2 * np.pi * self._f_d * np.cos(aoa) * t

    def _generate_taps(self, n_samples: int) -> np.ndarray:
        """Generate time-varying MIMO channel taps."""

        n_paths = len(self._powers)
        phase = self._doppler_phase(n_samples, n_paths)

        # Complex Gaussian base fading.
        g = (
            np.random.randn(self.n_rx, self.n_tx, n_paths, n_samples)
            + 1j * np.random.randn(self.n_rx, self.n_tx, n_paths, n_samples)
        ) / np.sqrt(2)

        # Doppler phasor.
        doppler = np.exp(1j * phase)

        # Apply path powers and Doppler.
        return (
            g
            * np.sqrt(self._powers)[np.newaxis, np.newaxis, :, np.newaxis]
            * doppler
        )

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply CDL multipath MIMO channel and AWGN."""

        # Normalize input shape to (n_tx, n_samples).
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]

        # If only one stream is provided for multi-TX channel, replicate it.
        if tx.shape[0] == 1 and self.n_tx > 1:
            tx = np.repeat(tx, self.n_tx, axis=0) / np.sqrt(self.n_tx)

        if tx.shape[0] != self.n_tx:
            raise ValueError(
                f"CDLChannel expected tx shape ({self.n_tx}, n_samples), "
                f"but got {tx.shape}"
            )

        n_samples = tx.shape[1]

        # Generate full MIMO multipath channel.
        h = self._generate_taps(n_samples)

        rx = np.zeros((self.n_rx, n_samples), dtype=complex)

        # Sum contribution from every delayed path.
        for p, delay in enumerate(self._delays_samp):
            d = int(round(delay))

            if d == 0:
                rx += np.einsum("ijk,jk->ik", h[:, :, p, :], tx)

            elif d < n_samples:
                tx_delayed = np.concatenate(
                    [
                        np.zeros((self.n_tx, d), dtype=complex),
                        tx[:, :-d],
                    ],
                    axis=1,
                )

                rx += np.einsum("ijk,jk->ik", h[:, :, p, :], tx_delayed)

        # Effective channel after summing all paths.
        h_eff = h.sum(axis=2)

        # Add AWGN.
        snr_lin = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))

        noise = noise_std * (
            np.random.randn(*rx.shape)
            + 1j * np.random.randn(*rx.shape)
        )

        return rx + noise, h_eff


def ber_bpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """Theoretical BPSK BER over AWGN."""
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * erfc(np.sqrt(ebn0))


def ber_qpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """Theoretical QPSK BER over AWGN with Gray coding."""
    return ber_bpsk_awgn(ebn0_db)


def ber_qam_awgn(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    """Approximate theoretical BER for square M-QAM over AWGN."""

    m = int(np.log2(M))
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    snr_sym = ebn0 * m
    q_arg = np.sqrt(3 * snr_sym / (M - 1))

    return (
        (4 / m)
        * (1 - 1 / np.sqrt(M))
        * 0.5
        * erfc(q_arg / np.sqrt(2))
    )


def ber_rayleigh_bpsk(ebn0_db: np.ndarray) -> np.ndarray:
    """Exact theoretical BPSK BER over flat Rayleigh fading."""
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def ber_rayleigh_qpsk(ebn0_db: np.ndarray) -> np.ndarray:
    """Theoretical QPSK BER over flat Rayleigh fading."""
    return ber_rayleigh_bpsk(ebn0_db)