"""
phy/channel.py — Wireless Channel Models
==========================================
  • AWGNChannel         — signal-power-based SNR (used by ran_link_sim)
  • RayleighChannel     — MIMO flat fading, independent per sample (ran_link_sim)
  • FlatRayleighChannel — single-antenna, per-symbol h, Jakes Doppler (ofdm_link_sim / BER sim)
  • EPAChannel          — Extended Pedestrian A, 7-tap, TS 36.104 Table B.2-1
  • CDLChannel          — Clustered Delay Line A/B/C, TS 38.901 Table 7.7.2

SNR conventions
---------------
  AWGNChannel      — dual mode: snr_db (signal-power) OR ebn0_db+bits_per_symbol (Eb/N0).
  RayleighChannel  — snr_db (signal-power referenced, MIMO).
  FlatRayleighChannel / EPAChannel — ebn0_db + bits_per_symbol (Eb/N0, single-antenna BER).
  CDLChannel       — snr_db (signal-power referenced, MIMO).
  Helper: _ebn0_to_noise_sigma() converts Eb/N0 + Qm → noise σ.

Theoretical BER functions (AWGN + Rayleigh, for validation)
------------------------------------------------------------
  ber_bpsk_awgn, ber_qpsk_awgn, ber_qam_awgn
  ber_rayleigh_bpsk, ber_rayleigh_qpsk
"""

from __future__ import annotations
import numpy as np
from typing import Literal
from scipy.special import erfc


# ─────────────────────────────────────────────────────────────────────────────
# EPA delay profile  (3GPP TS 36.104 Table B.2-1)
# ─────────────────────────────────────────────────────────────────────────────

EPA_TAPS = [
    # (delay_ns, power_dB)
    (0,    0.0),
    (30,  -1.0),
    (70,  -2.0),
    (90,  -3.0),
    (110, -8.0),
    (190, -17.2),
    (410, -20.8),
]

EPA_MAX_DELAY_NS = 410.0   # ns — CP must exceed this


# ─────────────────────────────────────────────────────────────────────────────
# CDL delay / power profiles  (TS 38.901 Table 7.7.2)
# Each entry: (excess_delay_ns, power_dB)
# ─────────────────────────────────────────────────────────────────────────────

_CDL_PROFILES: dict[str, list[tuple[float, float]]] = {
    "CDL-A": [
        (0.0,   -13.4), (9.35,  0.0),   (19.2,  -2.2),  (26.1,  -4.0),
        (40.0,  -6.0),  (66.4,  -8.2),  (80.7,  -9.9),  (109.0, -10.5),
        (200.0, -7.5),  (410.0, -15.9), (750.0, -20.1),
    ],
    "CDL-B": [
        (0.0,    0.0),  (10.0,  -2.2),  (20.0,  -4.0),  (30.0,  -3.2),
        (50.0,  -9.8),  (80.0,  -1.2),  (110.0, -3.4),  (140.0, -5.2),
        (180.0, -7.6),  (230.0, -3.0),  (340.0, -8.9),  (440.0, -11.6),
    ],
    "CDL-C": [
        (0.0,   -4.4),  (65.0,  -1.2),  (150.0,  -3.5), (228.0,  0.0),
        (280.0, -9.9),  (330.0, -16.9), (400.0, -15.2), (490.0, -10.8),
        (750.0, -11.1),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared noise helper (Eb/N0 convention)
# ─────────────────────────────────────────────────────────────────────────────

def _ebn0_to_noise_sigma(ebn0_db: float, bits_per_symbol: int) -> float:
    """
    Convert Eb/N0 (dB) to complex noise std σ assuming unit signal power.
    σ² = 1 / (2 · Eb/N0_lin · bits_per_symbol)  — per real dimension.
    """
    ebn0_lin = 10 ** (ebn0_db / 10)
    return float(np.sqrt(1.0 / (2.0 * ebn0_lin * bits_per_symbol)))


# ─────────────────────────────────────────────────────────────────────────────
# AWGN Channel  (signal-power-referenced SNR)
# ─────────────────────────────────────────────────────────────────────────────

class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    Supports two calling conventions so it works for both the link simulator
    (signal-power SNR) and the BER sweep (Eb/N0 + modulation order):

      AWGNChannel(snr_db=15.0)
          Noise scaled to measured signal power. Used by ran_link_sim.

      AWGNChannel(ebn0_db=10.0, bits_per_symbol=4)
          Noise scaled via Eb/N0 assuming unit signal power. Used by sim_ber.
          When ebn0_db is supplied it takes precedence over snr_db.

    Parameters
    ----------
    snr_db          : SNR in dB, signal-power referenced (default 15.0)
    ebn0_db         : Eb/N0 in dB — overrides snr_db when provided
    bits_per_symbol : Qm, required when using ebn0_db (2=QPSK, 4=16QAM, …)
    """

    name = "AWGN"

    def __init__(
        self,
        snr_db:          float        = 15.0,
        bits_per_symbol: int          = None,
        ebn0_db:         float        = None,
    ):
        self._use_ebn0 = ebn0_db is not None

        if self._use_ebn0:
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol is required when using ebn0_db")
            self.ebn0_db         = ebn0_db
            self.bits_per_symbol = bits_per_symbol
            self._sigma          = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)
            # expose snr_db as an approximate equivalent for inspection
            self.snr_db = ebn0_db + 10 * np.log10(bits_per_symbol)
        else:
            self.snr_db          = snr_db
            self.ebn0_db         = None
            self.bits_per_symbol = bits_per_symbol
            self._sigma          = None

    @property
    def sigma(self) -> float:
        """Noise std (only available in Eb/N0 mode)."""
        if self._sigma is None:
            raise AttributeError("sigma is only defined in ebn0_db mode")
        return self._sigma

    def apply(self, tx: np.ndarray) -> np.ndarray:
        """Add complex AWGN to signal."""
        if self._use_ebn0:
            # Unit-power noise — correct for normalised constellations
            noise = self._sigma * (
                np.random.randn(*tx.shape) + 1j * np.random.randn(*tx.shape)
            )
        else:
            # Signal-power-referenced noise — correct for any amplitude
            snr_lin   = 10 ** (self.snr_db / 10)
            sig_power = np.mean(np.abs(tx) ** 2)
            noise_std = np.sqrt(sig_power / (2 * snr_lin))
            noise     = noise_std * (
                np.random.randn(*tx.shape) + 1j * np.random.randn(*tx.shape)
            )
        return tx + noise


# ─────────────────────────────────────────────────────────────────────────────
# Flat Rayleigh Channel  (MIMO, per-sample, signal-power SNR)
# ─────────────────────────────────────────────────────────────────────────────

class RayleighChannel:
    """
    Flat Rayleigh fading channel — block-fading model.

    One complex Gaussian coefficient h ~ CN(0,1) is drawn per OFDM symbol
    (i.e. per call to apply()) and held constant for all samples within that
    symbol.  This is the standard block-fading assumption required for
    coherent OFDM equalisation: the channel must be flat across the FFT window.

    For SISO link simulation use n_tx=1, n_rx=1.
    For MIMO frequency-domain processing set n_tx/n_rx as needed.

    Parameters
    ----------
    snr_db     : operating SNR in dB (signal-power referenced)
    n_tx, n_rx : antenna counts
    """

    name = "Rayleigh"

    def __init__(self, snr_db: float = 15.0, n_tx: int = 1, n_rx: int = 1):
        self.snr_db = snr_db
        self.n_tx   = n_tx
        self.n_rx   = n_rx

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply block-flat Rayleigh fading + AWGN.

        Parameters
        ----------
        tx : (n_tx, n_samples) or (1, n_samples) or (n_samples,) complex array

        Returns
        -------
        rx     : (n_rx, n_samples) received signal + noise
        h_true : (n_rx, n_tx, n_samples) channel matrix — constant across samples
        """
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]
        if tx.shape[0] != self.n_tx:
            tx = np.broadcast_to(tx, (self.n_tx, tx.shape[1])).copy()
        n_samples = tx.shape[1]

        # Draw ONE coefficient per (rx, tx) pair — constant for all samples
        h_coeff = ((np.random.randn(self.n_rx, self.n_tx)
                    + 1j * np.random.randn(self.n_rx, self.n_tx)) / np.sqrt(2))
        # Broadcast to (n_rx, n_tx, n_samples)
        h = np.broadcast_to(h_coeff[:, :, np.newaxis],
                             (self.n_rx, self.n_tx, n_samples)).copy()

        rx = np.einsum('ijk,jk->ik', h, tx)

        snr_lin   = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))
        noise     = noise_std * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
        return rx + noise, h


# ─────────────────────────────────────────────────────────────────────────────
# Flat Rayleigh Channel  (single-antenna, per-symbol h, Eb/N0)
# ─────────────────────────────────────────────────────────────────────────────

class FlatRayleighChannel:
    """
    Flat (frequency-non-selective) Rayleigh fading — single antenna.
    Channel coefficient h ~ CN(0,1) is constant within one OFDM symbol
    and independent across symbols (block-fading model).
    Uses Eb/N0 noise scaling — suitable for BER sweep simulations.

    Parameters
    ----------
    ebn0_db         : Eb/N0 in dB
    bits_per_symbol : Qm (2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM)
    """

    name = "Flat Rayleigh"

    def __init__(self, ebn0_db: float = 10.0, bits_per_symbol: int = 2):
        self.ebn0_db        = ebn0_db
        self.bits_per_symbol = bits_per_symbol
        self._sigma         = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)
        self._h: complex    = 1.0 + 0j

    def new_realization(self):
        """Draw a fresh channel coefficient (call once per OFDM symbol)."""
        self._h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    @property
    def h(self) -> complex:
        """Current channel coefficient."""
        return self._h

    @property
    def sigma(self) -> float:
        return self._sigma

    def apply(self, tx: np.ndarray, per_sample: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply flat Rayleigh fading + AWGN.

        Parameters
        ----------
        tx         : (N,) complex transmit samples
        per_sample : if True, draw independent h per sample (fast fading / IID model)

        Returns
        -------
        rx     : (N,) received signal
        h_used : per-sample h array (N,) or scalar h broadcast to (N,)
        """
        if per_sample:
            h = (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx))) / np.sqrt(2)
        else:
            self.new_realization()
            h = self._h

        noise = self._sigma * (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx)))
        rx    = h * tx + noise
        return rx, h


# ─────────────────────────────────────────────────────────────────────────────
# EPA Multipath Channel  (TS 36.104 Table B.2-1)
# ─────────────────────────────────────────────────────────────────────────────

class EPAChannel:
    """
    Extended Pedestrian A (EPA) multipath channel.
    7-tap delay profile per TS 36.104 Table B.2-1.
    Max delay spread: 410 ns — CP must exceed this value.

    Each tap has an independent CN(0,1) coefficient with Jakes Doppler applied.
    Uses Eb/N0 noise scaling.

    Parameters
    ----------
    ebn0_db         : Eb/N0 in dB
    bits_per_symbol : Qm
    fs_hz           : sample rate in Hz (used to convert tap delays to samples)
    velocity_kmh    : UE velocity for Doppler (default 3 km/h = pedestrian)
    carrier_freq_hz : carrier frequency in Hz for Doppler shift calculation
    """

    name = "EPA"
    MAX_DELAY_NS = EPA_MAX_DELAY_NS

    def __init__(
        self,
        ebn0_db:         float = 10.0,
        bits_per_symbol: int   = 2,
        fs_hz:           float = 30.72e6,
        velocity_kmh:    float = 3.0,
        carrier_freq_hz: float = 2.0e9,
    ):
        self.ebn0_db         = ebn0_db
        self.bits_per_symbol = bits_per_symbol
        self.fs              = fs_hz
        self._sigma          = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)

        # Build tap delay (samples) and normalised amplitude arrays
        delays_ns         = np.array([t[0] for t in EPA_TAPS], dtype=float)
        powers_db         = np.array([t[1] for t in EPA_TAPS], dtype=float)
        self._delays_samp = np.round(delays_ns * 1e-9 * fs_hz).astype(int)
        powers_lin        = 10 ** (powers_db / 10.0)
        self._tap_gains   = np.sqrt(powers_lin / powers_lin.sum())  # normalised amplitude

        # Max Doppler frequency
        speed_mps = velocity_kmh / 3.6
        self._fd  = speed_mps * carrier_freq_hz / 3e8

    @property
    def max_delay_samples(self) -> int:
        return int(self._delays_samp.max())

    @property
    def max_delay_us(self) -> float:
        return float(self._delays_samp.max() / self.fs * 1e6)

    def _doppler_coeff(self, n_samples: int) -> np.ndarray:
        """
        Jakes Doppler phasor per tap: shape (n_taps, n_samples).
        Each tap gets an independent random angle of arrival.
        """
        n_taps = len(self._tap_gains)
        aoa    = np.random.uniform(-np.pi, np.pi, (n_taps, 1))
        t      = np.arange(n_samples) / self.fs
        return np.exp(1j * 2 * np.pi * self._fd * np.cos(aoa) * t)

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply EPA multipath fading + AWGN.

        Parameters
        ----------
        tx : (N,) complex transmit signal

        Returns
        -------
        rx        : (N,) received signal (same length; delayed taps zero-pad the tail)
        h_impulse : (n_taps,) complex tap amplitudes at t=0 (useful for channel inspection)
        """
        N_in    = len(tx)
        rx      = np.zeros(N_in, dtype=complex)
        doppler = self._doppler_coeff(N_in)

        h_impulse = np.zeros(len(self._tap_gains), dtype=complex)

        for p, (d, g) in enumerate(zip(self._delays_samp, self._tap_gains)):
            h_tap        = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * g
            h_impulse[p] = h_tap
            h_t          = h_tap * doppler[p]   # (N_in,) time-varying tap

            if d == 0:
                rx += h_t * tx
            elif d < N_in:
                rx[d:] += h_t[d:] * tx[:N_in - d]
            # taps with d >= N_in contribute nothing (signal not long enough)

        noise = self._sigma * (np.random.randn(N_in) + 1j * np.random.randn(N_in))
        return rx + noise, h_impulse


# ─────────────────────────────────────────────────────────────────────────────
# CDL Channel  (TS 38.901 Table 7.7.2)
# ─────────────────────────────────────────────────────────────────────────────

class CDLChannel:
    """
    Clustered Delay Line channel — CDL-A, CDL-B, or CDL-C.
    Time-varying multipath with per-path Jakes Doppler.
    Supports MIMO via n_tx / n_rx.
    Uses signal-power-referenced SNR.

    Parameters
    ----------
    model        : 'CDL-A', 'CDL-B', or 'CDL-C'
    snr_db       : operating SNR in dB
    velocity_kmh : UE velocity for Doppler spread
    scs_hz       : subcarrier spacing in Hz (sets sample rate = scs_hz * n_fft)
    n_fft        : FFT size
    n_tx, n_rx   : antenna counts
    """

    SPEED_OF_LIGHT  = 3e8
    CARRIER_FREQ_HZ = 3.5e9   # FR1 mid-band default

    def __init__(
        self,
        model:        Literal['CDL-A', 'CDL-B', 'CDL-C'] = 'CDL-A',
        snr_db:       float = 15.0,
        velocity_kmh: float = 30.0,
        scs_hz:       float = 30e3,
        n_fft:        int   = 2048,
        n_tx:         int   = 1,
        n_rx:         int   = 1,
    ):
        if model not in _CDL_PROFILES:
            raise ValueError(f"Unknown CDL model '{model}'. Choose CDL-A, CDL-B, or CDL-C.")
        self.model  = model
        self.snr_db = snr_db
        self.n_tx   = n_tx
        self.n_rx   = n_rx

        self._sample_rate = scs_hz * n_fft
        self._f_d = (velocity_kmh / 3.6) * self.CARRIER_FREQ_HZ / self.SPEED_OF_LIGHT

        profile              = _CDL_PROFILES[model]
        delays_ns, powers_db = zip(*profile)
        self._delays_samp    = np.array([d * 1e-9 * self._sample_rate for d in delays_ns])
        powers_lin           = 10 ** (np.array(powers_db) / 10)
        self._powers         = powers_lin / powers_lin.sum()   # normalised

    def _doppler_phase(self, n_samples: int, n_paths: int) -> np.ndarray:
        """Jakes Doppler phase per path: (n_paths, n_samples)."""
        t   = np.arange(n_samples) / self._sample_rate
        aoa = np.random.uniform(-np.pi, np.pi, (n_paths, 1))
        return 2 * np.pi * self._f_d * np.cos(aoa) * t

    def _generate_taps(self, n_samples: int) -> np.ndarray:
        """
        Generate channel taps: (n_rx, n_tx, n_paths, n_samples).

        Block-fading model: one complex Gaussian coefficient per (rx, tx, path)
        drawn at the start of the symbol, then modulated by the Doppler phasor.
        This preserves inter-sample coherence so OFDM equalisation works —
        the FFT averages over a slowly-varying (or constant) channel.
        """
        n_paths = len(self._powers)
        phase   = self._doppler_phase(n_samples, n_paths)  # (n_paths, n_samples)
        doppler = np.exp(1j * phase)                        # (n_paths, n_samples)

        # Draw ONE coefficient per (rx, tx, path) — constant within the symbol
        g = ((np.random.randn(self.n_rx, self.n_tx, n_paths, 1)
              + 1j * np.random.randn(self.n_rx, self.n_tx, n_paths, 1))
             / np.sqrt(2))   # (n_rx, n_tx, n_paths, 1) broadcast over n_samples

        # Scale by path power, apply Doppler time variation
        return (g * np.sqrt(self._powers)[np.newaxis, np.newaxis, :, np.newaxis]
                  * doppler[np.newaxis, np.newaxis, :, :])

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply CDL multipath + AWGN to a time-domain signal.

        Parameters
        ----------
        tx : (n_tx, n_samples) or (1, n_samples) or (n_samples,) complex array.
             Always broadcast to (n_tx, n_samples) internally.

        Returns
        -------
        rx     : (n_rx, n_samples) received signal
        h_eff  : (n_rx, n_tx, n_samples) effective per-sample channel (paths summed)
        """
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]                # (1, n_samples)
        # Broadcast to (n_tx, n_samples) so delayed-tap concatenation is consistent
        if tx.shape[0] != self.n_tx:
            tx = np.broadcast_to(tx, (self.n_tx, tx.shape[1])).copy()
        n_samples = tx.shape[1]

        h  = self._generate_taps(n_samples)       # (n_rx, n_tx, n_paths, n_samples)
        rx = np.zeros((self.n_rx, n_samples), dtype=complex)

        for p, delay in enumerate(self._delays_samp):
            d = int(round(delay))
            if d == 0:
                rx += np.einsum('ijk,jk->ik', h[:, :, p, :], tx)
            elif d < n_samples:
                tx_delayed = np.concatenate(
                    [np.zeros((self.n_tx, d), dtype=complex), tx[:, :-d]], axis=1
                )
                rx += np.einsum('ijk,jk->ik', h[:, :, p, :], tx_delayed)

        h_eff     = h.sum(axis=2)                 # (n_rx, n_tx, n_samples)
        snr_lin   = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))
        noise     = noise_std * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
        return rx + noise, h_eff


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical BER functions
# ─────────────────────────────────────────────────────────────────────────────

def ber_bpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """BPSK AWGN: BER = Q(√(2·Eb/N0))"""
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * erfc(np.sqrt(ebn0))


def ber_qpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """QPSK AWGN: identical to BPSK with Gray coding."""
    return ber_bpsk_awgn(ebn0_db)


def ber_qam_awgn(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    """
    Gray-coded M-QAM AWGN BER (approximate).
    BER ≈ (4/log2(M))·(1 - 1/√M)·Q(√(3·log2(M)·Eb/N0 / (M-1)))
    Valid for M = 4, 16, 64, 256.
    """
    m       = int(np.log2(M))
    ebn0    = 10 ** (np.asarray(ebn0_db) / 10)
    snr_sym = ebn0 * m
    Q_arg   = np.sqrt(3 * snr_sym / (M - 1))
    return (4 / m) * (1 - 1 / np.sqrt(M)) * 0.5 * erfc(Q_arg / np.sqrt(2))


def ber_rayleigh_bpsk(ebn0_db: np.ndarray) -> np.ndarray:
    """Flat Rayleigh BPSK BER (exact closed form)."""
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def ber_rayleigh_qpsk(ebn0_db: np.ndarray) -> np.ndarray:
    """Flat Rayleigh QPSK BER (same as BPSK with Gray coding)."""
    return ber_rayleigh_bpsk(ebn0_db)