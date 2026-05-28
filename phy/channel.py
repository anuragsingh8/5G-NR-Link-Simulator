"""
phy/channel.py — Wireless Channel Models
==========================================
  • AWGNChannel         — dual-mode: snr_db (signal-power) or ebn0_db+Qm (Eb/N0)
  • RayleighChannel     — MIMO block-flat fading, one h per OFDM symbol (ran_link_sim)
  • FlatRayleighChannel — single-antenna, per-symbol h, ebn0_db (BER sim)
  • EPAChannel          — Extended Pedestrian A, 7-tap, TS 36.104 Table B.2-1
  • CDLChannel          — Clustered Delay Line A/B/C, TS 38.901 Table 7.7.2

SNR conventions
---------------
  AWGNChannel      — snr_db (signal-power) OR ebn0_db+bits_per_symbol (Eb/N0)
  RayleighChannel  — snr_db (signal-power referenced, MIMO)
  FlatRayleighChannel / EPAChannel — ebn0_db + bits_per_symbol (Eb/N0)
  CDLChannel       — snr_db (signal-power referenced, MIMO)
  Helper: _ebn0_to_noise_sigma() converts Eb/N0 + Qm to noise sigma.

Fading model note
-----------------
  Both RayleighChannel and CDLChannel use a block-fading model: one complex
  Gaussian coefficient is drawn per OFDM symbol and held constant for all
  samples within that symbol. This preserves OFDM inter-sample coherence and
  makes frequency-domain equalisation correct. Doppler phase variation is
  applied on top of the stable base coefficient in CDLChannel.

Theoretical BER functions (for sim_ber.py validation)
------------------------------------------------------
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
# TDL profiles  (TS 38.901 Table 7.7.2 — normalised delays)
# (normalised_delay, power_dB)
# Actual delay = norm_delay * DS_rms_ns
# ─────────────────────────────────────────────────────────────────────────────

_TDL_PROFILES: dict[str, list[tuple[float, float]]] = {
    "TDL-A": [
        (0.0000, -13.4), (0.3819,   0.0), (0.4025,  -2.2), (0.5868,  -4.0),
        (0.4610,  -6.0), (0.5375,  -8.2), (0.6708,  -9.9), (0.5750, -10.5),
        (0.7618,  -7.5), (1.5375, -15.9), (1.8978, -20.1), (2.2242, -22.9),
        (2.1718, -22.4), (2.4942, -18.6), (2.5119, -20.8), (3.0582, -22.6),
        (4.0810, -22.3), (4.4579, -25.6), (4.5695, -20.1), (4.7966, -29.0),
        (5.0066, -24.0), (5.3043, -45.9), (9.6586, -35.7),
    ],
    "TDL-B": [
        (0.0000,   0.0), (0.1072,  -2.2), (0.2155,  -4.0), (0.2095,  -3.2),
        (0.2870,  -9.8), (0.2986,  -1.2), (0.3752,  -3.4), (0.5055,  -5.2),
        (0.3681,  -7.6), (0.3697,  -3.0), (0.5700,  -8.9), (0.5283, -11.6),
        (1.1021, -10.4), (1.2756, -11.3), (1.5474, -12.7), (1.7842, -16.2),
        (2.0169, -18.3), (2.8294, -18.9), (3.0219, -16.6), (3.6187, -19.9),
        (4.1067, -29.7), (4.2790, -24.8),
    ],
    "TDL-C": [
        (0.0000,  -4.4), (0.2099,  -1.2), (0.2219,   3.4), (0.2329,   0.0),
        (0.2176,  -6.2), (0.6366,  -4.0), (0.6448,  -6.0), (0.6560,  -8.2),
        (0.6584,  -9.9), (0.7935, -10.5), (0.8213,  -7.5), (0.9336, -15.9),
        (1.2285, -20.1), (1.3083, -22.9), (2.1704, -22.4), (2.7105, -18.6),
        (4.2589, -20.8), (4.6003, -22.6), (5.4902, -22.3), (5.6077, -25.6),
        (6.3065, -20.1), (6.6374, -29.0), (7.0427, -24.0), (8.6523, -45.9),
    ],
    "TDL-D": [
        (0.0000,  -0.2), (0.0350, -13.5), (0.6120, -18.8), (1.6710, -21.0),
        (1.5420, -22.8), (5.1150, -17.9), (2.8930, -20.1), (3.2610, -21.9),
        (1.9410, -22.9), (7.4050, -27.8), (7.0600, -23.6), (9.5280, -24.8),
        (3.1310, -30.0),
    ],
    "TDL-E": [
        (0.0000,  -0.03), (0.5133, -22.03), (0.5440, -15.8), (0.5630, -18.1),
        (0.5440, -19.8),  (0.7112, -22.9),  (1.9092, -22.4), (1.9293, -18.6),
        (1.9589, -20.8),  (2.6426, -22.6),  (3.7136, -22.3), (5.4524, -25.6),
        (12.0034,-20.1),  (20.6419,-29.0),  (20.6419,-24.0), (21.0345,-45.9),
        (22.6535,-35.7),  (27.7771,-41.7),  (39.1723,-40.0), (66.7334,-38.0),
        (108.7234,-49.0), (301.6519,-50.0), (600.0000,-50.0),
    ],
}

_TDL_LOS_K   = {'TDL-D': 13.3, 'TDL-E': 22.0}
_TDL_DEFAULT_DS = {
    'TDL-A': 100.0, 'TDL-B': 100.0, 'TDL-C': 300.0,
    'TDL-D':  30.0, 'TDL-E':  30.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared noise helper (Eb/N0 convention)
# ─────────────────────────────────────────────────────────────────────────────

def _ebn0_to_noise_sigma(ebn0_db: float, bits_per_symbol: int) -> float:
    """
    Convert Eb/N0 (dB) to complex noise std sigma assuming unit signal power.
    sigma^2 = 1 / (2 * Eb/N0_lin * bits_per_symbol) per real dimension.
    """
    ebn0_lin = 10 ** (ebn0_db / 10)
    return float(np.sqrt(1.0 / (2.0 * ebn0_lin * bits_per_symbol)))


# ─────────────────────────────────────────────────────────────────────────────
# AWGN Channel
# ─────────────────────────────────────────────────────────────────────────────

class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    Two calling conventions:

      AWGNChannel(snr_db=15.0)
          Noise scaled to measured signal power. Used by ran_link_sim.

      AWGNChannel(ebn0_db=10.0, bits_per_symbol=4)
          Noise from Eb/N0, assumes unit signal power. Used by sim_ber.

    Parameters
    ----------
    snr_db          : SNR in dB, signal-power referenced (default 15.0)
    bits_per_symbol : Qm, required when using ebn0_db
    ebn0_db         : Eb/N0 in dB — overrides snr_db when provided
    """

    name = "AWGN"

    def __init__(
        self,
        snr_db:          float = 15.0,
        bits_per_symbol: int   = None,
        ebn0_db:         float = None,
    ):
        self._use_ebn0 = ebn0_db is not None

        if self._use_ebn0:
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol is required when using ebn0_db")
            self.ebn0_db         = ebn0_db
            self.bits_per_symbol = bits_per_symbol
            self._sigma          = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)
            self.snr_db          = ebn0_db + 10 * np.log10(bits_per_symbol)
        else:
            self.snr_db          = snr_db
            self.bits_per_symbol = bits_per_symbol
            self._sigma          = None

    @property
    def sigma(self) -> float:
        if self._sigma is None:
            raise AttributeError("sigma is only defined in ebn0_db mode")
        return self._sigma

    def apply(self, tx: np.ndarray) -> np.ndarray:
        """Add complex AWGN to signal."""
        if self._use_ebn0:
            noise = self._sigma * (
                np.random.randn(*tx.shape) + 1j * np.random.randn(*tx.shape)
            )
        else:
            snr_lin   = 10 ** (self.snr_db / 10)
            sig_power = np.mean(np.abs(tx) ** 2)
            noise_std = np.sqrt(sig_power / (2 * snr_lin))
            noise     = noise_std * (
                np.random.randn(*tx.shape) + 1j * np.random.randn(*tx.shape)
            )
        return tx + noise


# ─────────────────────────────────────────────────────────────────────────────
# Rayleigh Channel — MIMO block-fading (one h per OFDM symbol)
# ─────────────────────────────────────────────────────────────────────────────

class RayleighChannel:
    """
    Flat Rayleigh fading channel — block-fading model.

    One complex Gaussian coefficient h ~ CN(0,1) is drawn per call to apply(),
    held constant for all samples in that call (i.e. one OFDM symbol).
    This is the standard block-fading assumption for coherent OFDM equalisation.

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
        rx     : (n_rx, n_samples) received signal
        h_true : (n_rx, n_tx, n_samples) channel — constant across samples
        """
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]
        if tx.shape[0] != self.n_tx:
            tx = np.broadcast_to(tx, (self.n_tx, tx.shape[1])).copy()
        n_samples = tx.shape[1]

        # One coefficient per (rx, tx) pair, constant within the symbol
        h_coeff = ((np.random.randn(self.n_rx, self.n_tx)
                    + 1j * np.random.randn(self.n_rx, self.n_tx)) / np.sqrt(2))
        h = np.broadcast_to(
            h_coeff[:, :, np.newaxis], (self.n_rx, self.n_tx, n_samples)
        ).copy()

        rx        = np.einsum('ijk,jk->ik', h, tx)
        snr_lin   = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))
        noise     = noise_std * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
        return rx + noise, h


# ─────────────────────────────────────────────────────────────────────────────
# Flat Rayleigh Channel — single-antenna, Eb/N0, for BER simulation
# ─────────────────────────────────────────────────────────────────────────────

class FlatRayleighChannel:
    """
    Flat Rayleigh fading — single antenna, block-fading, Eb/N0 convention.
    h ~ CN(0,1) constant within one OFDM symbol, independent across symbols.
    Used by sim_ber for BER vs Eb/N0 curves.

    Parameters
    ----------
    ebn0_db         : Eb/N0 in dB
    bits_per_symbol : Qm
    """

    name = "Flat Rayleigh"

    def __init__(self, ebn0_db: float = 10.0, bits_per_symbol: int = 2):
        self.ebn0_db         = ebn0_db
        self.bits_per_symbol = bits_per_symbol
        self._sigma          = _ebn0_to_noise_sigma(ebn0_db, bits_per_symbol)
        self._h: complex     = 1.0 + 0j

    def new_realization(self):
        """Draw a fresh channel coefficient (call once per OFDM symbol)."""
        self._h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    @property
    def h(self) -> complex:
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
        per_sample : if True, independent h per sample (fast fading)

        Returns
        -------
        rx     : (N,) received signal
        h_used : channel coefficient(s) used
        """
        if per_sample:
            h = (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx))) / np.sqrt(2)
        else:
            self.new_realization()
            h = self._h

        noise = self._sigma * (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx)))
        return h * tx + noise, h


# ─────────────────────────────────────────────────────────────────────────────
# EPA Multipath Channel  (TS 36.104 Table B.2-1)
# ─────────────────────────────────────────────────────────────────────────────

class EPAChannel:
    """
    Extended Pedestrian A (EPA) multipath channel.
    7-tap delay profile per TS 36.104 Table B.2-1. Max delay 410 ns.
    CP must exceed 410 ns — satisfied by mu=0,1,2 but not mu=3,4.
    Jakes Doppler applied per tap. Eb/N0 noise convention.

    Parameters
    ----------
    ebn0_db         : Eb/N0 in dB
    bits_per_symbol : Qm
    fs_hz           : sample rate in Hz
    velocity_kmh    : UE velocity for Doppler (default 3 km/h pedestrian)
    carrier_freq_hz : carrier frequency for Doppler calculation
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

        delays_ns         = np.array([t[0] for t in EPA_TAPS], dtype=float)
        powers_db         = np.array([t[1] for t in EPA_TAPS], dtype=float)
        self._delays_samp = np.round(delays_ns * 1e-9 * fs_hz).astype(int)
        powers_lin        = 10 ** (powers_db / 10.0)
        self._tap_gains   = np.sqrt(powers_lin / powers_lin.sum())

        speed_mps = velocity_kmh / 3.6
        self._fd  = speed_mps * carrier_freq_hz / 3e8

    @property
    def max_delay_samples(self) -> int:
        return int(self._delays_samp.max())

    @property
    def max_delay_us(self) -> float:
        return float(self._delays_samp.max() / self.fs * 1e6)

    def _doppler_coeff(self, n_samples: int) -> np.ndarray:
        """Jakes Doppler phasor per tap: (n_taps, n_samples)."""
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
        rx        : (N,) received signal
        h_impulse : (n_taps,) complex tap amplitudes at t=0
        """
        N_in    = len(tx)
        rx      = np.zeros(N_in, dtype=complex)
        doppler = self._doppler_coeff(N_in)

        h_impulse = np.zeros(len(self._tap_gains), dtype=complex)
        for p, (d, g) in enumerate(zip(self._delays_samp, self._tap_gains)):
            h_tap        = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * g
            h_impulse[p] = h_tap
            h_t          = h_tap * doppler[p]
            if d == 0:
                rx += h_t * tx
            elif d < N_in:
                rx[d:] += h_t[d:] * tx[:N_in - d]

        noise = self._sigma * (np.random.randn(N_in) + 1j * np.random.randn(N_in))
        return rx + noise, h_impulse


# ─────────────────────────────────────────────────────────────────────────────
# CDL Channel  (TS 38.901 Table 7.7.2)
# ─────────────────────────────────────────────────────────────────────────────

class CDLChannel:
    """
    Clustered Delay Line channel — CDL-A, CDL-B, or CDL-C.

    Block-fading model: one complex Gaussian coefficient per (rx, tx, path)
    drawn at construction of each OFDM symbol, then modulated by Jakes Doppler.
    This preserves OFDM inter-sample coherence required for equalisation.

    Parameters
    ----------
    model        : 'CDL-A', 'CDL-B', or 'CDL-C'
    snr_db       : operating SNR in dB (signal-power referenced)
    velocity_kmh : UE velocity for Doppler spread
    scs_hz       : subcarrier spacing in Hz (sets sample rate = scs_hz * n_fft)
    n_fft        : FFT size
    n_tx, n_rx   : antenna counts
    """

    SPEED_OF_LIGHT  = 3e8
    CARRIER_FREQ_HZ = 3.5e9

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
        self._powers         = powers_lin / powers_lin.sum()

    def _doppler_phase(self, n_samples: int, n_paths: int) -> np.ndarray:
        """Jakes Doppler phase per path: (n_paths, n_samples)."""
        t   = np.arange(n_samples) / self._sample_rate
        aoa = np.random.uniform(-np.pi, np.pi, (n_paths, 1))
        return 2 * np.pi * self._f_d * np.cos(aoa) * t

    def _generate_taps(self, n_samples: int) -> np.ndarray:
        """
        Generate channel taps: (n_rx, n_tx, n_paths, n_samples).

        Block-fading: one complex Gaussian coefficient per (rx, tx, path),
        modulated by Jakes Doppler phase across the symbol duration.
        """
        n_paths = len(self._powers)
        phase   = self._doppler_phase(n_samples, n_paths)  # (n_paths, n_samples)
        doppler = np.exp(1j * phase)                        # (n_paths, n_samples)

        # ONE coefficient per (rx, tx, path) — constant base, Doppler modulated
        g = ((np.random.randn(self.n_rx, self.n_tx, n_paths, 1)
              + 1j * np.random.randn(self.n_rx, self.n_tx, n_paths, 1))
             / np.sqrt(2))

        return (g
                * np.sqrt(self._powers)[np.newaxis, np.newaxis, :, np.newaxis]
                * doppler[np.newaxis, np.newaxis, :, :])

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply CDL multipath + AWGN.

        Parameters
        ----------
        tx : (n_tx, n_samples) or (1, n_samples) or (n_samples,) complex array

        Returns
        -------
        rx    : (n_rx, n_samples) received signal
        h_eff : (n_rx, n_tx, n_samples) effective per-sample channel (paths summed)
        """
        if tx.ndim == 1:
            tx = tx[np.newaxis, :]
        if tx.shape[0] != self.n_tx:
            tx = np.broadcast_to(tx, (self.n_tx, tx.shape[1])).copy()
        n_samples = tx.shape[1]

        h  = self._generate_taps(n_samples)
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

        h_eff     = h.sum(axis=2)
        snr_lin   = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin))
        noise     = noise_std * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))
        return rx + noise, h_eff


# ─────────────────────────────────────────────────────────────────────────────
# TDL Channel  (TS 38.901 Table 7.7.2)
# ─────────────────────────────────────────────────────────────────────────────

class TDLChannel:
    """
    Tapped Delay Line channel — TDL-A/B/C/D/E (TS 38.901 Table 7.7.2).

    Standard model for 3GPP link-level evaluation (TS 38.101-4).
    Frequency-selective, no spatial correlation (SISO by design).

    LOS models (D, E) use a Rician first tap with spec K-factor.
    Tap delays = normalised_delay × delay_spread_ns.

    Parameters
    ----------
    model           : 'TDL-A' | 'TDL-B' | 'TDL-C' | 'TDL-D' | 'TDL-E'
    snr_db          : operating SNR (signal-power referenced)
    velocity_kmh    : UE velocity for Doppler
    fs_hz           : sample rate in Hz
    delay_spread_ns : RMS delay spread (None = model default)
    carrier_freq_hz : carrier frequency for Doppler shift
    """

    SPEED_OF_LIGHT = 3e8
    name = "TDL"

    def __init__(
        self,
        model:           str   = 'TDL-A',
        snr_db:          float = 15.0,
        velocity_kmh:    float = 30.0,
        fs_hz:           float = 30.72e6,
        delay_spread_ns: float = None,
        carrier_freq_hz: float = 3.5e9,
    ):
        if model not in _TDL_PROFILES:
            raise ValueError(f"Unknown TDL model \'{model}\'. Use TDL-A/B/C/D/E.")
        self.model   = model
        self.snr_db  = snr_db
        self.fs      = fs_hz
        self._f_d    = (velocity_kmh / 3.6) * carrier_freq_hz / self.SPEED_OF_LIGHT
        self._is_los = model in _TDL_LOS_K
        self._K_lin  = 10 ** (_TDL_LOS_K.get(model, 0.0) / 10)

        ds_ns = delay_spread_ns if delay_spread_ns is not None \
                else _TDL_DEFAULT_DS[model]

        prof              = _TDL_PROFILES[model]
        norm_d            = np.array([t[0] for t in prof])
        pdb               = np.array([t[1] for t in prof])
        self._delays_s    = np.round(norm_d * ds_ns * 1e-9 * fs_hz).astype(int)
        pl                = 10 ** (pdb / 10.0)
        self._tap_pow     = pl / pl.sum()
        self.n_taps       = len(self._tap_pow)
        self.max_delay_us = float(self._delays_s.max() / fs_hz * 1e6)

    def _draw_taps(self, n_samples: int) -> np.ndarray:
        aoa     = np.random.uniform(-np.pi, np.pi, (self.n_taps, 1))
        t       = np.arange(n_samples) / self.fs
        doppler = np.exp(1j * 2 * np.pi * self._f_d * np.cos(aoa) * t)
        g       = ((np.random.randn(self.n_taps, 1)
                    + 1j * np.random.randn(self.n_taps, 1)) / np.sqrt(2))
        if self._is_los:
            K          = self._K_lin
            los_phase  = 2 * np.pi * np.random.uniform()
            g[0]       = np.sqrt(K/(K+1)) * np.exp(1j*los_phase) + g[0]/np.sqrt(K+1)
        return g * np.sqrt(self._tap_pow[:, np.newaxis]) * doppler

    def apply(self, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply TDL multipath fading + AWGN.

        Parameters
        ----------
        tx : (N,) complex signal

        Returns
        -------
        rx     : (N,) received signal
        h_taps : (n_taps,) tap amplitudes at t=0
        """
        tx_1d = tx.ravel()
        N     = len(tx_1d)
        rx    = np.zeros(N, dtype=complex)
        taps  = self._draw_taps(N)
        h_t   = taps[:, 0].copy()
        for p in range(self.n_taps):
            d = int(self._delays_s[p])
            if d == 0:
                rx += taps[p] * tx_1d
            elif d < N:
                rx[d:] += taps[p, d:] * tx_1d[:N - d]
        snr_lin   = 10 ** (self.snr_db / 10)
        sig_power = np.mean(np.abs(rx) ** 2)
        noise_std = np.sqrt(sig_power / (2 * snr_lin)) if sig_power > 0 else 0.0
        noise     = noise_std * (np.random.randn(N) + 1j * np.random.randn(N))
        return rx + noise, h_t


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical BER functions
# ─────────────────────────────────────────────────────────────────────────────

def ber_bpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """BPSK AWGN: BER = Q(sqrt(2 Eb/N0))"""
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * erfc(np.sqrt(ebn0))


def ber_qpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """QPSK AWGN: identical to BPSK with Gray coding."""
    return ber_bpsk_awgn(ebn0_db)


def ber_qam_awgn(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    """
    Gray-coded M-QAM AWGN BER (approximate).
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