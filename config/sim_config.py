from dataclasses import dataclass, field
from typing import Literal

# @dataclass: automatically creates __init__, __repr__, etc.
# field(init=False): marks variables that are computed later, not passed in
# Literal: restricts values to specific allowed strings

# “Make this class mainly for storing data.”
@dataclass
class SimConfig:
    # ── OFDM ──────────────────────────────────────────────────────────────────
    numerology: int = 1          # μ: 0=15kHz, 1=30kHz, 2=60kHz subcarrier spacing
    n_prb: int = 52              # number of Physical Resource Blocks
    n_fft: int = 2048            # FFT size for OFDM modulation
    cp_len: int = 144            # cyclic prefix length (samples) (helps with multipath)
    n_symb_per_slot: int = 14   # OFDM symbols per slot. Standard 5G slot = 14 OFDM symbols

    # ── Modulation ────────────────────────────────────────────────────────────
    mcs: int = 16                # MCS index 0-28 (3GPP TS 38.214 Table 5.1.3.1-2). MCS = Modulation and Coding Scheme (defines data rate & robustness)
    n_layers: int = 1            # spatial layers / rank indicator (RI). Number of spatial streams (MIMO rank)

    # ── Channel ───────────────────────────────────────────────────────────────
    #Restricts to valid channel types:
    # AWGN: simple noise
    # Rayleigh: fading
    # CDL: realistic 5G channel models
    # Signal-to-noise ratio
    # Affects Doppler (mobility effects)
    channel_model: Literal[
        'AWGN', 'Rayleigh', 'CDL-A', 'CDL-B', 'CDL-C'
    ] = 'CDL-A'
    snr_db: float = 15.0        # target operating SNR (dB)
    velocity_kmh: float = 30.0  # UE velocity for Doppler spread (km/h)

    # ── MIMO ──────────────────────────────────────────────────────────────────
    # 4 transmit antennas, 2 receive antennas
    # Detection algorithm:
    #   MRC: simple combining
    #   ZF: zero-forcing
    #   MMSE: balanced (default)
    #   SIC: advanced
    n_tx: int = 4                # transmit antennas
    n_rx: int = 2                # receive antennas
    detector: Literal[
        'MRC', 'ZF', 'MMSE', 'SIC'
    ] = 'MMSE'

    # ── Derived (computed post-init) ──────────────────────────────────────────
    scs_khz: int = field(init=False)          # subcarrier spacing in kHz
    n_subcarriers: int = field(init=False)    # total subcarriers (n_prb * 12)
    slot_duration_us: float = field(init=False)  # slot duration in µs

    def __post_init__(self):
        self.scs_khz = 15 * (2 ** self.numerology)
        # Total subcarriers:
        # 52 PRBs → 52 × 12 = 624 subcarriers
        self.n_subcarriers = self.n_prb * 12
        # one slot = 14 symbols; symbol duration = 1 / scs
        # Symbol duration (µs):
        # Example: 30 kHz → 1000 / 30 ≈ 33.33 µs
        symbol_duration_us = 1e3 / self.scs_khz          # µs
        # Slot duration:
        # 14 symbols → ~466.7 µs
        self.slot_duration_us = self.n_symb_per_slot * symbol_duration_us

    # ── Helpers ───────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            "=== SimConfig ===",
            f"  Numerology      : μ={self.numerology}  ({self.scs_khz} kHz SCS)",
            f"  PRBs / SCs      : {self.n_prb} PRBs  →  {self.n_subcarriers} subcarriers",
            f"  FFT size        : {self.n_fft}",
            f"  CP length       : {self.cp_len} samples",
            f"  Symbols/slot    : {self.n_symb_per_slot}",
            f"  Slot duration   : {self.slot_duration_us:.2f} µs",
            f"  MCS index       : {self.mcs}",
            f"  Spatial layers  : {self.n_layers}",
            f"  Channel model   : {self.channel_model}",
            f"  SNR             : {self.snr_db} dB",
            f"  UE velocity     : {self.velocity_kmh} km/h",
            f"  Antennas (Tx/Rx): {self.n_tx} / {self.n_rx}",
            f"  Detector        : {self.detector}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = SimConfig()
    print(cfg.summary())
