"""
config/sim_config.py — Simulation Configuration
=================================================
3GPP TS 38.101, TS 38.211, TS 38.214

SimConfig is the single source of truth for all simulation parameters.
All derived values (FFT size, CP length, slot duration, subcarrier count)
are computed automatically from the chosen numerology and bandwidth.

Supported bandwidths per numerology (TS 38.101 Table 5.3.5-1, FR1):
  μ=0 (15 kHz):  5, 10, 15, 20, 25, 30, 40, 50 MHz
  μ=1 (30 kHz):  5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100 MHz
  μ=2 (60 kHz): 10, 15, 20, 25, 30, 40, 50, 60, 80, 100 MHz

Supported MCS tables (TS 38.214):
  'qam64'  — Table 5.1.3.1-2 (MCS 0–28, up to 64-QAM,  default DL)
  'qam256' — Table 5.1.3.1-3 (MCS 0–27, up to 256-QAM, high-throughput)
  'qam64lr'— Table 5.1.3.1-1 (MCS 0–28, low-code-rate QPSK)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# 3GPP TS 38.101 Table 5.3.5-1 — n_PRB per BW per numerology (FR1)
# ─────────────────────────────────────────────────────────────────────────────

_N_PRB_TABLE: dict[int, dict[int, int]] = {
    #  μ : {bw_mhz: n_prb}  (TS 38.101 Table 5.3.5-1, FR1 + FR2)
    0: {5:25,  10:52,  15:79,  20:106, 25:133, 30:160, 40:216, 50:270},
    1: {5:11,  10:24,  15:38,  20:51,  25:65,  30:78,  40:106, 50:133,
        60:162, 80:217, 100:273},
    2: {10:11, 15:18,  20:24,  25:31,  30:38,  40:51,  50:65,
        60:79,  80:107, 100:135},
    # FR2 (TS 38.101-2 Table 5.3.5-1)
    3: {50:66,  100:132, 200:264, 400:None},  # 400=reserved
    4: {50:32,  100:66,  200:132, 400:264},
}

# Remove None entries (unsupported combos)
for _mu in list(_N_PRB_TABLE):
    _N_PRB_TABLE[_mu] = {k:v for k,v in _N_PRB_TABLE[_mu].items() if v is not None}

_N_FFT_TABLE: dict[int, dict[int, int]] = {
    0: {5:512,  10:1024, 15:1536, 20:2048, 25:2048, 30:2048, 40:4096, 50:4096},
    1: {5:256,  10:512,  15:768,  20:1024, 25:1024, 30:1024, 40:2048, 50:2048,
        60:2048, 80:4096, 100:4096},
    2: {10:256, 15:256,  20:512,  25:512,  30:512,  40:1024, 50:1024,
        60:1024, 80:2048, 100:2048},
    3: {50:512, 100:2048, 200:4096, 400:8192},
    4: {50:256, 100:512,  200:1024, 400:2048},
}

# Normal CP length in samples = round(SCS_hz * n_fft * cp_us * 1e-6)
# Standard: cp_us = 4.6875 / (2^μ) — computed in __post_init__


# ─────────────────────────────────────────────────────────────────────────────
# SimConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """
    Complete 5G NR simulation configuration.

    Quick-start presets
    -------------------
      SimConfig()                          # defaults: μ=1, 20 MHz, MCS 16, CDL-A
      SimConfig.from_preset('nr_fr1_20mhz')
      SimConfig.from_preset('nr_fr1_100mhz_256qam')

    Manual configuration
    --------------------
      SimConfig(numerology=1, bw_mhz=20, mcs=16, mcs_table='qam64')

    Leave n_prb / n_fft / cp_len as None to auto-derive from bw_mhz + numerology.
    Supply them explicitly to override (e.g. for custom allocations).
    """

    # ── Numerology & Bandwidth ────────────────────────────────────────────────
    numerology: int   = 1        # μ: 0=15kHz, 1=30kHz, 2=60kHz, 3=120kHz, 4=240kHz
    bw_mhz:     int   = 20       # channel bandwidth in MHz

    # ── Resource allocation (None = auto-derive from bw_mhz + numerology) ────
    n_prb:      int   | None = None   # PRBs (None → from _N_PRB_TABLE)
    n_fft:      int   | None = None   # FFT size (None → from _N_FFT_TABLE)
    cp_len:     int   | None = None   # CP samples (None → computed from SCS)

    # ── Frame structure ───────────────────────────────────────────────────────
    n_symb_per_slot: int   = 14       # 14 (normal CP) or 12 (extended CP, μ=2 only)
    n_dmrs_symbols:  int   = 2        # DMRS symbols per slot (Type A: 2, Type B: 4)
    use_extended_cp: bool  = False    # extended CP (μ=2 only)

    # ── Modulation & Coding ───────────────────────────────────────────────────
    mcs:       int   = 16             # MCS index
    mcs_table: Literal['qam64', 'qam256', 'qam64lr'] = 'qam64'
    n_layers:  int   = 1              # spatial layers / rank indicator

    # ── Channel ───────────────────────────────────────────────────────────────
    channel_model: Literal[
        'AWGN', 'Rayleigh',
        'CDL-A', 'CDL-B', 'CDL-C',
        'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E',
    ] = 'CDL-A'
    snr_db:        float = 15.0       # operating SNR (dB)
    velocity_kmh:  float = 30.0       # UE velocity for Doppler (km/h)
    carrier_freq_ghz: float = 3.5     # carrier frequency in GHz (FR1: 0.5–7.125)
    delay_spread_ns:  float = 100.0   # RMS delay spread override for TDL (ns)

    # ── MIMO ──────────────────────────────────────────────────────────────────
    n_tx:     int  = 4
    n_rx:     int  = 2
    detector: Literal['ZF', 'MMSE', 'SIC'] = 'MMSE'
    antenna_correlation: float = 0.0  # ρ ∈ [0,1]

    # ── HARQ ──────────────────────────────────────────────────────────────────
    harq_processes:  int = 8          # parallel HARQ processes (4 or 8)
    max_harq_tx:     int = 4          # max transmissions per TB (1=no HARQ, 4=standard)
    harq_combining:  Literal['CC', 'IR'] = 'IR'  # Chase Combining or IR

    # ── AMC ───────────────────────────────────────────────────────────────────
    target_bler:   float = 0.1        # OLLA target BLER
    olla_enabled:  bool  = True       # outer-loop link adaptation

    # ── Simulation ────────────────────────────────────────────────────────────
    seed:       int   = 42
    n_slots:    int   = 500

    # ── Derived (auto-computed in __post_init__) ──────────────────────────────
    scs_khz:          int   = field(init=False)
    scs_hz:           float = field(init=False)
    n_prb_eff:        int   = field(init=False)   # effective n_prb (resolved)
    n_fft_eff:        int   = field(init=False)   # effective n_fft (resolved)
    cp_len_eff:       int   = field(init=False)   # effective CP length (resolved)
    n_subcarriers:    int   = field(init=False)   # n_prb_eff * 12
    slot_duration_us: float = field(init=False)
    cp_duration_us:   float = field(init=False)
    sample_rate_mhz:  float = field(init=False)
    n_data_symbols:   int   = field(init=False)   # n_symb_per_slot - n_dmrs_symbols

    def __post_init__(self):
        # Subcarrier spacing
        self.scs_khz = 15 * (2 ** self.numerology)
        self.scs_hz  = self.scs_khz * 1e3

        # Resolve n_prb
        if self.n_prb is not None:
            self.n_prb_eff = self.n_prb
        else:
            tbl = _N_PRB_TABLE.get(self.numerology, {})
            if self.bw_mhz not in tbl:
                # Find nearest supported BW
                supported = sorted(tbl.keys())
                nearest   = min(supported, key=lambda b: abs(b - self.bw_mhz))
                self.n_prb_eff = tbl[nearest]
            else:
                self.n_prb_eff = tbl[self.bw_mhz]

        # Resolve n_fft
        if self.n_fft is not None:
            self.n_fft_eff = self.n_fft
        else:
            tbl = _N_FFT_TABLE.get(self.numerology, {})
            if self.bw_mhz in tbl:
                self.n_fft_eff = tbl[self.bw_mhz]
            else:
                # Minimum power-of-2 that fits n_prb_eff * 12 subcarriers + 10% guard
                n_sc  = self.n_prb_eff * 12
                n_min = int(n_sc * 1.1)
                p     = 1
                while p < n_min:
                    p <<= 1
                self.n_fft_eff = p

        # Resolve CP length
        if self.cp_len is not None:
            self.cp_len_eff = self.cp_len
        else:
            # Normal CP: 4.6875 μs / 2^μ (TS 38.211 Table 4.3.2-1)
            if self.use_extended_cp and self.numerology == 2:
                cp_us = 4.1667
            else:
                cp_us = 4.6875 / (2 ** self.numerology)
            self.cp_len_eff = int(round(cp_us * 1e-6 * self.scs_hz * self.n_fft_eff))

        # Derived quantities
        self.n_subcarriers    = self.n_prb_eff * 12
        self.sample_rate_mhz  = self.scs_hz * self.n_fft_eff / 1e6
        symbol_duration_us    = 1e6 / self.scs_hz
        self.slot_duration_us = self.n_symb_per_slot * symbol_duration_us
        self.cp_duration_us   = self.cp_len_eff / (self.scs_hz * self.n_fft_eff) * 1e6
        self.n_data_symbols   = self.n_symb_per_slot - self.n_dmrs_symbols

        # Validate
        assert self.n_subcarriers <= self.n_fft_eff, \
            f"n_subcarriers={self.n_subcarriers} > n_fft={self.n_fft_eff}"
        assert 0 <= self.mcs <= 28, f"MCS {self.mcs} out of range 0-28"
        assert self.numerology in range(5), f"Numerology μ={self.numerology} must be 0-4"
        assert self.n_layers >= 1, "n_layers must be >= 1"

    # ── Preset factory ────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str, **overrides) -> 'SimConfig':
        """
        Create a SimConfig from a named preset.

        Available presets
        -----------------
        'nr_fr1_5mhz'          μ=0, 5 MHz,   MCS 16
        'nr_fr1_10mhz'         μ=1, 10 MHz,  MCS 16
        'nr_fr1_20mhz'         μ=1, 20 MHz,  MCS 16  ← default
        'nr_fr1_100mhz'        μ=1, 100 MHz, MCS 16
        'nr_fr1_100mhz_256qam' μ=1, 100 MHz, MCS 24, 256-QAM table
        'nr_fr2_100mhz'        μ=3, 100 MHz, MCS 16
        'embb_high_tput'       μ=1, 100 MHz, MCS 27, 256-QAM, 4×4 MIMO
        'urllc'                μ=3, 40 MHz,  MCS 4,  AWGN, 1 HARQ tx
        'iiot_low_snr'         μ=0, 10 MHz,  MCS 2,  CDL-A, high velocity
        """
        presets: dict[str, dict] = {
            'nr_fr1_5mhz': dict(
                numerology=0, bw_mhz=5, mcs=16, channel_model='AWGN'),
            'nr_fr1_10mhz': dict(
                numerology=1, bw_mhz=10, mcs=16, channel_model='CDL-A'),
            'nr_fr1_20mhz': dict(
                numerology=1, bw_mhz=20, mcs=16, channel_model='CDL-A'),
            'nr_fr1_100mhz': dict(
                numerology=1, bw_mhz=100, mcs=16, channel_model='CDL-A'),
            'nr_fr1_100mhz_256qam': dict(
                numerology=1, bw_mhz=100, mcs=24, mcs_table='qam256',
                channel_model='CDL-A', n_tx=4, n_rx=4),
            'nr_fr2_100mhz': dict(
                numerology=3, bw_mhz=100, mcs=16, channel_model='CDL-A',
                velocity_kmh=3.0, carrier_freq_ghz=28.0),
            'embb_high_tput': dict(
                numerology=1, bw_mhz=100, mcs=27, mcs_table='qam256',
                channel_model='AWGN', n_tx=4, n_rx=4, n_layers=2,
                harq_processes=8, max_harq_tx=4),
            'urllc': dict(
                numerology=3, bw_mhz=40, mcs=4, channel_model='AWGN',
                max_harq_tx=1, target_bler=0.00001, n_slots=200),
            'iiot_low_snr': dict(
                numerology=0, bw_mhz=10, mcs=2, channel_model='CDL-A',
                snr_db=0.0, velocity_kmh=120.0),
        }
        if name not in presets:
            available = ', '.join(f"'{k}'" for k in presets)
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        params = {**presets[name], **overrides}
        return cls(**params)

    # ── Validation helpers ────────────────────────────────────────────────────

    def validate_cp_vs_channel(self) -> list[str]:
        """
        Check CP duration covers expected delay spread for the selected channel.
        Returns list of warning strings (empty = all OK).
        """
        # Max delay spreads in μs per channel model
        max_delay = {
            'AWGN': 0.0, 'Rayleigh': 0.0,
            'CDL-A': 0.75, 'CDL-B': 0.44, 'CDL-C': 0.75,
            'TDL-A': self.delay_spread_ns / 1000,
            'TDL-B': self.delay_spread_ns / 1000,
            'TDL-C': self.delay_spread_ns / 1000,
            'TDL-D': self.delay_spread_ns / 1000,
            'TDL-E': self.delay_spread_ns / 1000,
        }
        warnings = []
        ch = self.channel_model
        if ch in max_delay and max_delay[ch] > 0:
            if self.cp_duration_us < max_delay[ch]:
                warnings.append(
                    f"CP ({self.cp_duration_us:.3f} μs) < {ch} max delay "
                    f"({max_delay[ch]:.3f} μs) — ISI possible"
                )
        return warnings

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        w = self.validate_cp_vs_channel()
        lines = [
            "=== SimConfig ===",
            f"  Numerology      : μ={self.numerology}  ({self.scs_khz} kHz SCS)",
            f"  Bandwidth       : {self.bw_mhz} MHz",
            f"  PRBs / SCs      : {self.n_prb_eff} PRBs  →  {self.n_subcarriers} subcarriers",
            f"  FFT size        : {self.n_fft_eff}",
            f"  CP length       : {self.cp_len_eff} samples  ({self.cp_duration_us:.4f} μs)",
            f"  Sample rate     : {self.sample_rate_mhz:.3f} MHz",
            f"  Symbols/slot    : {self.n_symb_per_slot}  "
            f"({self.n_data_symbols} data + {self.n_dmrs_symbols} DMRS)",
            f"  Slot duration   : {self.slot_duration_us:.2f} μs",
            f"  MCS index       : {self.mcs}  (table: {self.mcs_table})",
            f"  Spatial layers  : {self.n_layers}",
            f"  Channel model   : {self.channel_model}",
            f"  SNR             : {self.snr_db} dB",
            f"  UE velocity     : {self.velocity_kmh} km/h",
            f"  Carrier freq    : {self.carrier_freq_ghz} GHz",
            f"  Antennas (Tx/Rx): {self.n_tx} / {self.n_rx}",
            f"  Detector        : {self.detector}",
            f"  HARQ            : {self.harq_processes} procs, "
            f"max {self.max_harq_tx} tx, {self.harq_combining}",
            f"  OLLA            : {'enabled' if self.olla_enabled else 'disabled'}"
            f"  target BLER={self.target_bler}",
        ]
        if w:
            lines += [f"  ⚠ WARNING: {msg}" for msg in w]
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print("── Default config ──")
    print(SimConfig().summary())

    print("\n── Presets ──")
    for name in ['nr_fr1_20mhz', 'nr_fr1_100mhz_256qam', 'embb_high_tput', 'urllc']:
        cfg = SimConfig.from_preset(name)
        print(f"\n[{name}]")
        print(cfg.summary())

    print("\n── Auto n_prb from bw_mhz ──")
    for mu, bw in [(0,20),(1,100),(2,100)]:
        cfg = SimConfig(numerology=mu, bw_mhz=bw)
        print(f"  μ={mu} {bw}MHz → {cfg.n_prb_eff} PRBs, n_fft={cfg.n_fft_eff}, "
              f"CP={cfg.cp_duration_us:.3f}μs, fs={cfg.sample_rate_mhz:.1f}MHz")