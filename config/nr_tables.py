"""
config/nr_tables.py — 3GPP NR Lookup Tables
=============================================
  • MCS tables  — TS 38.214 Tables 5.1.3.1-1 / 5.1.3.1-2 / 5.1.3.1-3
  • TBS         — TS 38.214 Section 5.1.3.2 (full two-branch procedure)
  • CQI table   — TS 38.214 Table 5.2.2.1-3 (up to 256-QAM)

MCS table selection
-------------------
  'qam64'   (default) — Table 5.1.3.1-2: MCS 0-28, up to 64-QAM
  'qam256'            — Table 5.1.3.1-3: MCS 0-27, up to 256-QAM
  'qam64lr'           — Table 5.1.3.1-1: MCS 0-28, low-code-rate QPSK/16QAM
"""

from __future__ import annotations
import math
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# MCS Table 5.1.3.1-2 — 64-QAM (default DL, TS 38.214)
# (Qm, R×1024, spectral_efficiency)
# ─────────────────────────────────────────────────────────────────────────────
MCS_TABLE: dict[int, tuple[int, int, float]] = {
     0: (2,   120, 0.2344),
     1: (2,   157, 0.3066),
     2: (2,   193, 0.3770),
     3: (2,   251, 0.4902),
     4: (2,   308, 0.6016),
     5: (2,   379, 0.7402),
     6: (2,   449, 0.8770),
     7: (2,   526, 1.0273),
     8: (2,   602, 1.1758),
     9: (2,   679, 1.3262),
    10: (4,   340, 1.3281),
    11: (4,   378, 1.4766),
    12: (4,   434, 1.6953),
    13: (4,   490, 1.9141),
    14: (4,   553, 2.1602),
    15: (4,   616, 2.4063),
    16: (4,   658, 2.5703),
    17: (6,   438, 2.5664),
    18: (6,   466, 2.7305),
    19: (6,   517, 3.0293),
    20: (6,   567, 3.3223),
    21: (6,   616, 3.6094),
    22: (6,   666, 3.9023),
    23: (6,   719, 4.2129),
    24: (6,   772, 4.5234),
    25: (6,   822, 4.8164),
    26: (6,   873, 5.1152),
    27: (6,   910, 5.3320),
    28: (6,   948, 5.5547),
}

# ─────────────────────────────────────────────────────────────────────────────
# MCS Table 5.1.3.1-3 — 256-QAM (TS 38.214)
# MCS 0-27, Qm up to 8
# ─────────────────────────────────────────────────────────────────────────────
MCS_TABLE_256QAM: dict[int, tuple[int, int, float]] = {
     0: (2,   120, 0.2344),
     1: (2,   193, 0.3770),
     2: (2,   308, 0.6016),
     3: (2,   449, 0.8770),
     4: (2,   602, 1.1758),
     5: (4,   378, 1.4766),
     6: (4,   434, 1.6953),
     7: (4,   490, 1.9141),
     8: (4,   553, 2.1602),
     9: (4,   616, 2.4063),
    10: (4,   658, 2.5703),
    11: (6,   466, 2.7305),
    12: (6,   517, 3.0293),
    13: (6,   567, 3.3223),
    14: (6,   616, 3.6094),
    15: (6,   666, 3.9023),
    16: (6,   719, 4.2129),
    17: (6,   772, 4.5234),
    18: (6,   822, 4.8164),
    19: (6,   873, 5.1152),
    20: (6,   910, 5.3320),
    21: (6,   948, 5.5547),
    22: (8,   567, 5.5547),
    23: (8,   616, 6.0000),
    24: (8,   666, 6.5000),
    25: (8,   719, 7.0000),
    26: (8,   772, 7.5000),
    27: (8,   822, 8.0000),
}

# ─────────────────────────────────────────────────────────────────────────────
# MCS Table 5.1.3.1-1 — Low spectral efficiency / QPSK only (TS 38.214)
# Used for coverage-limited scenarios (URLLC, IoT edge)
# ─────────────────────────────────────────────────────────────────────────────
MCS_TABLE_LR: dict[int, tuple[int, int, float]] = {
     0: (2,    30, 0.0586),
     1: (2,    40, 0.0781),
     2: (2,    50, 0.0977),
     3: (2,    64, 0.1250),
     4: (2,    78, 0.1523),
     5: (2,    99, 0.1934),
     6: (2,   120, 0.2344),
     7: (2,   157, 0.3066),
     8: (2,   193, 0.3770),
     9: (2,   251, 0.4902),
    10: (2,   308, 0.6016),
    11: (2,   379, 0.7402),
    12: (2,   449, 0.8770),
    13: (2,   526, 1.0273),
    14: (2,   602, 1.1758),
    15: (4,   340, 1.3281),
    16: (4,   378, 1.4766),
    17: (4,   434, 1.6953),
    18: (4,   490, 1.9141),
    19: (4,   553, 2.1602),
    20: (4,   616, 2.4063),
    21: (6,   438, 2.5664),
    22: (6,   466, 2.7305),
    23: (6,   517, 3.0293),
    24: (6,   567, 3.3223),
    25: (6,   616, 3.6094),
    26: (6,   666, 3.9023),
    27: (6,   719, 4.2129),
    28: (6,   772, 4.5234),
}

_MCS_TABLES = {
    'qam64':   MCS_TABLE,
    'qam256':  MCS_TABLE_256QAM,
    'qam64lr': MCS_TABLE_LR,
}

_MCS_TABLE_NAMES = {
    'qam64':   'TS 38.214 Table 5.1.3.1-2 (64-QAM)',
    'qam256':  'TS 38.214 Table 5.1.3.1-3 (256-QAM)',
    'qam64lr': 'TS 38.214 Table 5.1.3.1-1 (low-rate)',
}


# ─────────────────────────────────────────────────────────────────────────────
# CQI Table 5.2.2.1-3 — up to 64-QAM (TS 38.214)
# CQI index 0 = out of range; 1-15 = valid
# ─────────────────────────────────────────────────────────────────────────────
CQI_TABLE: dict[int, tuple[str, int, float]] = {
    #  idx : (modulation, R×1024, efficiency)
     1: ('QPSK',    78, 0.1523),
     2: ('QPSK',   120, 0.2344),
     3: ('QPSK',   193, 0.3770),
     4: ('QPSK',   308, 0.6016),
     5: ('QPSK',   449, 0.8770),
     6: ('QPSK',   602, 1.1758),
     7: ('16QAM',  378, 1.4766),
     8: ('16QAM',  490, 1.9141),
     9: ('16QAM',  616, 2.4063),
    10: ('64QAM',  466, 2.7305),
    11: ('64QAM',  567, 3.3223),
    12: ('64QAM',  666, 3.9023),
    13: ('64QAM',  772, 4.5234),
    14: ('64QAM',  873, 5.1152),
    15: ('64QAM',  948, 5.5547),
}

# CQI Table for 256-QAM (TS 38.214 Table 5.2.2.1-4)
CQI_TABLE_256QAM: dict[int, tuple[str, int, float]] = {
     1: ('QPSK',    78, 0.1523),
     2: ('QPSK',   193, 0.3770),
     3: ('QPSK',   449, 0.8770),
     4: ('16QAM',  378, 1.4766),
     5: ('16QAM',  490, 1.9141),
     6: ('16QAM',  616, 2.4063),
     7: ('64QAM',  466, 2.7305),
     8: ('64QAM',  567, 3.3223),
     9: ('64QAM',  666, 3.9023),
    10: ('64QAM',  772, 4.5234),
    11: ('64QAM',  873, 5.1152),
    12: ('64QAM',  948, 5.5547),
    13: ('256QAM', 567, 5.5547),
    14: ('256QAM', 666, 6.5000),
    15: ('256QAM', 772, 7.5000),
}


# ─────────────────────────────────────────────────────────────────────────────
# TBS — TS 38.214 Section 5.1.3.2  (full procedure)
# ─────────────────────────────────────────────────────────────────────────────

# Quantised TBS values — TS 38.214 Table 5.1.3.2-2
_TBS_QUANT = [
     24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
    104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
    184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528,
    552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
    888, 928, 984,1032,1064,1128,1160,1192,1224,1256,
   1288,1320,1352,1416,1480,1544,1608,1672,1736,1800,
   1864,1928,2024,2088,2152,2216,2280,2408,2472,2536,
   2600,2664,2728,2792,2856,2976,3104,3240,3368,3496,
   3624,3752,3824,
]


def _quantise_tbs(n_info: float) -> int:
    """Round up to nearest value in TS 38.214 Table 5.1.3.2-2."""
    for tbs in _TBS_QUANT:
        if tbs >= n_info:
            return tbs
    return _TBS_QUANT[-1]


def get_tbs(
    mcs_idx:   int,
    n_prb:     int | None,
    n_symb:    int   = 12,
    n_layers:  int   = 1,
    mcs_table: str   = 'qam64',
    n_re_oh:   int   = 0,
) -> int:
    """
    Compute Transport Block Size per TS 38.214 §5.1.3.2.

    Parameters
    ----------
    mcs_idx   : MCS index
    n_prb     : allocated PRBs (pass cfg.n_prb_eff, not cfg.n_prb)
    n_symb    : data-bearing OFDM symbols per slot (default 12 = 14 - 2 DMRS)
    n_layers  : spatial layers
    mcs_table : 'qam64' | 'qam256' | 'qam64lr'
    n_re_oh   : additional overhead REs per PRB (PTRS, CSI-RS, etc.)

    Returns
    -------
    TBS in bits
    """
    if n_prb is None:
        raise ValueError(
            "get_tbs() received n_prb=None. "            "Use cfg.n_prb_eff (resolved value) not cfg.n_prb (user override)."
        )
    table = _MCS_TABLES.get(mcs_table, MCS_TABLE)
    if mcs_idx not in table:
        raise ValueError(
            f"MCS {mcs_idx} not in table '{mcs_table}' "
            f"(valid range 0–{max(table)})"
        )

    qm, r_x1024, _ = table[mcs_idx]
    # Step 1: N_RE — REs per PRB, capped at 156, minus overhead
    n_re_per_prb = min(156, 12 * n_symb) - n_re_oh
    n_re         = max(0, n_re_per_prb) * n_prb

    # Step 2: N_info
    n_info = n_re * (r_x1024 / 1024) * qm * n_layers

    if n_info <= 3824:
        # Step 3a: n = floor(log2(N_info - 24)) - 1
        n            = max(0, math.floor(math.log2(max(n_info - 24, 1))) - 1)
        n_info_prime = max(24, 2**n * round((n_info - 24) / 2**n))
        return _quantise_tbs(n_info_prime)
    else:
        # Step 3b: n = floor(log2(N_info - 24))  [no -1]
        n            = math.floor(math.log2(n_info - 24))
        n_info_prime = max(3840, 2**n * math.ceil((n_info - 24) / 2**n))
        r            = r_x1024 / 1024
        if r <= 0.25:
            c = math.ceil((n_info_prime + 24) / 3816)
        else:
            c = math.ceil((n_info_prime + 24) / 8424)
        return int(8 * c * math.ceil((n_info_prime + 24) / (8 * c)) - 24)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience lookups
# ─────────────────────────────────────────────────────────────────────────────

def get_mcs_params(mcs_idx: int, mcs_table: str = 'qam64') -> dict:
    """Return modulation order, code rate, and spectral efficiency."""
    table = _MCS_TABLES.get(mcs_table, MCS_TABLE)
    if mcs_idx not in table:
        raise ValueError(f"MCS {mcs_idx} not in table '{mcs_table}'")
    qm, r_x1024, se = table[mcs_idx]
    mod = {2: 'QPSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'}.get(qm, f'{2**qm}-QAM')
    return {
        'mcs_idx':             mcs_idx,
        'mcs_table':           mcs_table,
        'modulation':          mod,
        'Qm':                  qm,
        'code_rate':           round(r_x1024 / 1024, 4),
        'spectral_efficiency': se,
    }


def get_mcs_table(mcs_table: str = 'qam64') -> dict:
    """Return the full MCS table dict."""
    if mcs_table not in _MCS_TABLES:
        raise ValueError(f"Unknown MCS table '{mcs_table}'. Use: {list(_MCS_TABLES)}")
    return _MCS_TABLES[mcs_table]


def get_cqi_params(cqi_idx: int, cqi_table: str = 'qam64') -> dict:
    """Return modulation, code rate, and efficiency for a CQI index."""
    tbl = CQI_TABLE_256QAM if cqi_table == 'qam256' else CQI_TABLE
    if cqi_idx not in tbl:
        raise ValueError(f"CQI {cqi_idx} out of range (1-15)")
    mod, r_x1024, eff = tbl[cqi_idx]
    return {
        'cqi_idx':    cqi_idx,
        'modulation': mod,
        'code_rate':  round(r_x1024 / 1024, 4),
        'efficiency': eff,
    }


def mcs_from_cqi(cqi_idx: int, mcs_table: str = 'qam64') -> int:
    """Map CQI to best-matching MCS by spectral efficiency."""
    cqi_tbl = CQI_TABLE_256QAM if mcs_table == 'qam256' else CQI_TABLE
    if cqi_idx not in cqi_tbl:
        raise ValueError(f"CQI {cqi_idx} out of range (1-15)")
    target_se = cqi_tbl[cqi_idx][2]
    table     = _MCS_TABLES[mcs_table]
    return min(table, key=lambda m: abs(table[m][2] - target_se))


def max_mcs(mcs_table: str = 'qam64') -> int:
    """Return maximum MCS index for a given table."""
    return max(_MCS_TABLES[mcs_table].keys())


def tput_peak_mbps(
    mcs_idx:      int,
    n_prb:        int,
    slot_duration_us: float,
    n_symb:       int  = 12,
    n_layers:     int  = 1,
    mcs_table:    str  = 'qam64',
) -> float:
    """Peak throughput in Mbps = TBS / slot_duration."""
    tbs = get_tbs(mcs_idx, n_prb, n_symb, n_layers, mcs_table)
    return tbs / slot_duration_us


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Print all three tables
    for tname in ['qam64', 'qam256', 'qam64lr']:
        tbl = _MCS_TABLES[tname]
        print(f"\n── {_MCS_TABLE_NAMES[tname]} ──")
        print(f"{'MCS':>4} {'Mod':^8} {'Qm':>3} {'Rate':>7} "
              f"{'TBS@52PRB':>10} {'Peak(Mbps)':>11}")
        print("─" * 52)
        prev_tbs = 0
        for m in sorted(tbl):
            p    = get_mcs_params(m, tname)
            tbs  = get_tbs(m, 52, mcs_table=tname)
            peak = tbs / 466.67
            mono = '' if tbs >= prev_tbs else ' ← DROP'
            print(f"{m:4d} {p['modulation']:^8} {p['Qm']:3d} "
                  f"{p['code_rate']:7.4f} {tbs:10d} {peak:11.2f}{mono}")
            prev_tbs = tbs

    # CQI → MCS mapping
    print("\n── CQI → MCS (qam64) ──")
    for cqi in range(1, 16):
        mcs = mcs_from_cqi(cqi)
        p   = get_mcs_params(mcs)
        print(f"  CQI {cqi:2d} → MCS {mcs:2d}  {p['modulation']:6s}  "
              f"R={p['code_rate']:.3f}")