#Below code is a small helper module for 5G NR link adaptation. It does three main things:
#1. Calculates TBS, or Transport Block Size, from MCS and resource allocation.
#2. Looks up MCS parameters.
#3. Looks up CQI parameters and maps CQI to a likely MCS.
"""
nr_tables.py — 3GPP NR lookup tables
  • MCS table (TS 38.214 Table 5.1.3.1-2, 64-QAM)
  • TBS lookup (TS 38.214 Section 5.1.3.2, simplified)
  • CQI table (TS 38.214 Table 5.2.2.1-3, 64-QAM)
"""

from __future__ import annotations
import math
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# MCS Table (3GPP TS 38.214 Table 5.1.3.1-2)
# Columns: modulation order (Qm), target code rate × 1024 (R×1024), spectral efficiency
# MCS index 0-28
# ─────────────────────────────────────────────────────────────────────────────
MCS_TABLE: dict[int, Tuple[int, int, float]] = {
    #  idx : (Qm,  R×1024,  SE)
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
# CQI Table (3GPP TS 38.214 Table 5.2.2.1-3, up to 64-QAM)
# Columns: modulation, code rate × 1024, efficiency
# CQI index 1-15  (0 = out of range)
# ─────────────────────────────────────────────────────────────────────────────
CQI_TABLE: dict[int, Tuple[str, int, float]] = {
    #  idx : (modulation,  R×1024,  efficiency)
     1: ('QPSK',   78,  0.1523),
     2: ('QPSK',  120,  0.2344),
     3: ('QPSK',  193,  0.3770),
     4: ('QPSK',  308,  0.6016),
     5: ('QPSK',  449,  0.8770),
     6: ('QPSK',  602,  1.1758),
     7: ('16QAM', 378,  1.4766),
     8: ('16QAM', 490,  1.9141),
     9: ('16QAM', 616,  2.4063),
    10: ('64QAM', 466,  2.7305),
    11: ('64QAM', 567,  3.3223),
    12: ('64QAM', 666,  3.9023),
    13: ('64QAM', 772,  4.5234),
    14: ('64QAM', 873,  5.1152),
    15: ('64QAM', 948,  5.5547),
}

# ─────────────────────────────────────────────────────────────────────────────
# TBS (Transport Block Size) — TS 38.214 Section 5.1.3.2 (simplified procedure)
# ─────────────────────────────────────────────────────────────────────────────

# Quantised TBS values from 3GPP TS 38.214 Table 5.1.3.2-2
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

#Below function rounds a calculated information-bit value up to the nearest allowed TBS value greater than calculated one.
#_TBS_QUANT is assumed to be a predefined list of allowed small TBS values.

def _quantise_tbs(n_info: float) -> int:
    """Round n_info up to the nearest quantised TBS value."""
    for tbs in _TBS_QUANT:
        if tbs >= n_info:
            return tbs
    return _TBS_QUANT[-1]

#TBS: Number of Tranport Block Size. 
# In 5G NR, TBS is the number of payload bits that can be carried in one transport block for a given allocation.
# Factors effecting TBS Computations are:
## 1. The MCS index. MCS means Modulation and Coding Scheme. It determines:
#         a. modulation order, such as QPSK, 16QAM, 64QAM
#         b. code rate
#         c. spectral efficiency
## 2. Number of allocated PRBs.A PRB is a Physical Resource Block. In NR, one PRB has 12 subcarriers in frequency.
## 3. Number of usable OFDM symbols for data. A slot may have 14 OFDM symbols, 
#       but some OFDM symbols are used by DMRS/control/overhead. This code assumes 12 data symbols by default.
####  Real NR scheduling is dynamic i.e., usable data symbols will change.The actual number of usable data symbols depends on:
#           a. DMRS configuration	        - consumes symbols
#           b. Mapping type A/B	            - affects symbol locations
#           c. Additional DMRS positions	- more overhead
#           d. PDCCH duration	            - fewer PDSCH symbols
#           e. TDD pattern	                - guard symbols
#           f. Mini-slot scheduling	        - fewer symbols
#           g. CSI-RS	                    - punctures resources
#           h. CORESET size	                - affects data regionv 
# In Typical scenario: DMRS overhead - ~1-2 and Control overhead - ~0-1. Remaining data symbols: ~12.
# In real work this can be ~7symbols for mini slot or ~11 when PDCCH(0th), PDSCH(Any Symbol between 1-11), DMRS(12th Symbol)
# guard(13th Symbol) sallocations are also done.
# Actual scheduled resources after overhead accounting.The scheduler determines:
#       a. which symbols are allocated
#       b. where DMRS exists
#       c. puncturing: resources that would normally carry data are intentionally removed or overwritten.
#       d. reserved REs
## 4. Number of spatial MIMO layers. More layers mean more parallel data streams.

def get_tbs(mcs_idx: int, n_prb: int, n_symb: int = 12, n_layers: int = 1) -> int:
    """
    Compute Transport Block Size (bits) per 3GPP TS 38.214 §5.1.3.2.

    Parameters
    ----------
    mcs_idx  : MCS index (0-28)
    n_prb    : number of allocated PRBs
    n_symb   : number of OFDM data symbols per slot (default 12, allowing for DMRS)
    n_layers : number of spatial layers

    Returns
    -------
    TBS in bits
    """
    # Step 1: Validate MCS index. The code checks whether the requested MCS index exists in MCS_TABLE. 
    # If not, it raises an error.
    if mcs_idx not in MCS_TABLE:
        raise ValueError(f"MCS index {mcs_idx} out of range (0-28)")
    
    # Step 2: Read MCS parameters. 
    #       Each MCS entry contains something like:(Qm, R_x1024, spectral_efficiency)
    #           1. qm: Modulation order. This is the number of bits per modulation symbol.       
    #               | Qm | Modulation |
    #               |  2 | QPSK       |
    #               |  4 | 16QAM      |
    #               |  6 | 64QAM      |
    #               |  8 | 256QAM     |
    #           2. r_x1024: Code rate scaled by 1024. Example : if r_x1024 = 490 then actual code "490 / 1024 = 0.4785"
    #           3. The underscore _ means the spectral efficiency value is ignored in this function.
    qm, r_x1024, _ = MCS_TABLE[mcs_idx]
    # Step 3: Calculate number of REs
    # Lets understand PRB using a real world analogy:
    # “12 lanes on a highway” If lane spacing increases: the highway becomes wider even though: 
    # number of lanes stays 12 Similarly: PRB always has 12 subcarriers but spacing between them changes so 
    # total PRB bandwidth changes.
    # For example if Each PRB has 12 subcarriers. So if n_symb = 12. n_re_per_prb = 12 * 12 = 144. 
    # That means each PRB has 144 usable resource elements before the cap.
    
    n_re_per_prb = 12 * n_symb                   # REs per PRB (12 SCs × symbols)
    
    # As per 3gpp 38.214 for TBS calculation methodology 156 is a standardized upper bound approximation used in 
    # TBS determination, not the exact real RE count in every slot. “Per PRB, never assume more than 156 usable REs 
    # for TBS calculation.” even when total number of REs exits per slot "12*14=168" where Normal CP slot: 
    # 14 OFDM symbols , 12 subcarriers. This is the absolute physical maximum. But some REs are never usable for data
    # DMRS, PTRS, control overlap, reserved REs, CSI-RS, synchronization.Capped RE count avoids:overestimating throughput,
    # unrealistic TBS,scheduler mismatch, excessive signaling complexity. 
    # But exact RE accounting becomes extremely complicated because of: varying DMRS patterns, antenna ports, mini-slots,
    # puncturing, CSI-RS, CORESET overlap, DSS coexistence, So NR standardized a simplified method.
    # n_prb = 52, n_symb = 12. n_re_per_prb = 144, n_re = min(156, 144) * 52, n_re = 144 * 52, n_re = 7488. 
    # So the allocation contains 7488 usable REs.
    
    n_re = min(156, n_re_per_prb) * n_prb        # cap at 156 RE/PRB (overhead)

    # Step 4: Compute raw information bits. This estimates the number of information bits before final TBS rounding.
    # n_info = number_of_REs × code_rate × modulation_order × layers
    # This estimates the number of information bits before final TBS rounding.
    # n_re = 7488, Qm = 4, code_rate = 490 / 1024, n_layers = 1 then n_info = 7488 * 0.4785 * 4 * 1 ≈ 14333 bits.
    # This is not necessarily the final TBS. It still needs 3GPP-style rounding.
    # This computes raw information bits (before 3GPP rounding/quantization).
    # Ninfo​=NRE​×Qm​×R×v,  This equation is basically, How many bits can I pack into all the REs I have?
    # | Term     | Meaning                            |
    # | -------- | ---------------------------------- |
    # | (N_{RE}) | number of usable resource elements |
    # | (Q_m)    | bits per symbol (modulation order) |
    # | (R)      | code rate                          |
    # | (v)      | number of layers (MIMO streams)    |
    # Step 1: Each RE carries one modulation symbol From OFDM: 1 RE = 1 complex symbol (like QPSK/QAM point)
    # So:NRE⇒ Nsymbols
    # Step 2: Each symbol carries Qm bits. Depending 
    #| Modulation | (Q_m) | Bits per symbol |
    # | ---------- | ----- | --------------- |
    # | QPSK       | 2     | 2 bits          |
    # | 16QAM      | 4     | 4 bits          |
    # | 64QAM      | 6     | 6 bits          |
    # | 256QAM     | 8     | 8 bits          |
    # So, NRE​×Qm​=coded bits, These are coded bits, not actual user data yet.
    # Step 3: Apply code rate R: Coding adds redundancy for error correction.
    # R=0.5
    # then: 50% useful bits, 50% redundancy So: information bits=coded bits×R
    # Step 4: Multiply by number of layers
    #   With MIMO:
    # * each layer carries an independent data stream
    # So: ×v, Example: 2 layers → double throughput (ideally)
    # Putting it together
    # N_info = REs * bits per symbol * code rate * layers
    # Concrete numerical example
    # Let’s walk through a realistic case.
    # Given:
    #   * PRBs = 50
    #   * symbols = 12
    #   * Qm =4 (16QAM)
    #   * R=0.5
    #   * layers = 1
    #   Step 1: Compute REs
    #   NRE =50×12×12=7200
    #   Step 2: Apply modulation
    #   7200×4=28800 coded bits
    #   Step 3: Apply coding rate
    #   28800×0.5=14400 information bits
    #   Final result
    #   Ninfo =14400
# This is what your code computes before TBS rounding.
# Why this is called “raw” information bits
# Because it is not yet the final TBS.
# Still missing:
#   * CRC bits
#   * segmentation constraints
#   * alignment to allowed sizes
#   * quantization (small TBS table)
# So:
#   n_info → intermediate value
#   TBS → final standardized value
#   Deep physical meaning
#   This equation combines three independent limits:
#   1. Time-frequency resources → NRE => How many “slots” you have to send symbols.
#   2. Modulation →
#   Qm: How many bits each symbol can carry. This depends on:
#   * SNR
#   * channel quality
#   * CQI
#   3. Coding → How aggressively you pack data vs redundancy.
#   Higher R:
#   * more throughput
#   * less reliability
#   4. Spatial dimension → v: Parallel streams via MIMO.
#   Think of it like packing boxes:
#   Factor	Analogy
#   NRE	number of boxes
#   Qm	size of each box
#   R	usable fraction of box
#   v	number of parallel trucks
#   Important subtlety: why multiply in this order?
#   Because the chain is:
#   RE → symbol → coded bits → information bits

#   So mathematically:
# NRE →Nsymbols →Ncoded bits →Ninfo bits

# Real-world complications
# In real systems, this step is still slightly idealized.

# 1. Not all REs are equal
# Some REs:
# * have lower SINR
# * suffer interference
# * may be punctured
# Schedulers assume average behavior.

# 2. Code rate is “target”, not exact
# The actual LDPC coding process:
# * works in blocks
# * has discrete sizes
# So real effective rate may differ slightly.

# 3. MIMO layers are not always independent
# In practice:
# * layers interfere
# * precoding matters
# * channel rank limits performance
# So throughput may be less than v× ideal.

# 4. Link adaptation ties everything together
# The scheduler chooses MCS such that:
# BLER target e.g. 10% is met
# So, Qm and R are not arbitrary.

# Why this step is critical
# This equation is the bridge between PHY resources and actual data rate.
# Everything upstream defines:
# * how many symbols you have
# Everything downstream adjusts:
# * how many bits you’re allowed to report

# Connection to spectral efficiency
# Spectral efficiency:
# Ninfo =NRE ×η×v
# This is often how engineers think about it.

# Final takeaway
# This line: n_info = n_re * (r_x1024 / 1024) * qm * n_layers
# means:
# “Take all usable resource elements, multiply by how many bits each symbol carries, adjust for coding redundancy, and scale by number of parallel streams to estimate how many real payload bits can be transmitted.”
# It is the core throughput equation of the physical layer.

    n_info = n_re * (r_x1024 / 1024) * qm * n_layers
    # For small transport blocks, the standard uses a table/quantisation process.
    if n_info <= 3824:
        # This rounds n_info to a suitable intermediate value called n_info_prime. 
        # The max(24, ...) ensures the result is never below 24 bits. 
        # The result is rounded up to one of the allowed TBS table values.
        # Important note: this code assumes n_info > 24. If n_info <= 24, then: math.log2(n_info - 24)
        # would fail because the logarithm of zero or a negative number is invalid. 
        # A safer implementation would handle that case explicitly.
        n_info_prime = max(24, 2 ** math.floor(math.log2(n_info - 24)) * round((n_info - 24) / 2 ** math.floor(math.log2(n_info - 24))))
        return _quantise_tbs(n_info_prime)
    else:
        # For larger transport blocks, the code uses a formula instead of the small TBS quantisation table.
        # This rounds n_info upward to a power-of-two-based boundary. The result is at least 3840. 
        n_info_prime = max(3840, 2 ** math.floor(math.log2(n_info - 24)) * math.ceil((n_info - 24) / 2 ** math.floor(math.log2(n_info - 24))))
        # Then the code checks the code rate: 
        # Here c is the number of code blocks after segmentation. Low-rate blocks use a smaller threshold, 3816. 
        # Higher-rate blocks use 8424.
        # This rounds the final TBS to a multiple of 8 * c, then subtracts 24 CRC bits.
        if (r_x1024 / 1024) <= 0.25:
            # For low code rates, it uses:
            c = math.ceil((n_info_prime + 24) / 3816)
            tbs = 8 * c * math.ceil((n_info_prime + 24) / (8 * c)) - 24
        else:
            # For higher code rates, it uses:
            c = math.ceil((n_info_prime + 24) / 8424)
            tbs = 8 * c * math.ceil((n_info_prime + 24) / (8 * c)) - 24
        return int(tbs)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience lookup functions
# ─────────────────────────────────────────────────────────────────────────────
# This function returns readable information for an MCS index.
# Again, it validates the MCS index.
# It extracts:
#   qm: modulation order
#   r_x1024: code rate scaled by 1024
#   se: spectral efficiency
# This converts Qm into a modulation name. But there is a small issue here:
# Examples: 
#   Qm = 2  → QPSK
#   Qm = 4  → 16QAM
#   Qm = 6  → 64QAM
#   But there is a small issue here: 
#   For Qm = 8, this becomes: 256-QAM
#   But for Qm = 10, it would become: 1024-QAM 
#   also mathematically consistent. The function returns a dictionary:
def get_mcs_params(mcs_idx: int) -> dict:
    """Return modulation order, code rate, and spectral efficiency for an MCS index."""
    if mcs_idx not in MCS_TABLE:
        raise ValueError(f"MCS index {mcs_idx} out of range (0-28)")
    qm, r_x1024, se = MCS_TABLE[mcs_idx]
    mod_order = {2: 'QPSK', 4: '16QAM', 6: '64QAM'}.get(qm, f'{2**qm}-QAM')
    return {
        'mcs_idx':    mcs_idx,
        'modulation': mod_order,
        'Qm':         qm,
        'code_rate':  round(r_x1024 / 1024, 4),
        'spectral_efficiency': se,
    }

# This function returns readable information for a CQI index.
# CQI means Channel Quality Indicator. It is reported by the UE to indicate radio channel quality.
# Valid CQI values are usually 1 to 15. 
# It extracts:
#   modulation name
#   code rate scaled by 1024
#   efficiency
# For Example: get_cqi_params(9)
#   {
#   'cqi_idx': 9,
#   'modulation': '16QAM',
#   'code_rate': 0.6016,
#   'efficiency': 2.4063}
def get_cqi_params(cqi_idx: int) -> dict:
    """Return modulation, code rate, and efficiency for a CQI index (1-15)."""
    if cqi_idx not in CQI_TABLE:
        raise ValueError(f"CQI index {cqi_idx} out of range (1-15)")
    mod, r_x1024, eff = CQI_TABLE[cqi_idx]
    return {
        'cqi_idx':    cqi_idx,
        'modulation': mod,
        'code_rate':  round(r_x1024 / 1024, 4),
        'efficiency': eff,
    }

# This function maps a CQI index to a suggested MCS index.
# It gets the spectral efficiency associated with the CQI.
# This searches all MCS indices and finds the one whose spectral efficiency is closest to the CQI efficiency.
# For example, if CQI 9 has efficiency 2.4063, the function finds the MCS whose spectral efficiency is closest to 2.4063.
# This is a heuristic, not necessarily a strict standards-defined mapping.
def mcs_from_cqi(cqi_idx: int) -> int:
    """
    Map CQI index to a suggested MCS index.
    Simple heuristic: match closest spectral efficiency.
    """
    if cqi_idx not in CQI_TABLE:
        raise ValueError(f"CQI index {cqi_idx} out of range (1-15)")
    target_se = CQI_TABLE[cqi_idx][2]
    best_mcs = min(MCS_TABLE, key=lambda m: abs(MCS_TABLE[m][2] - target_se))
    return best_mcs


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── MCS Table (selected) ──")
    for idx in [0, 9, 16, 28]:
        p = get_mcs_params(idx)
        tbs = get_tbs(idx, n_prb=52)
        print(f"  MCS {idx:2d}: {p['modulation']:6s}  R={p['code_rate']:.3f}  SE={p['spectral_efficiency']:.4f}  TBS={tbs} bits")

    print("\n── CQI → MCS mapping ──")
    for cqi in [1, 5, 9, 12, 15]:
        mcs = mcs_from_cqi(cqi)
        print(f"  CQI {cqi:2d} → MCS {mcs:2d}  ({get_cqi_params(cqi)['modulation']})")
