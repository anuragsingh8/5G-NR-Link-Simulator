"""
coding/ldpc.py — LDPC Encoder, Belief-Propagation Decoder, Rate Matching
3GPP TS 38.212 Section 5.3 / 5.4 (Base Graph 1 & 2)

Note on LDPC matrix
-------------------
The parity-check matrix H is constructed via a random regular LDPC ensemble
(column weight 3). Encoding uses GF(2) Gaussian elimination to derive a
systematic generator matrix G such that H G^T = 0, guaranteeing H @ cw = 0
for every encoded codeword.  Info bits sit at cw[0..k-1] by construction.

For exact 3GPP BG1/BG2 performance, load the lifted matrices from TS 38.212
Annex A and replace _build_H() below.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GF(2) utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Reduced row echelon form over GF(2).
    Returns (rref_matrix, pivot_columns).
    """
    M = M.copy().astype(np.uint8)
    nrows, ncols = M.shape
    pivot_cols = []
    row = 0
    for col in range(ncols):
        candidates = np.where(M[row:, col] == 1)[0]
        if len(candidates) == 0:
            continue
        r = candidates[0] + row
        M[[row, r]] = M[[r, row]]
        for r2 in range(nrows):
            if r2 != row and M[r2, col] == 1:
                M[r2] ^= M[row]
        pivot_cols.append(col)
        row += 1
        if row == nrows:
            break
    return M, pivot_cols


def _systematic_generator(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive systematic generator matrix G from H over GF(2).
    Returns (G, perm) where G is (k x n) in permuted column order and
    H[:, perm] = [A | I_m].  Info bits at G columns 0..k-1.
    """
    m, n = H.shape
    k = n - m

    Hrref, pivots = _gf2_rref(H)
    pivots = np.array(pivots)
    non_pivots = np.array([c for c in range(n) if c not in pivots])

    perm   = np.concatenate([non_pivots, pivots])
    H_perm = Hrref[:, perm]   # [A | I_m]

    A = H_perm[:, :k]         # (m x k)
    G = np.hstack([np.eye(k, dtype=np.uint8), (A.T % 2)])  # (k x n)
    return G, perm


# ─────────────────────────────────────────────────────────────────────────────
# Random regular LDPC parity-check matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_H(k: int, n: int, col_weight: int = 3, seed: int = 0) -> np.ndarray:
    """
    Build a random regular LDPC parity-check matrix H of size (n-k) x n.
    Column weight = col_weight.
    """
    m = n - k
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)
    for col in range(n):
        rows = rng.choice(m, size=col_weight, replace=False)
        H[rows, col] = 1
    for row in range(m):
        if H[row].sum() == 0:
            H[row, rng.integers(0, n)] = 1
    return H


# ─────────────────────────────────────────────────────────────────────────────
# LDPC Encoder
# ─────────────────────────────────────────────────────────────────────────────

class LDPCEncoder:
    """
    Systematic LDPC encoder.

    Uses a generator matrix G derived from H via GF(2) Gaussian elimination.
    Guarantees H @ cw = 0 for every encoded codeword.
    Info bits are at cw[0..k-1] by construction.

    Parameters
    ----------
    k : information bits per codeword
    n : total codeword bits  (rate = k/n)
    """

    def __init__(self, k: int = 128, n: int = 200):
        self.k    = k
        self.n    = n
        self.rate = k / n

        H_raw         = _build_H(k, n)
        self._G, perm = _systematic_generator(H_raw)
        # Keep H in permuted column order — consistent with codeword layout
        self.H        = H_raw[:, perm]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Systematic encode.  codeword[:k] = bits, codeword[k:] = parity bits.
        H @ codeword = 0 (mod 2) is guaranteed.

        Parameters
        ----------
        bits : (k,) binary array

        Returns
        -------
        codeword : (n,) binary array
        """
        assert len(bits) == self.k, f"Expected {self.k} bits, got {len(bits)}"
        return (bits.astype(np.uint8) @ self._G) % 2

    def rate_match(self, codeword: np.ndarray, e_bits: int) -> np.ndarray:
        """
        Circular-buffer rate matching to e_bits output length.
        Repeats codeword if e_bits > n, truncates if e_bits < n.

        Parameters
        ----------
        codeword : (n,) binary codeword
        e_bits   : desired output length

        Returns
        -------
        rm : (e_bits,) binary array
        """
        if e_bits <= self.n:
            return codeword[:e_bits]
        reps = (e_bits // self.n) + 1
        return np.tile(codeword, reps)[:e_bits]


# ─────────────────────────────────────────────────────────────────────────────
# BP Decoder — vectorised min-sum
# ─────────────────────────────────────────────────────────────────────────────

class BPDecoder:
    """
    Belief Propagation LDPC decoder — vectorised min-sum approximation.

    Uses min-sum (box-plus) instead of tanh/arctanh for speed and numerical
    stability. Within ~0.5 dB of sum-product for block sizes used here.
    Built once and reused across slots — do not instantiate inside a loop.

    Parameters
    ----------
    H        : parity-check matrix (n_check x n)
    max_iter : maximum BP iterations
    """

    def __init__(self, H: np.ndarray, max_iter: int = 50):
        self.H        = H
        self.max_iter = max_iter
        self.n_check, self.n = H.shape

        rows, cols      = np.where(H == 1)
        self._rows      = rows
        self._cols      = cols
        self._n_edges   = len(rows)
        self._cn_edges  = [np.where(rows == i)[0] for i in range(self.n_check)]
        self._vn_edges  = [np.where(cols == j)[0] for j in range(self.n)]

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """
        Decode using vectorised min-sum BP.

        Parameters
        ----------
        llrs : (n,) channel LLR values  (positive = bit likely 0)

        Returns
        -------
        bits : (n,) hard-decision decoded bits
        """
        v2c = llrs[self._cols].astype(float).copy()
        c2v = np.zeros(self._n_edges)

        for _ in range(self.max_iter):
            # Check-to-variable update (min-sum)
            for i, eidx in enumerate(self._cn_edges):
                if len(eidx) < 2:
                    continue
                msgs       = v2c[eidx]
                signs      = np.sign(msgs)
                abs_m      = np.abs(msgs)
                tot_sgn    = np.prod(signs)
                sorted_abs = np.sort(abs_m)
                min1, min2 = sorted_abs[0], sorted_abs[1]
                for k_local, e in enumerate(eidx):
                    sgn_ex = tot_sgn * signs[k_local] if signs[k_local] != 0 else tot_sgn
                    min_ex = min2 if abs_m[k_local] == min1 else min1
                    c2v[e] = float(sgn_ex) * min_ex

            # Total belief at each variable node
            total = llrs.copy()
            np.add.at(total, self._cols, c2v)

            # Variable-to-check update
            for j, eidx in enumerate(self._vn_edges):
                for e in eidx:
                    v2c[e] = total[j] - c2v[e]

            # Hard decision + early exit on valid codeword
            bits = (total < 0).astype(np.uint8)
            if np.all(self.H @ bits % 2 == 0):
                return bits

        return bits


# ─────────────────────────────────────────────────────────────────────────────
# Rate Matching  (TS 38.212 Section 5.4.2)
# ─────────────────────────────────────────────────────────────────────────────

class RateMatch:
    """
    LDPC Rate Matching / De-Rate Matching (circular buffer).
    3GPP TS 38.212 Section 5.4.2

    Parameters
    ----------
    n_cb   : circular buffer length (codeword length)
    e_bits : number of output bits
    rv     : redundancy version (0-3)
    """

    def __init__(self, n_cb: int, e_bits: int, rv: int = 0):
        self.n_cb   = n_cb
        self.e_bits = e_bits
        offsets     = [0, n_cb // 4, n_cb // 2, 3 * n_cb // 4]
        self.k0     = offsets[rv % 4]

    def match(self, codeword: np.ndarray) -> np.ndarray:
        """Extract e_bits from circular buffer starting at k0."""
        buf = np.tile(codeword, (self.e_bits // self.n_cb) + 2)
        return buf[self.k0: self.k0 + self.e_bits]

    def dematch(self, llrs: np.ndarray, n: int) -> np.ndarray:
        """
        De-rate-match: accumulate LLRs back into circular buffer positions.
        Combining rule: soft MRC accumulation (HARQ IR combining).
        """
        buf = np.zeros(n)
        for i, llr in enumerate(llrs):
            buf[(self.k0 + i) % n] += llr
        return buf