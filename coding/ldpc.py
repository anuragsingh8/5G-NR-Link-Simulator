"""
coding/ldpc.py — LDPC Encoder, Belief-Propagation Decoder, and Rate Matching

This module implements a lightweight LDPC (Low Density Parity Check) coding
chain inspired by 5G NR.

Implemented components:
  • LDPCEncoder
      Systematic LDPC encoder.

  • BPDecoder
      Belief Propagation LDPC decoder using the Sum-Product Algorithm.

  • RateMatch
      Circular-buffer rate matching and de-rate-matching.

References:
  • 3GPP TS 38.212 Section 5.3
  • 3GPP TS 38.212 Section 5.4
"""

from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Small illustrative Base Graph construction
# ─────────────────────────────────────────────────────────────────────────────
#
# Real 5G NR uses:
#   • Base Graph 1 (BG1)
#   • Base Graph 2 (BG2)
#
# These are large quasi-cyclic LDPC matrices defined by 3GPP.
#
# For educational simulation purposes, this implementation generates a small
# random sparse parity-check matrix instead of loading full 3GPP tables.
#
# LDPC matrices are sparse:
#   • few 1s,
#   • many 0s.
#
# Sparse structure enables:
#   • efficient decoding,
#   • iterative message passing,
#   • near-Shannon-limit performance.
# ─────────────────────────────────────────────────────────────────────────────


def _make_bg1_small(
    k: int = 22,
    n: int = 68,
) -> np.ndarray:
    """
    Generate a small sparse LDPC parity-check matrix.

    Parameters
    ----------
    k : int
        Number of information bits.

    n : int
        Total codeword length.

    Returns
    -------
    np.ndarray
        Binary parity-check matrix H with shape:
            (n-k, n)
    """

    # Create deterministic random generator for reproducibility.
    rng = np.random.default_rng(0)

    # Number of parity-check equations.
    n_check = n - k

    # Allocate parity-check matrix initialized to zeros.
    #
    # dtype=uint8 is sufficient because entries are binary.
    H = np.zeros((n_check, n), dtype=np.uint8)

    # Construct sparse columns.
    #
    # Each column corresponds to one codeword bit.
    # Each bit participates in exactly 3 parity equations.
    for col in range(n):

        # Randomly choose 3 rows where this bit participates.
        rows = rng.choice(
            n_check,
            size=3,
            replace=False,
        )

        # Set parity-check connections.
        H[rows, col] = 1

    # Ensure every parity-check equation has at least one participating bit.
    #
    # This prevents invalid empty rows.
    for row in range(n_check):

        if H[row].sum() == 0:

            # Randomly activate one bit in this row.
            H[row, rng.integers(0, n)] = 1

    # Return generated LDPC parity-check matrix.
    return H


class LDPCEncoder:
    """
    Systematic LDPC encoder.

    Systematic encoding means:

        codeword = [information_bits | parity_bits]

    The original information bits appear unchanged at the start of the
    codeword.

    Parameters
    ----------
    k : int
        Number of information bits.

    n : int
        Total codeword length.
    """

    def __init__(
        self,
        k: int = 22,
        n: int = 68,
    ):
        """
        Initialize LDPC encoder.
        """

        # Information bit length.
        self.k = k

        # Total codeword length.
        self.n = n

        # Coding rate:
        #
        #   R = k / n
        #
        # Higher rate:
        #   more throughput,
        #   less redundancy.
        self.rate = k / n

        # Generate parity-check matrix.
        self.H = _make_bg1_small(k, n)

        # Number of parity bits.
        self.n_check = n - k

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Systematic LDPC encoding.

        Parameters
        ----------
        bits : np.ndarray
            Information bits with shape ``(k,)``.

        Returns
        -------
        np.ndarray
            Encoded codeword with shape ``(n,)``.
        """

        # Ensure correct information block length.
        assert len(bits) == self.k

        # Copy information bits.
        #
        # These remain unchanged in systematic encoding.
        u = bits.copy()

        # Split parity-check matrix:
        #
        #   H = [H_s | H_p]
        #
        # H_s:
        #   systematic part operating on information bits.
        #
        # H_p:
        #   parity portion operating on parity bits.
        H_s = self.H[:, :self.k]

        H_p = self.H[:, self.k:]

        # Compute syndrome contribution from information bits:
        #
        #   s = H_s * u
        #
        # All operations occur in GF(2).
        s = H_s @ u % 2

        # Allocate parity bit vector.
        p = np.zeros(
            self.n_check,
            dtype=np.uint8,
        )

        # Compute parity bits sequentially using back-substitution.
        #
        # This assumes H_p behaves approximately like a lower-triangular matrix.
        for i in range(self.n_check):

            # Compute parity equation modulo-2.
            p[i] = (
                s[i]
                + H_p[i, :i] @ p[:i]
            ) % 2

        # Final systematic codeword:
        #
        #   [information | parity]
        return np.concatenate([u, p])

    def rate_match(
        self,
        codeword: np.ndarray,
        e_bits: int,
    ) -> np.ndarray:
        """
        Perform simple circular-buffer rate matching.

        Parameters
        ----------
        codeword : np.ndarray
            Encoded LDPC codeword.

        e_bits : int
            Desired number of transmitted bits.

        Returns
        -------
        np.ndarray
            Rate-matched output bits.
        """

        # Circular buffer length.
        n_cb = len(codeword)

        # If fewer bits are needed than available:
        # simply puncture/truncate.
        if e_bits <= n_cb:
            return codeword[:e_bits]

        # Otherwise repeat circularly.
        #
        # Used for low coding rates or retransmissions.
        reps = (e_bits // n_cb) + 1

        # Tile repeatedly and truncate to requested size.
        return np.tile(codeword, reps)[:e_bits]


class BPDecoder:
    """
    Belief Propagation LDPC decoder.

    Uses iterative Sum-Product Algorithm (SPA).

    Decoder operates on:
      • variable nodes (bits),
      • check nodes (parity equations).

    Messages are exchanged iteratively until:
      • parity checks are satisfied,
      • or maximum iterations reached.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix.

    max_iter : int
        Maximum decoding iterations.
    """

    def __init__(
        self,
        H: np.ndarray,
        max_iter: int = 50,
    ):
        """
        Initialize BP decoder.
        """

        # Store parity-check matrix.
        self.H = H

        # Maximum allowed decoding iterations.
        self.max_iter = max_iter

        # Matrix dimensions:
        #
        #   n_check = parity equations
        #   n       = codeword bits
        self.n_check, self.n = H.shape

        # ─────────────────────────────────────────────────────────────────
        # Precompute graph connectivity
        # ─────────────────────────────────────────────────────────────────
        #
        # LDPC decoding operates on a Tanner graph.
        #
        # Variable node:
        #   codeword bit
        #
        # Check node:
        #   parity equation
        # ─────────────────────────────────────────────────────────────────

        # Check-node neighbors:
        #
        # For each parity equation,
        # store connected bit indices.
        self._cn_nbrs = [
            np.where(H[i] == 1)[0]
            for i in range(self.n_check)
        ]

        # Variable-node neighbors:
        #
        # For each bit,
        # store connected parity equations.
        self._vn_nbrs = [
            np.where(H[:, j] == 1)[0]
            for j in range(self.n)
        ]

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """
        Decode LDPC codeword using belief propagation.

        Parameters
        ----------
        llrs : np.ndarray
            Channel LLR values.

        Returns
        -------
        np.ndarray
            Hard-decision decoded bits.
        """

        # ─────────────────────────────────────────────────────────────────
        # Initialize variable-to-check messages
        # ─────────────────────────────────────────────────────────────────
        #
        # Initial belief for every edge comes directly from channel LLR.
        # ─────────────────────────────────────────────────────────────────
        v2c = {
            (j, i): llrs[j]
            for i in range(self.n_check)
            for j in self._cn_nbrs[i]
        }

        # Iterative decoding loop.
        for _ in range(self.max_iter):

            # ─────────────────────────────────────────────────────────────
            # Check-node update
            # ─────────────────────────────────────────────────────────────
            #
            # Computes parity consistency messages using tanh rule:
            #
            #   m_c→v =
            #       2 atanh( Π tanh(m_v→c / 2) )
            # ─────────────────────────────────────────────────────────────
            c2v: dict[tuple, float] = {}

            for i in range(self.n_check):

                # Neighboring bits connected to this parity equation.
                nbrs = self._cn_nbrs[i]

                for j in nbrs:

                    # All neighboring bits except current one.
                    others = [
                        k for k in nbrs
                        if k != j
                    ]

                    # No neighbors:
                    # message becomes neutral.
                    if not others:
                        c2v[(i, j)] = 0.0
                        continue

                    # Compute product term.
                    prod = np.prod([
                        np.tanh(v2c[(k, i)] / 2)
                        for k in others
                    ])

                    # Numerical protection:
                    # avoid atanh(±1).
                    prod = np.clip(
                        prod,
                        -1 + 1e-10,
                        1 - 1e-10,
                    )

                    # Final check-node message.
                    c2v[(i, j)] = 2 * np.arctanh(prod)

            # ─────────────────────────────────────────────────────────────
            # Variable-node update
            # ─────────────────────────────────────────────────────────────
            #
            # Combine:
            #   channel LLR
            #   +
            #   incoming parity messages
            # ─────────────────────────────────────────────────────────────
            for j in range(self.n):

                nbrs = self._vn_nbrs[j]

                # Total belief for this bit.
                total = (
                    llrs[j]
                    + sum(c2v[(i, j)] for i in nbrs)
                )

                # Send extrinsic information to each neighboring check node.
                for i in nbrs:
                    v2c[(j, i)] = total - c2v[(i, j)]

            # ─────────────────────────────────────────────────────────────
            # Hard decision
            # ─────────────────────────────────────────────────────────────
            #
            # Positive LLR → bit 0
            # Negative LLR → bit 1
            # ─────────────────────────────────────────────────────────────
            bits = np.array([
                0
                if (
                    llrs[j]
                    + sum(
                        c2v[(i, j)]
                        for i in self._vn_nbrs[j]
                    )
                ) >= 0
                else 1
                for j in range(self.n)
            ], dtype=np.uint8)

            # ─────────────────────────────────────────────────────────────
            # Syndrome check
            # ─────────────────────────────────────────────────────────────
            #
            # Valid codeword satisfies:
            #
            #   H * c = 0  (mod 2)
            # ─────────────────────────────────────────────────────────────
            if np.all(self.H @ bits % 2 == 0):

                # Early stopping:
                # decoding successful.
                return bits

        # Maximum iterations reached.
        #
        # Return best estimate obtained so far.
        return bits


class RateMatch:
    """
    LDPC Rate Matching and De-Rate Matching.

    Implements simplified circular-buffer extraction similar to NR.

    Different redundancy versions (RVs) use different starting offsets inside
    the circular buffer.

    Parameters
    ----------
    n_cb : int
        Circular buffer length.

    e_bits : int
        Number of transmitted bits.

    rv : int
        Redundancy version index.
    """

    def __init__(
        self,
        n_cb: int,
        e_bits: int,
        rv: int = 0,
    ):
        """
        Initialize rate matcher.
        """

        # Circular buffer size.
        self.n_cb = n_cb

        # Requested output bit count.
        self.e_bits = e_bits

        # RV-dependent offsets.
        #
        # Different RVs expose different parity subsets.
        offsets = [
            0,
            n_cb // 4,
            n_cb // 2,
            3 * n_cb // 4,
        ]

        # Starting extraction offset.
        self.k0 = offsets[rv % 4]

    def match(self, codeword: np.ndarray) -> np.ndarray:
        """
        Extract transmitted bits from circular buffer.

        Parameters
        ----------
        codeword : np.ndarray
            Full LDPC codeword.

        Returns
        -------
        np.ndarray
            Rate-matched output bits.
        """

        # Extend circular buffer by repetition.
        #
        # Extra repetition ensures extraction window always fits.
        buf = np.tile(
            codeword,
            (self.e_bits // self.n_cb) + 2,
        )

        # Extract transmission window beginning at RV offset.
        return buf[
            self.k0:
            self.k0 + self.e_bits
        ]

    def dematch(
        self,
        llrs: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """
        De-rate-match received LLR values.

        Parameters
        ----------
        llrs : np.ndarray
            Received soft LLR values.

        n : int
            Original circular buffer size.

        Returns
        -------
        np.ndarray
            Reconstructed circular-buffer LLRs.
        """

        # Allocate reconstructed circular buffer.
        buf = np.zeros(n)

        # Scatter incoming LLRs back into original RV positions.
        for i, llr in enumerate(llrs):

            # Circular indexing enables HARQ combining.
            buf[(self.k0 + i) % n] += llr

        # Return reconstructed soft buffer.
        return buf