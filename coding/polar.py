"""
coding/polar.py — Polar Encoder & CA-SCL Decoder

This module implements a simplified Polar coding chain inspired by
5G NR control channel coding.

Implemented components:
  • PolarEncoder
      Systematic polar encoder with CRC attachment.

  • CASCLDecoder
      CRC-Aided Successive Cancellation List decoder.

References:
  • 3GPP TS 38.212 Section 5.3
  • Polar coding theory by Arıkan
"""

from __future__ import annotations

import numpy as np

from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# Polar reliability sequence
# ─────────────────────────────────────────────────────────────────────────────
#
# Polar coding relies on "channel polarization".
#
# Some bit positions become:
#   • highly reliable,
# while others become:
#   • highly unreliable.
#
# Information bits are placed only into the most reliable positions.
#
# In real 5G NR:
#   • a standardized 1024-element reliability sequence is used.
#
# Here:
#   • reliability is generated algorithmically using Bhattacharyya recursion.
# ─────────────────────────────────────────────────────────────────────────────


def _reliability_sequence(n: int) -> np.ndarray:
    """
    Compute reliability ordering for polar bit channels.

    Parameters
    ----------
    n : int
        Polar code block length.

    Returns
    -------
    np.ndarray
        Indices sorted from least reliable → most reliable.
    """

    # Allocate Bhattacharyya parameter array.
    #
    # Smaller values:
    #   more reliable channel.
    #
    # Larger values:
    #   noisier channel.
    z = np.zeros(n)

    # Initial Bhattacharyya parameter.
    #
    # z=0:
    #   perfect channel
    #
    # z=1:
    #   useless channel
    z[0] = 0.5

    # Recursively polarize channels.
    #
    # Each stage doubles number of polarized channels.
    for step in range(int(np.log2(n))):

        # Number of existing channels before expansion.
        half = 2 ** step

        # Allocate next-stage reliability array.
        new_z = np.zeros(2 * half)

        for i in range(half):

            # "Bad" polarized channel:
            #
            # reliability decreases.
            new_z[2 * i] = (
                2 * z[i] - z[i] ** 2
            )

            # "Good" polarized channel:
            #
            # reliability improves.
            new_z[2 * i + 1] = z[i] ** 2

        # Move to next polarization stage.
        z = new_z

    # Sort channels:
    #
    # least reliable → most reliable
    return np.argsort(z)


class PolarEncoder:
    """
    Systematic Polar encoder with CRC attachment.

    Polar encoding process:
      1. attach CRC,
      2. place bits into reliable positions,
      3. freeze unreliable positions to zero,
      4. apply polar transform.

    Parameters
    ----------
    k : int
        Number of information bits including CRC.

    n : int
        Polar codeword length.
        Must be power of 2.
    """

    # CRC-6 polynomial used for simplified UCI simulation.
    #
    # Real NR supports multiple CRC sizes depending on channel type.
    CRC_POLY = 0x1B

    # Number of CRC bits.
    CRC_LEN = 6

    def __init__(
        self,
        k: int = 32,
        n: int = 64,
    ):
        """
        Initialize polar encoder.
        """

        # Polar block length must be power-of-two.
        assert (n & (n - 1)) == 0, "n must be a power of 2"

        # Information size cannot exceed block length.
        assert k <= n

        # Store payload size including CRC.
        self.k = k

        # Store codeword length.
        self.n = n

        # Compute reliability ordering.
        reliability = _reliability_sequence(n)

        # Select most reliable positions for information bits.
        #
        # These carry:
        #   payload + CRC
        self.info_idx = np.sort(
            reliability[-k:]
        )

        # Remaining positions become frozen bits.
        #
        # Frozen bits are fixed to zero.
        self.frozen_idx = np.sort(
            reliability[:-k]
        )

        # Precompute polar generator matrix:
        #
        #   G_N = F^(⊗m)
        #
        # where:
        #   F = [[1,0],[1,1]]
        self._G = self._kernel_power(
            int(np.log2(n))
        )

    @staticmethod
    def _kernel_power(m: int) -> np.ndarray:
        """
        Compute Kronecker power of polar kernel matrix.

        Parameters
        ----------
        m : int
            Number of polarization stages.

        Returns
        -------
        np.ndarray
            Polar generator matrix.
        """

        # Base polar kernel:
        #
        #   [1 0]
        #   [1 1]
        #
        # This performs recursive XOR operations.
        F = np.array([
            [1, 0],
            [1, 1]
        ], dtype=np.uint8)

        # Initialize result with first kernel.
        result = F

        # Kronecker expansion:
        #
        # Generates:
        #   F^(⊗m)
        #
        # Example:
        #   m=2 → 4x4 matrix
        for _ in range(m - 1):

            result = np.kron(result, F)

        return result

    def _crc6(self, bits: np.ndarray) -> np.ndarray:
        """
        Compute CRC-6 checksum.

        Parameters
        ----------
        bits : np.ndarray
            Payload bits.

        Returns
        -------
        np.ndarray
            CRC bits.
        """

        # Shift-register initialized to zero.
        reg = 0

        # Process input bits sequentially.
        for b in bits:

            # Extract MSB before shift.
            msb = (reg >> 5) & 1

            # Shift left and insert next bit.
            reg = (
                (reg << 1) | int(b)
            ) & 0x3F

            # Apply generator polynomial if feedback bit active.
            if msb:
                reg ^= self.CRC_POLY

        # Convert CRC register into bit array.
        crc = np.array([
            (reg >> (5 - i)) & 1
            for i in range(6)
        ], dtype=np.uint8)

        return crc

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Polar encode payload bits.

        Parameters
        ----------
        bits : np.ndarray
            Payload bits excluding CRC.

        Returns
        -------
        np.ndarray
            Polar codeword.
        """

        # Generate CRC bits.
        crc = self._crc6(bits)

        # Append CRC to payload.
        u_info = np.concatenate([
            bits,
            crc
        ]).astype(np.uint8)

        # Validate payload size.
        assert len(u_info) == self.k

        # Allocate full polar input vector.
        #
        # Frozen positions remain zero.
        u = np.zeros(
            self.n,
            dtype=np.uint8,
        )

        # Place information bits into reliable positions.
        u[self.info_idx] = u_info

        # Polar transform:
        #
        #   x = u * G_N  (mod 2)
        #
        # Matrix multiplication occurs over GF(2).
        return (u @ self._G) % 2


class CASCLDecoder:
    """
    CRC-Aided Successive Cancellation List decoder.

    CA-SCL improves standard SC decoding by:
      • keeping multiple decoding candidates,
      • pruning unlikely paths,
      • selecting CRC-valid candidate at the end.

    Parameters
    ----------
    k : int
        Number of information bits including CRC.

    n : int
        Polar block length.

    L : int
        List size.

        Larger L:
          • better performance,
          • higher complexity.
    """

    def __init__(
        self,
        k: int = 32,
        n: int = 64,
        L: int = 8,
    ):
        """
        Initialize CA-SCL decoder.
        """

        # Polar length must be power-of-two.
        assert (n & (n - 1)) == 0

        # Store payload size.
        self.k = k

        # Store block length.
        self.n = n

        # Store list size.
        self.L = L

        # Compute reliability ordering.
        reliability = _reliability_sequence(n)

        # Information-bit locations.
        self.info_idx = set(
            reliability[-k:]
        )

        # Frozen-bit locations.
        self.frozen_idx = set(
            reliability[:-k]
        )

        # Internal encoder reused for CRC verification.
        self._encoder = PolarEncoder(k, n)

    def _f(self, a: float, b: float) -> float:
        """
        Min-sum f-node update.

        Used in SC decoding tree.
        """

        return (
            np.sign(a)
            * np.sign(b)
            * min(abs(a), abs(b))
        )

    def _g(
        self,
        a: float,
        b: float,
        u: int,
    ) -> float:
        """
        g-node update rule.

        Depends on previously decoded bit u.
        """

        return b + (1 - 2 * u) * a

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """
        Perform CA-SCL decoding.

        Parameters
        ----------
        llrs : np.ndarray
            Channel LLR values.

        Returns
        -------
        np.ndarray
            Decoded payload bits excluding CRC.
        """

        # Local aliases for readability.
        n = self.n
        L = self.L

        # ─────────────────────────────────────────────────────────────────
        # Initialize decoding list
        # ─────────────────────────────────────────────────────────────────
        #
        # Each path contains:
        #   (llrs, decoded_bits, path_metric)
        #
        # Path metric:
        #   lower = more likely candidate.
        # ─────────────────────────────────────────────────────────────────
        paths = [
            (
                llrs.copy(),
                np.zeros(n, dtype=np.uint8),
                0.0,
            )
        ]

        # Decode one bit position at a time.
        for i in range(n):

            # Candidate paths after branching.
            new_paths = []

            # Expand every active path.
            for path_llr, path_bits, pm in paths:

                # Simplified SC decoder:
                #
                # directly use current channel LLR.
                channel_llr = path_llr[i]

                # ─────────────────────────────────────────────────────────
                # Frozen bit
                # ─────────────────────────────────────────────────────────
                #
                # Frozen bits must equal zero.
                # Any evidence supporting bit=1 increases penalty.
                # ─────────────────────────────────────────────────────────
                if i in self.frozen_idx:

                    new_pm = (
                        pm
                        + max(0.0, -channel_llr)
                    )

                    new_paths.append((
                        path_llr,
                        path_bits.copy(),
                        new_pm,
                    ))

                # ─────────────────────────────────────────────────────────
                # Information bit
                # ─────────────────────────────────────────────────────────
                #
                # Fork decoding path into:
                #   bit=0
                #   bit=1
                # ─────────────────────────────────────────────────────────
                else:

                    for bit in (0, 1):

                        # Copy current decoded bits.
                        b = path_bits.copy()

                        # Insert candidate decision.
                        b[i] = bit

                        # Compute path metric penalty.
                        #
                        # Penalty increases when chosen bit disagrees with LLR.
                        penalty = max(
                            0.0,
                            (2 * bit - 1) * channel_llr,
                        )

                        # Store branched path.
                        new_paths.append((
                            path_llr,
                            b,
                            pm + penalty,
                        ))

            # ─────────────────────────────────────────────────────────────
            # Path pruning
            # ─────────────────────────────────────────────────────────────
            #
            # Keep only best L candidates.
            #
            # Smaller path metric:
            #   higher likelihood.
            # ─────────────────────────────────────────────────────────────
            new_paths.sort(
                key=lambda x: x[2]
            )

            paths = new_paths[:L]

        # ─────────────────────────────────────────────────────────────────
        # CRC validation
        # ─────────────────────────────────────────────────────────────────
        #
        # CRC selects the most likely valid candidate.
        # ─────────────────────────────────────────────────────────────────
        enc = self._encoder

        for _, bits, _ in paths:

            # Extract information bits from reliable positions.
            info_bits = bits[
                sorted(self.info_idx)
            ]

            # Split payload and received CRC.
            payload = info_bits[:-enc.CRC_LEN]

            crc_recv = info_bits[-enc.CRC_LEN:]

            # Recompute CRC.
            crc_calc = enc._crc6(payload)

            # CRC match:
            # decoding successful.
            if np.array_equal(
                crc_recv,
                crc_calc,
            ):

                return payload

        # ─────────────────────────────────────────────────────────────────
        # CRC failure fallback
        # ─────────────────────────────────────────────────────────────────
        #
        # If no candidate passes CRC:
        # return best-metric candidate anyway.
        # ─────────────────────────────────────────────────────────────────
        best_bits = paths[0][1]

        info_bits = best_bits[
            sorted(self.info_idx)
        ]

        return info_bits[:-enc.CRC_LEN]