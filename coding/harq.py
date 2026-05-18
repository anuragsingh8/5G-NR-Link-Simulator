"""
coding/harq.py — HARQ Process, HARQ Manager, and Soft Buffer

This module implements a simplified HARQ (Hybrid Automatic Repeat reQuest)
framework similar to what is used in LTE and 5G NR systems.

Implemented components:
  • HARQProcess
      Represents one HARQ process state machine.

  • SoftBuffer
      Stores accumulated soft LLR values for retransmissions.

  • HARQManager
      Coordinates multiple HARQ processes and handles ACK/NACK feedback.

References:
  • 3GPP TS 38.212 Section 5.4
  • 3GPP TS 38.321 Section 5.4
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HARQProcess:
    """
    Represents one HARQ process.

    HARQ allows failed packets to be retransmitted while combining information
    from previous attempts.

    State transitions:

        IDLE
          ↓
        WAITING_ACK
          ↓
        ACK  → reset / complete
        NACK → retransmit

    Parameters
    ----------
    process_id : int
        HARQ process identifier.

    n_max_tx : int
        Maximum number of transmissions allowed for one transport block.

        Example:
          1 → no retransmissions
          4 → typical NR HARQ operation
    """

    # HARQ process index.
    process_id: int

    # Maximum allowed transmissions for one TB.
    n_max_tx: int = 4

    # ─────────────────────────────────────────────────────────────────────
    # Runtime state variables
    # ─────────────────────────────────────────────────────────────────────

    # Current transmission count.
    #
    # Example:
    #   1 → initial transmission
    #   2 → first retransmission
    n_tx: int = 0

    # True if transport block decoded successfully.
    acked: bool = False

    # Current redundancy version.
    #
    # NR redundancy sequence:
    #   0 → 2 → 3 → 1
    rv: int = 0

    # Transport block size in bits.
    tbs_bits: int = 0

    # Soft LLR accumulation buffer.
    soft_buffer: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    # Standard NR redundancy version order.
    #
    # Different RVs transmit different parity subsets.
    _RV_SEQ = [0, 2, 3, 1]

    @property
    def active(self) -> bool:
        """
        Return True if HARQ process is currently active.

        Active means:
          • at least one transmission occurred,
          • transport block not yet ACKed.
        """

        return self.n_tx > 0 and not self.acked

    @property
    def can_transmit(self) -> bool:
        """
        Check whether another retransmission is allowed.
        """

        return self.n_tx < self.n_max_tx and not self.acked

    def new_transmission(self, tbs_bits: int):
        """
        Initialize HARQ process for a new transport block.

        Parameters
        ----------
        tbs_bits : int
            Transport block size in bits.
        """

        # First transmission attempt.
        self.n_tx = 1

        # Block not yet acknowledged.
        self.acked = False

        # Start with first redundancy version.
        self.rv = self._RV_SEQ[0]

        # Store TB size.
        self.tbs_bits = tbs_bits

        # Clear previous soft combining buffer.
        self.soft_buffer = np.zeros(0)

    def retransmit(self):
        """
        Advance HARQ process to next redundancy version.

        Raises
        ------
        RuntimeError
            If maximum retransmissions already reached.
        """

        # Prevent retransmissions beyond configured HARQ limit.
        if self.n_tx >= self.n_max_tx:
            raise RuntimeError(
                f"HARQ process {self.process_id}: "
                f"max retransmissions reached"
            )

        # Increment transmission counter.
        self.n_tx += 1

        # Select next redundancy version cyclically.
        #
        # Example:
        #   tx1 → RV0
        #   tx2 → RV2
        #   tx3 → RV3
        #   tx4 → RV1
        self.rv = self._RV_SEQ[self.n_tx % 4]

    def combine(self, new_llrs: np.ndarray) -> np.ndarray:
        """
        Perform soft combining of retransmitted LLRs.

        HARQ combining improves decoding probability by accumulating reliability
        information from multiple transmissions.

        Parameters
        ----------
        new_llrs : np.ndarray
            New LLR values from current transmission.

        Returns
        -------
        np.ndarray
            Combined accumulated LLR values.
        """

        # First transmission:
        # initialize soft buffer directly.
        if len(self.soft_buffer) == 0:

            self.soft_buffer = new_llrs.copy()

        # Retransmission:
        # accumulate LLRs from multiple receptions.
        else:

            # Combine only overlapping portion.
            n = min(
                len(self.soft_buffer),
                len(new_llrs),
            )

            # Chase combining:
            # simply add LLR values together.
            self.soft_buffer[:n] += new_llrs[:n]

        # Return copy to avoid accidental external modification.
        return self.soft_buffer.copy()

    def ack(self):
        """
        Mark HARQ process as successfully decoded.
        """

        # Transport block decoded successfully.
        self.acked = True

        # Release soft combining memory.
        self.soft_buffer = np.array([])

    def reset(self):
        """
        Reset HARQ process back to idle state.
        """

        # Clear transmission counter.
        self.n_tx = 0

        # Clear ACK state.
        self.acked = False

        # Reset redundancy version.
        self.rv = 0

        # Reset TB size.
        self.tbs_bits = 0

        # Clear soft combining memory.
        self.soft_buffer = np.array([])


class SoftBuffer:
    """
    Global soft buffer manager.

    Stores accumulated LLR values for all HARQ processes.

    Each HARQ process owns its own LLR memory region.

    Parameters
    ----------
    n_processes : int
        Number of HARQ processes.

    buffer_size : int
        Maximum LLR buffer size per process.
    """

    def __init__(
        self,
        n_processes: int = 16,
        buffer_size: int = 25344,
    ):
        """
        Initialize soft buffer pool.
        """

        # Store number of HARQ processes.
        self.n_processes = n_processes

        # Maximum LLR memory size per process.
        self.buffer_size = buffer_size

        # Allocate zero-initialized LLR memory for every HARQ process.
        #
        # Dictionary structure:
        #   process_id → LLR buffer
        self._buffers: dict[int, np.ndarray] = {
            i: np.zeros(buffer_size)
            for i in range(n_processes)
        }

    def accumulate(
        self,
        process_id: int,
        llrs: np.ndarray,
        k0: int = 0,
    ):
        """
        Accumulate incoming LLRs into HARQ soft buffer.

        Parameters
        ----------
        process_id : int
            HARQ process index.

        llrs : np.ndarray
            Incoming LLR values.

        k0 : int
            Offset position inside the soft buffer.

            Different RVs may map to different offsets.
        """

        # Retrieve process-specific buffer.
        buf = self._buffers[process_id]

        # Prevent writing beyond allocated memory.
        n = min(
            len(llrs),
            self.buffer_size - k0,
        )

        # Accumulate soft information.
        buf[k0: k0 + n] += llrs[:n]

    def get(self, process_id: int) -> np.ndarray:
        """
        Retrieve accumulated LLR values for one process.
        """

        return self._buffers[process_id].copy()

    def reset(self, process_id: int):
        """
        Clear one HARQ process soft buffer.
        """

        self._buffers[process_id][:] = 0.0


class HARQManager:
    """
    Central HARQ controller.

    Manages:
      • HARQ process allocation,
      • retransmissions,
      • ACK/NACK handling,
      • soft combining memory,
      • HARQ statistics.

    Parameters
    ----------
    n_processes : int
        Number of parallel HARQ processes.

    n_max_tx : int
        Maximum transmissions allowed per TB.
    """

    def __init__(
        self,
        n_processes: int = 8,
        n_max_tx: int = 4,
    ):
        """
        Initialize HARQ manager.
        """

        # Store number of HARQ processes.
        self.n_processes = n_processes

        # Create all HARQ process objects.
        self.processes = [
            HARQProcess(i, n_max_tx)
            for i in range(n_processes)
        ]

        # Shared soft buffer manager.
        self.soft_buffer = SoftBuffer(n_processes)

        # Runtime HARQ statistics.
        self._stats = {
            "total_tx": 0,
            "acks": 0,
            "nacks": 0,
            "dtx": 0,
        }

    def get_free_process(self) -> Optional[HARQProcess]:
        """
        Find first idle HARQ process.

        Returns
        -------
        HARQProcess | None
            Free process if available.
        """

        for p in self.processes:

            # Inactive process is available for new scheduling.
            if not p.active:
                return p

        # No free HARQ process available.
        return None

    def get_process(self, pid: int) -> HARQProcess:
        """
        Retrieve HARQ process by ID.
        """

        return self.processes[pid]

    def schedule_new_tx(
        self,
        pid: int,
        tbs_bits: int,
    ):
        """
        Schedule new transport block transmission.

        Parameters
        ----------
        pid : int
            HARQ process ID.

        tbs_bits : int
            Transport block size.
        """

        # Retrieve HARQ process.
        p = self.processes[pid]

        # Initialize new TB transmission.
        p.new_transmission(tbs_bits)

        # Clear previous soft combining memory.
        self.soft_buffer.reset(pid)

        # Increment transmission statistics.
        self._stats["total_tx"] += 1

    def handle_feedback(
        self,
        pid: int,
        ack: bool,
        new_llrs: Optional[np.ndarray] = None,
    ):
        """
        Process ACK/NACK feedback from receiver.

        Parameters
        ----------
        pid : int
            HARQ process ID.

        ack : bool
            True  → ACK
            False → NACK

        new_llrs : np.ndarray | None
            Newly received soft information.
        """

        # Retrieve HARQ process.
        p = self.processes[pid]

        # ACK:
        # decoding successful.
        if ack:

            # Mark HARQ process complete.
            p.ack()

            # Release soft memory.
            self.soft_buffer.reset(pid)

            # Update ACK statistics.
            self._stats["acks"] += 1

        # NACK:
        # decoding failed.
        else:

            # Increment NACK counter.
            self._stats["nacks"] += 1

            # Store incoming soft information for combining.
            if new_llrs is not None:
                self.soft_buffer.accumulate(
                    pid,
                    new_llrs,
                )

            # If retransmissions are still allowed:
            if p.can_transmit:

                # Move to next redundancy version.
                p.retransmit()

            # Otherwise flush HARQ process completely.
            else:

                # Reset HARQ process state.
                p.reset()

                # Clear soft combining memory.
                self.soft_buffer.reset(pid)

                # Count dropped transport block.
                self._stats["dtx"] += 1

    def combined_llrs(self, pid: int) -> np.ndarray:
        """
        Retrieve accumulated soft values for decoder.
        """

        return self.soft_buffer.get(pid)

    @property
    def stats(self) -> dict:
        """
        Return HARQ performance statistics.
        """

        # Avoid division by zero.
        total = max(
            self._stats["acks"] + self._stats["nacks"],
            1,
        )

        # Return raw counters plus estimated HARQ BLER.
        return {
            **self._stats,

            # HARQ BLER:
            # fraction of transmissions resulting in NACK.
            "bler_harq": self._stats["nacks"] / total,
        }