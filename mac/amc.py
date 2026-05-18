"""
mac/amc.py — Adaptive Modulation & Coding

This module implements simple link adaptation logic used in wireless systems.

Main components:
  • LinkAdaptor
      Maps CQI reports to MCS index and applies an SNR/MCS offset.

  • OLLAController
      Outer-Loop Link Adaptation controller that adjusts the offset based on
      ACK/NACK feedback.

Reference:
  3GPP TS 38.214 Section 5.1 / 5.2
"""

from __future__ import annotations

import numpy as np
from collections import deque


class LinkAdaptor:
    """
    CQI-to-MCS link adaptation block.

    CQI stands for Channel Quality Indicator.
    It is reported by the receiver to describe current channel quality.

    MCS stands for Modulation and Coding Scheme.
    It controls:
      • modulation order,
      • coding rate,
      • spectral efficiency,
      • robustness of transmission.

    Higher CQI generally allows higher MCS.
    Higher MCS gives higher throughput but needs better channel quality.
    """

    # CQI-to-MCS lookup table.
    #
    # This table is a simplified precomputed mapping inspired by
    # 3GPP TS 38.214 CQI efficiency matching.
    #
    # CQI 0 means the channel is out of range, so the lowest MCS is selected.
    _CQI_TO_MCS: dict[int, int] = {
        0: 0,
        1: 0,
        2: 1,
        3: 2,
        4: 4,
        5: 6,
        6: 8,
        7: 10,
        8: 12,
        9: 14,
        10: 16,
        11: 18,
        12: 20,
        13: 23,
        14: 26,
        15: 28,
    }

    def __init__(
        self,
        target_bler: float = 0.1,
        snr_margin: float = 1.0,
    ):
        """
        Initialize the link adaptor.

        Parameters
        ----------
        target_bler : float
            Target block error rate.

        snr_margin : float
            Conservative SNR margin in dB.
        """

        # Store target BLER for reference.
        self.target_bler = target_bler

        # Store configured SNR margin.
        self.snr_margin = snr_margin

        # Current selected MCS index.
        self._mcs_idx = 0

        # Dynamic offset controlled by OLLA.
        self._snr_offset = 0.0

    def select_mcs(self, cqi: int) -> int:
        """
        Select MCS index from CQI report.

        Parameters
        ----------
        cqi : int
            CQI index in the range 0 to 15.

        Returns
        -------
        int
            Selected MCS index in the range 0 to 28.
        """

        # Look up nominal MCS from reported CQI.
        # Unknown CQI values fall back to safest MCS 0.
        base_mcs = self._CQI_TO_MCS.get(cqi, 0)

        # Apply dynamic OLLA offset.
        #
        # In this simplified implementation, 1 dB of offset is treated roughly
        # like one MCS step.
        adjusted_mcs = base_mcs + round(self._snr_offset)

        # Clamp MCS to valid range.
        self._mcs_idx = int(np.clip(adjusted_mcs, 0, 28))

        # Return final selected MCS.
        return self._mcs_idx

    def update_offset(self, delta: float):
        """
        Update internal SNR/MCS offset.

        Parameters
        ----------
        delta : float
            Offset change supplied by OLLA controller.
        """

        # Accumulate offset and restrict it to a reasonable range.
        self._snr_offset = np.clip(
            self._snr_offset + delta,
            -6.0,
            6.0,
        )

    @property
    def current_mcs(self) -> int:
        """
        Return the most recently selected MCS index.
        """

        return self._mcs_idx


class OLLAController:
    """
    Outer-Loop Link Adaptation controller.

    OLLA adjusts the link adaptation offset using ACK/NACK feedback.

    Idea:
      • ACK means the block decoded successfully.
      • NACK means the block failed.
      • Too many NACKs means the selected MCS is too aggressive.
      • Too many ACKs means the link may support higher throughput.

    The offset is updated with:
      • a larger upward step after NACK,
      • a smaller downward step after ACK.

    This drives the long-term BLER toward the configured target.
    """

    def __init__(
        self,
        target_bler: float = 0.10,
        step_up: float = 1.0,
        step_down: float = 0.1,
        window_size: int = 100,
    ):
        """
        Initialize OLLA controller.

        Parameters
        ----------
        target_bler : float
            Desired BLER target.

        step_up : float
            Offset increase after NACK.

        step_down : float
            Offset decrease after ACK.

        window_size : int
            Number of recent ACK/NACK results used for BLER estimation.
        """

        # Store target BLER.
        self.target_bler = target_bler

        # NACK step. This is usually larger because failures need fast recovery.
        self.step_up = step_up

        # ACK step. If not explicitly provided, derive it from target BLER.
        #
        # The relation keeps the average offset stable when observed BLER
        # matches the target BLER.
        self.step_down = (
            step_down
            if step_down
            else step_up * target_bler / (1 - target_bler)
        )

        # Current accumulated OLLA offset in dB-like units.
        self._offset = 0.0

        # Sliding window history:
        #   0 = ACK
        #   1 = NACK
        self._history: deque[int] = deque(maxlen=window_size)

        # Optional LinkAdaptor instance that receives offset updates directly.
        self._adaptor: LinkAdaptor | None = None

    def attach(self, adaptor: LinkAdaptor):
        """
        Attach a LinkAdaptor to this OLLA controller.

        Once attached, every OLLA update is injected into the adaptor so future
        MCS selections include the latest offset.
        """

        self._adaptor = adaptor

    def update(self, ack: bool) -> float:
        """
        Update OLLA offset from ACK/NACK feedback.

        Parameters
        ----------
        ack : bool
            True if transport block decoded successfully.
            False if decoding failed.

        Returns
        -------
        float
            Current OLLA offset.
        """

        # Save latest decoding outcome.
        #
        # ACK is stored as 0.
        # NACK is stored as 1.
        self._history.append(0 if ack else 1)

        # Successful decoding means the link may be conservative, so reduce
        # offset slowly.
        if ack:
            self._offset -= self.step_down

        # Failed decoding means selected MCS was too aggressive, so increase
        # offset quickly.
        else:
            self._offset += self.step_up

        # Limit offset so it cannot grow without bound.
        self._offset = np.clip(
            self._offset,
            -10.0,
            10.0,
        )

        # Push offset into attached LinkAdaptor, if one exists.
        if self._adaptor:
            self._adaptor.update_offset(self._offset)

        # Return updated offset.
        return self._offset

    @property
    def estimated_bler(self) -> float:
        """
        Estimate BLER over the recent ACK/NACK history.
        """

        # No samples means no block errors observed yet.
        if not self._history:
            return 0.0

        # Since NACK is stored as 1 and ACK as 0, the mean gives BLER.
        return float(np.mean(list(self._history)))

    @property
    def offset_db(self) -> float:
        """
        Return current OLLA offset.
        """

        return float(self._offset)

    def status(self) -> dict:
        """
        Return current OLLA status as a dictionary.
        """

        return {
            "snr_offset_db": self.offset_db,
            "estimated_bler": self.estimated_bler,
            "target_bler": self.target_bler,
            "n_samples": len(self._history),
        }