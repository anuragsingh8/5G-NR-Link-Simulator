"""
sim/metrics.py — Simulation Metric Counters
  • BERCounter        — Bit Error Rate
  • BLERCounter       — Block Error Rate
  • ThroughputCalc    — Throughput in Mbps
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BERCounter:
    """Accumulates bit errors for BER computation."""

    n_bits:   int = 0
    n_errors: int = 0

    def update(self, tx_bits: np.ndarray, rx_bits: np.ndarray):
        """
        Compare transmitted and received bits.

        Parameters
        ----------
        tx_bits : (N,) transmitted binary array
        rx_bits : (N,) decoded binary array
        """
        assert len(tx_bits) == len(rx_bits)
        self.n_bits   += len(tx_bits)
        self.n_errors += int(np.sum(tx_bits != rx_bits))

    @property
    def ber(self) -> float:
        """Bit Error Rate."""
        return self.n_errors / max(self.n_bits, 1)

    def reset(self):
        self.n_bits = 0
        self.n_errors = 0

    def __repr__(self) -> str:
        return f"BERCounter(BER={self.ber:.2e}, errors={self.n_errors}/{self.n_bits})"


@dataclass
class BLERCounter:
    """Accumulates block errors for BLER computation."""

    n_blocks:      int = 0
    n_block_errors: int = 0

    def update(self, tx_bits: np.ndarray, rx_bits: np.ndarray):
        """
        Check if transport block was decoded correctly.

        Parameters
        ----------
        tx_bits : transmitted transport block bits
        rx_bits : decoded transport block bits
        """
        self.n_blocks += 1
        if not np.array_equal(tx_bits, rx_bits):
            self.n_block_errors += 1

    def update_from_ack(self, ack: bool):
        """Update directly from ACK/NACK (when CRC is used)."""
        self.n_blocks += 1
        if not ack:
            self.n_block_errors += 1

    @property
    def bler(self) -> float:
        """Block Error Rate."""
        return self.n_block_errors / max(self.n_blocks, 1)

    def reset(self):
        self.n_blocks = 0
        self.n_block_errors = 0

    def __repr__(self) -> str:
        return f"BLERCounter(BLER={self.bler:.4f}, {self.n_block_errors}/{self.n_blocks})"


class ThroughputCalc:
    """
    Computes average and instantaneous throughput.

    Parameters
    ----------
    slot_duration_us : slot duration in microseconds (default: 0.5ms for μ=1)
    """

    def __init__(self, slot_duration_us: float = 500.0, bw_hz: float = 20e6):
        self.slot_duration_us = slot_duration_us
        self._bw_hz = bw_hz
        self._total_bits: int = 0
        self._total_slots: int = 0
        self._tput_history: list[float] = []

    def update(self, tbs_bits: int, ack: bool, n_slots: int = 1):
        """
        Record one slot's worth of data.

        Parameters
        ----------
        tbs_bits : transport block size in bits
        ack      : True if block was decoded correctly
        n_slots  : number of slots this TB spans
        """
        self._total_slots += n_slots
        if ack:
            self._total_bits += tbs_bits

        instant_tput = (tbs_bits if ack else 0) / (n_slots * self.slot_duration_us)  # Mbps
        self._tput_history.append(instant_tput)

    @property
    def avg_throughput_mbps(self) -> float:
        """Average throughput in Mbps."""
        total_time_us = self._total_slots * self.slot_duration_us
        return self._total_bits / max(total_time_us, 1.0)

    @property
    def peak_throughput_mbps(self) -> float:
        return float(np.max(self._tput_history)) if self._tput_history else 0.0

    @property
    def spectral_efficiency(self) -> float:
        """Bits/s/Hz based on configured bandwidth."""
        return self.avg_throughput_mbps * 1e6 / max(self._bw_hz, 1.0)

    def set_bandwidth(self, bw_hz: float):
        """Set channel bandwidth for spectral efficiency calculation."""
        self._bw_hz = bw_hz

    def reset(self):
        self._total_bits = 0
        self._total_slots = 0
        self._tput_history.clear()

    def summary(self) -> dict:
        return {
            "avg_throughput_mbps": round(self.avg_throughput_mbps, 4),
            "peak_throughput_mbps": round(self.peak_throughput_mbps, 4),
            "spectral_efficiency_bps_hz": round(self.spectral_efficiency, 4),
            "total_bits_decoded": self._total_bits,
            "total_slots": self._total_slots,
        }

    def __repr__(self) -> str:
        return (f"ThroughputCalc(avg={self.avg_throughput_mbps:.3f} Mbps, "
                f"slots={self._total_slots})")
    