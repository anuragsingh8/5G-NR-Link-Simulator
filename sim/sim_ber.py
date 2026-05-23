"""
sim_ber.py — BER Simulation & Plotting
=====================================

Purpose
-------
This script validates the physical-layer OFDM implementation using
Bit Error Rate (BER) simulations.

It compares:
    • simulated BER
    • theoretical BER

for multiple modulation orders over different wireless channels.

Supported Modulations
---------------------
    • BPSK
    • QPSK
    • 16-QAM
    • 64-QAM
    • 256-QAM

Supported Channels
------------------
    • AWGN
        Noise only.

    • Flat Rayleigh
        Single-tap fading channel.

    • EPA
        Multipath fading channel from 3GPP.

Outputs
-------
The script generates:

    ber_compare.png
        Simulated vs theoretical BER curves.

    frame_structure.png
        NR slot / PRB / DMRS visualization.

    cp_analysis.png
        Cyclic prefix analysis vs delay spread.

    mcs_comparison.png
        BER waterfall comparison for representative NR MCS levels.

Simulation Flow
---------------
For every Eb/N0 point:

    random bits
        ↓
    QAM mapper
        ↓
    OFDM modulation
        ↓
    wireless channel
        ↓
    OFDM demodulation
        ↓
    equalisation
        ↓
    hard demapper
        ↓
    BER computation

Theoretical BER equations are plotted together with simulated points
to validate correctness of the PHY chain.

Key Concepts
------------

Eb/N0
-----
    Energy per information bit divided by noise spectral density.

    Used for modulation BER analysis.

OFDM
----
    Orthogonal Frequency Division Multiplexing.

    Converts a frequency-selective channel into many narrowband
    flat-fading subcarriers.

Cyclic Prefix (CP)
------------------
    A guard interval copied from the end of the OFDM symbol.

    CP duration must exceed maximum channel delay spread to avoid:

        • ISI  (Inter-Symbol Interference)
        • ICI  (Inter-Carrier Interference)

Example:
    μ=0 (15 kHz SCS)
    CP ≈ 4.69 μs

    EPA delay spread:
        0.41 μs

    Since:
        4.69 μs > 0.41 μs

    OFDM orthogonality is preserved.

Perfect CSI Assumption
----------------------
Equalisation currently assumes perfect channel knowledge:

    equalized_symbol = received_symbol / channel_estimate

This isolates modulation/channel behaviour from channel-estimation errors.

References
----------
    • 3GPP TS 38.211
    • 3GPP TS 38.214
    • Digital Communications theory
"""

from __future__ import annotations

# Numerical computing.
import numpy as np

# Force matplotlib to use non-interactive backend.
#
# Important for:
#   • remote execution
#   • headless servers
#   • CI systems
#   • Docker environments
import matplotlib
matplotlib.use('Agg')

# Plotting library.
import matplotlib.pyplot as plt

# Used for custom plot shapes/annotations.
import matplotlib.patches as mpatches

# Grid-based subplot layouts.
from matplotlib.gridspec import GridSpec

# Runtime measurement.
import time

# System/path utilities.
import sys, os


# Add project root to import path.
#
# Allows:
#   python sim_ber.py
#
# without package-install requirements.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..')
)


# ─────────────────────────────────────────────────────────────────────────────
# OFDM / NR imports
# ─────────────────────────────────────────────────────────────────────────────

from phy.ofdm import (
    OFDMModulator,
    NRFrameStructure,
    NR_NUMEROLOGIES,
    dmrs_pilot_sequence,
)

from phy.channel import (

    # Channel models
    AWGNChannel,
    FlatRayleighChannel,
    EPAChannel,

    # Theoretical BER equations
    ber_bpsk_awgn,
    ber_qpsk_awgn,
    ber_qam_awgn,
    ber_rayleigh_bpsk,
    ber_rayleigh_qpsk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Gray-code helper
# ─────────────────────────────────────────────────────────────────────────────
#
# Gray coding ensures adjacent constellation points differ by only one bit.
#
# This minimizes BER because nearest-neighbour symbol errors typically
# flip only one bit.
#
# Formula:
#
#   gray(n) = n XOR (n >> 1)
#
# Example:
#
#   Binary:
#       00 01 10 11
#
#   Gray:
#       00 01 11 10
# ─────────────────────────────────────────────────────────────────────────────

def gray_code(n: int) -> np.ndarray:
    """
    Compute Gray-code index.
    """

    return n ^ (n >> 1)


# ─────────────────────────────────────────────────────────────────────────────
# QAM constellation generator
# ─────────────────────────────────────────────────────────────────────────────
#
# Generates normalized Gray-coded constellations.
#
# Supported:
#
#   M=2    → BPSK
#   M=4    → QPSK
#   M=16   → 16QAM
#   M=64   → 64QAM
#   M=256  → 256QAM
#
# Average constellation power is normalized to 1:
#
#   E[|s|²] = 1
#
# This ensures fair Eb/N0 comparison across modulations.
# ─────────────────────────────────────────────────────────────────────────────

def qam_constellation(M: int) -> np.ndarray:
    """
    Return Gray-coded complex constellation.
    """

    # ─────────────────────────────────────────────────────────────────
    # BPSK special case
    # ─────────────────────────────────────────────────────────────────
    #
    # BPSK is 1D:
    #
    #   bit 0 → -1
    #   bit 1 → +1
    #
    # No imaginary axis used.
    # ─────────────────────────────────────────────────────────────────
    if M == 2:
        return np.array([
            -1.0 + 0j,
             1.0 + 0j,
        ])

    # Number of bits per symbol.
    m = int(np.log2(M))

    # Ensure valid square QAM order.
    assert (
        2 ** m == M
        and m % 2 == 0
    ), "M must be square QAM order"

    # Example:
    #
    #   16QAM → sqrtM = 4
    #   64QAM → sqrtM = 8
    sqrtM = int(np.sqrt(M))

    # ─────────────────────────────────────────────────────────────────
    # Generate PAM axis
    # ─────────────────────────────────────────────────────────────────
    #
    # Example:
    #
    #   16QAM:
    #       [-3, -1, +1, +3]
    #
    # Gray ordering applied.
    # ─────────────────────────────────────────────────────────────────
    pam = np.array([
        gray_code(i)
        for i in range(sqrtM)
    ], dtype=float)

    # Center around zero.
    pam = 2 * pam - (sqrtM - 1)

    # Explicit Gray-code ordering.
    pam_sorted = np.zeros(sqrtM)

    for i in range(sqrtM):

        pam_sorted[gray_code(i)] = (
            2 * i - (sqrtM - 1)
        )

    # ─────────────────────────────────────────────────────────────────
    # Build 2D QAM constellation
    # ─────────────────────────────────────────────────────────────────
    #
    # Every real-axis value combines with every imaginary-axis value.
    #
    # Example:
    #
    #   16QAM:
    #
    #       (-3-3j), (-3-1j), ...
    # ─────────────────────────────────────────────────────────────────
    const = np.array([

        r + 1j * i

        for r in pam_sorted
        for i in pam_sorted[::-1]
    ])

    # Normalize average power to 1.
    const /= np.sqrt(
        np.mean(np.abs(const) ** 2)
    )

    return const