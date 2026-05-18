"""
phy/ofdm.py — OFDM Modulator / Demodulator

This module converts between:

  • frequency-domain QAM symbols
  • time-domain OFDM samples

OFDM is used in LTE, WiFi, and 5G NR because it splits a wideband channel into
many narrowband subcarriers that are easier to equalise.

Reference:
  3GPP TS 38.211 Section 5.3 / 7.4
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift


class OFDMModulator:
    """
    OFDM modulator.

    Converts one row of frequency-domain symbols into one time-domain OFDM
    symbol by:
      1. placing active QAM symbols on FFT bins,
      2. zero-padding unused subcarriers,
      3. applying IFFT,
      4. adding cyclic prefix.

    Parameters
    ----------
    n_fft : int
        FFT size.

    cp_len : int
        Cyclic prefix length in samples.

    n_subcarriers : int
        Number of active subcarriers. Must be less than or equal to n_fft.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        cp_len: int = 144,
        n_subcarriers: int = 624,
    ):
        """
        Initialize OFDM modulator settings.
        """

        # Active subcarriers must fit inside the FFT grid.
        assert n_subcarriers <= n_fft, "n_subcarriers must be <= n_fft"

        # Store FFT size. This controls total frequency bins and time samples.
        self.n_fft = n_fft

        # Store cyclic prefix length.
        self.cp_len = cp_len

        # Store number of active data/pilot subcarriers.
        self.n_subcarriers = n_subcarriers

        # One transmitted OFDM symbol contains the IFFT output plus cyclic prefix.
        self.symbol_len = n_fft + cp_len

    def _map_to_grid(self, symbols: np.ndarray) -> np.ndarray:
        """
        Map active subcarrier symbols into the full FFT grid.

        Parameters
        ----------
        symbols : np.ndarray
            Active frequency-domain symbols with shape ``(n_subcarriers,)``.

        Returns
        -------
        np.ndarray
            Full FFT grid with shape ``(n_fft,)``.
        """

        # Create full frequency-domain grid initialized with zeros.
        # Inactive subcarriers remain zero-padded.
        grid = np.zeros(self.n_fft, dtype=complex)

        # Split active subcarriers equally around DC.
        half = self.n_subcarriers // 2

        # Map positive-frequency subcarriers.
        #
        # Bin 0 is DC, so it is intentionally skipped.
        # symbols[half:] are placed after DC.
        grid[1: half + 1] = symbols[half:]

        # Map negative-frequency subcarriers.
        #
        # These are placed at the end of the FFT vector.
        grid[-half:] = symbols[:half]

        # Return DC-centered frequency grid.
        return grid

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Modulate one OFDM symbol.

        Parameters
        ----------
        symbols : np.ndarray
            Frequency-domain QAM symbols with shape ``(n_subcarriers,)``.

        Returns
        -------
        np.ndarray
            Time-domain OFDM symbol with cyclic prefix.
        """

        # Validate input size.
        assert symbols.shape[-1] == self.n_subcarriers

        # Place active subcarriers into full FFT grid.
        freq = self._map_to_grid(symbols)

        # Convert frequency-domain symbols to time domain using IFFT.
        #
        # ifftshift aligns the DC-centered grid with NumPy's IFFT ordering.
        # sqrt(n_fft) keeps power scaling consistent.
        time = ifft(ifftshift(freq)) * np.sqrt(self.n_fft)

        # Copy the last cp_len samples to the front as cyclic prefix.
        #
        # This protects the OFDM symbol from multipath delay spread.
        cp = time[-self.cp_len:]

        # Prepend cyclic prefix to the OFDM time-domain symbol.
        return np.concatenate([cp, time])

    def modulate_slot(self, symbol_grid: np.ndarray) -> np.ndarray:
        """
        Modulate a full slot containing multiple OFDM symbols.

        Parameters
        ----------
        symbol_grid : np.ndarray
            Frequency-domain grid with shape ``(n_symb, n_subcarriers)``.

        Returns
        -------
        np.ndarray
            Serialized time-domain transmit waveform.
        """

        # Modulate each OFDM symbol row and concatenate them in time order.
        return np.concatenate([
            self.modulate(row)
            for row in symbol_grid
        ])


class OFDMDemodulator:
    """
    OFDM demodulator.

    Converts received time-domain OFDM samples back into frequency-domain
    subcarrier symbols by:
      1. removing cyclic prefix,
      2. applying FFT,
      3. extracting active subcarriers.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        cp_len: int = 144,
        n_subcarriers: int = 624,
    ):
        """
        Initialize OFDM demodulator settings.
        """

        # Store FFT size used by the transmitter.
        self.n_fft = n_fft

        # Store cyclic prefix length.
        self.cp_len = cp_len

        # Store number of active subcarriers.
        self.n_subcarriers = n_subcarriers

        # Total samples per OFDM symbol including CP.
        self.symbol_len = n_fft + cp_len

    def _extract_subcarriers(self, freq: np.ndarray) -> np.ndarray:
        """
        Extract active subcarriers from full FFT output.

        Parameters
        ----------
        freq : np.ndarray
            Full FFT output with shape ``(n_fft,)``.

        Returns
        -------
        np.ndarray
            Active subcarrier symbols with shape ``(n_subcarriers,)``.
        """

        # Number of active subcarriers on each side of DC.
        half = self.n_subcarriers // 2

        # Allocate output vector for active subcarriers.
        symbols = np.empty(self.n_subcarriers, dtype=complex)

        # Extract positive-frequency bins.
        #
        # These were originally mapped to grid[1:half+1].
        symbols[half:] = freq[1: half + 1]

        # Extract negative-frequency bins.
        #
        # These were originally mapped to grid[-half:].
        symbols[:half] = freq[-half:]

        # Return active subcarrier symbols in original ordering.
        return symbols

    def demodulate(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Demodulate one OFDM symbol.

        Parameters
        ----------
        rx_signal : np.ndarray
            Time-domain received OFDM symbol including cyclic prefix.

        Returns
        -------
        np.ndarray
            Recovered frequency-domain active subcarriers.
        """

        # Remove cyclic prefix.
        time = rx_signal[self.cp_len:]

        # Convert time-domain signal back to frequency domain.
        #
        # fftshift returns bins to the DC-centered layout used by extraction.
        freq = fftshift(fft(time)) / np.sqrt(self.n_fft)

        # Extract only active subcarriers.
        return self._extract_subcarriers(freq)

    def demodulate_slot(
        self,
        rx_signal: np.ndarray,
        n_symb: int = 14,
    ) -> np.ndarray:
        """
        Demodulate a full slot.

        Parameters
        ----------
        rx_signal : np.ndarray
            Serialized received time-domain slot waveform.

        n_symb : int
            Number of OFDM symbols in the slot.

        Returns
        -------
        np.ndarray
            Frequency-domain symbol grid with shape
            ``(n_symb, n_subcarriers)``.
        """

        # Allocate output grid for all demodulated OFDM symbols.
        grid = np.zeros(
            (n_symb, self.n_subcarriers),
            dtype=complex,
        )

        # Process each OFDM symbol separately.
        for i in range(n_symb):

            # Compute start index of this OFDM symbol in the time waveform.
            start = i * self.symbol_len

            # Slice one complete OFDM symbol including cyclic prefix.
            symbol = rx_signal[start: start + self.symbol_len]

            # Demodulate this symbol and store it in the output grid.
            grid[i] = self.demodulate(symbol)

        # Return full slot resource grid.
        return grid