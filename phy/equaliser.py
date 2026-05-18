"""
phy/equaliser.py — MIMO Equalisers

This module contains linear and non-linear equalisation methods used in
MIMO-OFDM receivers.

Equalisers implemented here:

  • ZFEqualiser
      Zero Forcing equaliser. Removes inter-layer interference by inverting
      the channel, but can amplify noise.

  • MMSEEqualiser
      Linear Minimum Mean Square Error equaliser. Balances interference
      suppression and noise enhancement.

  • SICEqualiser
      Successive Interference Cancellation equaliser, also known as V-BLAST.
      Detects one transmit layer at a time and cancels its contribution from
      the received signal.

References:
  3GPP TS 38.211 / 38.212
"""

from __future__ import annotations

import numpy as np


def _pseudo_inv(H: np.ndarray) -> np.ndarray:
    """
    Compute the Moore-Penrose pseudo-inverse of a channel matrix.

    The pseudo-inverse is used when the channel matrix is not square or is not
    directly invertible. This is common in MIMO systems where the number of
    receive antennas and transmit layers may be different.

    Parameters
    ----------
    H : np.ndarray
        Channel matrix.

    Returns
    -------
    np.ndarray
        Pseudo-inverse of H.
    """

    # NumPy computes the pseudo-inverse internally using SVD, which is more
    # numerically stable than manually applying a direct matrix inverse.
    return np.linalg.pinv(H)


class ZFEqualiser:
    """
    Zero-Forcing MIMO equaliser.

    The ZF equaliser tries to completely remove inter-layer interference by
    applying the inverse, or pseudo-inverse, of the channel matrix.

    Mathematical form:

        W_ZF = (HᴴH)^(-1) Hᴴ

    where:
        Hᴴ is the Hermitian transpose of H.

    ZF works well when the channel is strong and well-conditioned, but it can
    amplify noise when the channel matrix is close to singular.
    """

    def __init__(self, n_tx: int = 4, n_rx: int = 2):
        """
        Initialize the Zero-Forcing equaliser.

        Parameters
        ----------
        n_tx : int
            Number of transmit antennas or spatial layers.

        n_rx : int
            Number of receive antennas.
        """

        # Store number of transmit layers.
        self.n_tx = n_tx

        # Store number of receive antennas.
        self.n_rx = n_rx

    def equalise(self, rx: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Equalise the received MIMO signal using Zero Forcing.

        Parameters
        ----------
        rx : np.ndarray
            Received signal with shape ``(n_rx, n_sc)``.
            Each column corresponds to one subcarrier.

        H : np.ndarray
            Channel matrix with shape ``(n_rx, n_tx, n_sc)``.
            A separate MIMO channel matrix is provided for each subcarrier.

        Returns
        -------
        np.ndarray
            Equalised transmit-layer symbols with shape ``(n_tx, n_sc)``.
        """

        # Number of OFDM subcarriers.
        n_sc = rx.shape[-1]

        # Allocate output buffer for estimated transmitted symbols.
        x_hat = np.zeros((self.n_tx, n_sc), dtype=complex)

        # Equalise each subcarrier independently because each subcarrier has
        # its own channel matrix in OFDM.
        for k in range(n_sc):

            # Extract channel matrix for subcarrier k.
            # Shape: (n_rx, n_tx)
            Hk = H[:, :, k]

            # Compute pseudo-inverse of the channel matrix.
            # Shape: (n_tx, n_rx)
            W = _pseudo_inv(Hk)

            # Apply the equaliser to the received vector on this subcarrier.
            # This estimates the transmitted symbols from each layer.
            x_hat[:, k] = W @ rx[:, k]

        # Return equalised symbols for all transmit layers and subcarriers.
        return x_hat


class MMSEEqualiser:
    """
    Linear Minimum Mean Square Error equaliser.

    The MMSE equaliser improves on ZF by considering noise power. Instead of
    forcing interference completely to zero, it balances interference removal
    with noise amplification.

    Mathematical form:

        W_MMSE = (HᴴH + σ²I)^(-1) Hᴴ

    where:
        σ² is the noise variance,
        I is the identity matrix.

    MMSE usually performs better than ZF at low or moderate SNR.
    """

    def __init__(self, n_tx: int = 4, n_rx: int = 2, snr_db: float = 15.0):
        """
        Initialize the MMSE equaliser.

        Parameters
        ----------
        n_tx : int
            Number of transmit antennas or layers.

        n_rx : int
            Number of receive antennas.

        snr_db : float
            Operating SNR in decibels.
        """

        # Store number of transmit layers.
        self.n_tx = n_tx

        # Store number of receive antennas.
        self.n_rx = n_rx

        # Convert SNR in dB into an approximate noise variance.
        #
        # Since:
        #   SNR_linear = 10^(SNR_dB / 10)
        #
        # then:
        #   noise_variance ≈ 1 / SNR_linear = 10^(-SNR_dB / 10)
        self.sigma2 = 10 ** (-snr_db / 10)

    def equalise(self, rx: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Equalise the received signal using the MMSE criterion.

        Parameters
        ----------
        rx : np.ndarray
            Received signal with shape ``(n_rx, n_sc)``.

        H : np.ndarray
            Channel matrix with shape ``(n_rx, n_tx, n_sc)``.

        Returns
        -------
        np.ndarray
            Estimated transmitted symbols with shape ``(n_tx, n_sc)``.
        """

        # Number of OFDM subcarriers.
        n_sc = rx.shape[-1]

        # Allocate output array for equalised symbols.
        x_hat = np.zeros((self.n_tx, n_sc), dtype=complex)

        # Identity matrix used for MMSE noise regularisation.
        I = np.eye(self.n_tx)

        # Process each subcarrier separately.
        for k in range(n_sc):

            # Extract channel matrix for current subcarrier.
            Hk = H[:, :, k]

            # Compute HᴴH, where Hᴴ is the conjugate transpose of H.
            # This captures layer-to-layer coupling through the channel.
            HhH = Hk.conj().T @ Hk

            # Solve:
            #   (HᴴH + σ²I) W = Hᴴ
            #
            # This is numerically better than explicitly computing the inverse.
            W = np.linalg.solve(
                HhH + self.sigma2 * I,
                Hk.conj().T,
            )

            # Apply MMSE equaliser to recover transmitted layers.
            x_hat[:, k] = W @ rx[:, k]

        # Return equalised symbol estimates.
        return x_hat

    def post_snr(self, H: np.ndarray) -> np.ndarray:
        """
        Estimate post-equalisation SNR for each layer and subcarrier.

        This is useful for soft demodulation and LLR scaling, where the decoder
        needs to know how reliable each equalised symbol is.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix with shape ``(n_rx, n_tx, n_sc)``.

        Returns
        -------
        np.ndarray
            Post-equalisation SNR with shape ``(n_tx, n_sc)``.
        """

        # Number of subcarriers.
        n_sc = H.shape[-1]

        # Allocate output SNR matrix.
        snr_post = np.zeros((self.n_tx, n_sc))

        # Identity matrix used in MMSE filtering.
        I = np.eye(self.n_tx)

        # Compute post-SNR independently for each subcarrier.
        for k in range(n_sc):

            # Extract channel matrix for this subcarrier.
            Hk = H[:, :, k]

            # Compute Gram matrix HᴴH.
            HhH = Hk.conj().T @ Hk

            # Compute equivalent MMSE gain matrix:
            #
            #   G = (HᴴH + σ²I)^(-1) HᴴH
            #
            # The diagonal values indicate how much of each layer survives
            # after MMSE equalisation.
            G = np.linalg.solve(
                HhH + self.sigma2 * I,
                HhH,
            )

            # Extract diagonal terms and keep them in a safe numerical range.
            # This avoids division by zero in the SINR calculation.
            g = np.real(np.diag(G)).clip(1e-9, 1 - 1e-9)

            # Approximate SINR from MMSE gain:
            #
            #   SINR = g / (1 - g)
            snr_post[:, k] = g / (1 - g)

        # Return post-equalisation SNR values.
        return snr_post


class SICEqualiser:
    """
    Successive Interference Cancellation equaliser.

    SIC detects transmit layers one at a time. After detecting one layer, it
    reconstructs that layer's received contribution and subtracts it from the
    received signal.

    This is similar to the V-BLAST receiver structure.

    Detection strategy:
      1. Estimate all remaining layers using MMSE.
      2. Choose the strongest layer based on post-MMSE SNR.
      3. Make a hard decision on that layer.
      4. Cancel its contribution from the received signal.
      5. Repeat until all layers are detected.
    """

    def __init__(self, n_tx: int = 4, n_rx: int = 2, snr_db: float = 15.0):
        """
        Initialize the SIC equaliser.

        Parameters
        ----------
        n_tx : int
            Number of transmit antennas or layers.

        n_rx : int
            Number of receive antennas.

        snr_db : float
            Operating SNR in decibels.
        """

        # Store number of transmit layers.
        self.n_tx = n_tx

        # Store number of receive antennas.
        self.n_rx = n_rx

        # Store approximate noise variance.
        self.sigma2 = 10 ** (-snr_db / 10)

        # Create an internal MMSE equaliser used during each SIC step.
        self._mmse = MMSEEqualiser(n_tx, n_rx, snr_db)

    def _nearest_qpsk(self, x: complex) -> complex:
        """
        Map a complex symbol to the nearest QPSK constellation point.

        This hard decision is used during cancellation. The detected symbol is
        reconstructed as a clean QPSK symbol before subtracting it from the
        received signal.

        Parameters
        ----------
        x : complex
            Equalised complex symbol.

        Returns
        -------
        complex
            Nearest normalized QPSK symbol.
        """

        # Decide real and imaginary signs independently.
        #
        # Positive real part maps to +1, negative maps to -1.
        # Positive imaginary part maps to +j, negative maps to -j.
        s = (np.sign(x.real) + 1j * np.sign(x.imag)) / np.sqrt(2)

        # Return normalized QPSK constellation point.
        return s

    def equalise(self, rx: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Equalise the received signal using SIC / V-BLAST detection.

        Parameters
        ----------
        rx : np.ndarray
            Received signal with shape ``(n_rx, n_sc)``.

        H : np.ndarray
            Channel matrix with shape ``(n_rx, n_tx, n_sc)``.

        Returns
        -------
        np.ndarray
            Estimated transmitted symbols with shape ``(n_tx, n_sc)``.
        """

        # Number of subcarriers.
        n_sc = rx.shape[-1]

        # Allocate final output symbol matrix.
        x_hat = np.zeros((self.n_tx, n_sc), dtype=complex)

        # Residual received signal.
        # This is updated after each detected layer is cancelled.
        rx_res = rx.copy()

        # Residual channel matrix.
        # Kept as a copy so the original H is not modified.
        H_res = H.copy()

        # List of transmit layers that have already been detected.
        detected = []

        # Repeat until all transmit layers are detected.
        for step in range(self.n_tx):

            # Build list of layers that are still not detected.
            remaining = [i for i in range(self.n_tx) if i not in detected]

            # Stop early if all layers have already been processed.
            if not remaining:
                break

            # Extract channel matrix for the remaining layers only.
            H_sub = H_res[:, remaining, :]

            # MMSE-equalise the residual received signal using only the
            # remaining layers.
            x_sub = self._mmse.equalise(rx_res, H_sub)

            # Estimate reliability of each remaining layer.
            snr_sub = self._mmse.post_snr(H_sub)

            # Choose the layer with the highest average post-MMSE SNR.
            # This detects the most reliable layer first.
            best_local = int(np.argmax(snr_sub.mean(axis=1)))

            # Convert local index within "remaining" back to original layer ID.
            best_global = remaining[best_local]

            # Mark this layer as detected.
            detected.append(best_global)

            # Store the soft MMSE estimate for the detected layer.
            x_hat[best_global] = x_sub[best_local]

            # Cancel the detected layer from the residual received signal.
            for k in range(n_sc):

                # Convert soft symbol estimate to nearest QPSK symbol.
                s_hat = self._nearest_qpsk(x_hat[best_global, k])

                # Subtract reconstructed contribution:
                #
                #   received contribution = H[:, layer, k] * detected_symbol
                rx_res[:, k] -= H_res[:, best_global, k] * s_hat

        # Return estimated symbols for all transmit layers.
        return x_hat