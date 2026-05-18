Why Channel Estimation Is Needed?

Imagine hearing someone in a large stadium with echoes, background noise, multiple reflections.
But still you understand what being said. Brain subconsciously estimates: how the environment distorted the sound,which parts are echoes, what the original voice should be. Wireless receivers do the same thing mathematically.

Similarly, in wireless communication the transmitted signal does not travel perfectly from transmitter to receiver. Instead, the signal experiences: 
a. Fading,
b. Reflections, 
c. Interference, 
d. Attenuation, 
e. Phase shifts,
f. Doppler effects,
g. Multipath propagation.

Because of this, the receiver gets a distorted version of the original signal.The receiver therefore does not observe the original transmitted symbols directly. Instead, the received OFDM signal is approximately:
Y[k]=H[k]X[k]+N[k]

Where:
X[k] → transmitted symbol on subcarrier k
H[k] → channel response (unknown)
N[k] → noise
Y[k] → received symbol

The receiver wants to recover: X[k] but the channel H[k] is unknown.So before decoding data, the receiver must first estimate the channel. This is called Channel Estimation
The receiver must estimate H[k] so it can recover the transmitted data.

Real-World Use Case 1 — 5G Mobile Phone Downloading Video Scenario
Some is watching YouTube on your phone while driving. Your phone communicates with a 5G base station while moving, surrounded by buildings, cars, trees, other users.

What Happens to the Signal:
The transmitted signal reaches your phone through many paths:
a. Base Station <--> Direct-Path <--> Phone
b. Additional reflected paths:
    - buildings
    - glass
    - cars
    - walls
Where each path:- has different delay, different attenuation, different phase. These paths combine at the receiver.

Original transmitted symbol:
X = 1 + j
Now Suppose there is no Channel estimation, the receiver cannot directly determine whether the distortion came from:
fading, phase rotation, noise, interference. So decoding fails resulting in corrupted video, buffering, dropped packets,low throughput.
In case no channel estimation: Received: Y = 0.3 + 1.7j

With Channel Estimation The receiver estimates:
H ≈ 0.5 + 1.2j

Then equalizes the signal:
X^[k]= Y[k]/H^[k]
Now the recovered signal becomes close to the original transmitted symbol resulting in stable video, higher data rate,fewer errors,smooth streaming.

Why OFDM Especially Needs Channel Estimation
5G uses OFDM. OFDM divides bandwidth into many narrow subcarriers. Example: hundreds or thousands of subcarriers. Each subcarrier experiences:slightly different fading i.e.,
Subcarrier 1 → strong signal
Subcarrier 2 → weak signal
Subcarrier 3 → phase shifted
Subcarrier 4 → deep fade

So the receiver must estimate: H[k] for every subcarrier.
This is why pilot tones (DMRS) are inserted.

Real-World Use Case 2 — High-Speed Train
A passenger on a high-speed train uses 5G internet. Now the channel changes extremely fast because of:
a. Doppler shift,
b. rapid movement,
c. continuously changing reflections.
d. Problem now is that channel at time t becomes invalid shortly afterward.
e. Meaning Channel now is H(t) but a few milliseconds later: H(t + Δt) ≠ H(t)
Without continuous channel estimation: symbols decode incorrectly, packets fail, communication drops.
Solution: 5G inserts DMRS pilot symbols periodically.
The receiver continuously:
measures pilots,
estimates channel,
updates equalizer,
tracks mobility.

This enables: uninterrupted connectivity, handovers, reliable communication at high speed.

Real-World Use Case 3 — Massive MIMO
Modern 5G base stations may use: 64 antennas, 128 antennas, beamforming. Each antenna path has a different channel. Without channel estimation beamforming cannot work.
because beamforming requires knowing:
Which antenna phase/amplitude will constructively combine at the user. The base station estimates channels and computes beamforming weights.
Resulting in stronger signal, longer range, higher throughput, reduced interference.

A simplified receiver chain:
Received RF Signal
        ↓
ADC Sampling
        ↓
OFDM FFT
        ↓
DMRS Extraction
        ↓
Channel Estimation
        ↓
Equalization
        ↓
MIMO Detection
        ↓
Demodulation
        ↓
Decoding
        ↓
Recovered Bits
Channel estimation is one of the most critical stages.


DMRS Pilots (Reference Signals): Both estimators use DMRS pilot tones.Demodulation Reference Signal are special symbols known by both:
a. transmitter
b. receiver

Because the receiver already knows what was transmitted on pilot subcarriers, it can compare:
a. what was sent
b. what was received
to estimate the channel

self.pilot_idx = get_pilot_indices(n_subcarriers)
Pilots are placed on:
a. even-numbered subcarriers
b. every other frequency tone

Example:
Subcarrier index:
0 1 2 3 4 5 6 7
Pilot locations:
P   P   P   P
1. LS Estimator (Least Squares) assumes:
"The channel at a pilot subcarrier can be estimated directly by dividing the received pilot by the transmitted pilot."

Mathematically:
H^LS[k]=Y[k]/X[k]
	​
This comes directly from the received signal equation:
Y[k]=H[k]X[k]+N[k]

Ignoring noise:
H[k]≈ Y[k]/ X[k]

Step-by-Step LS Estimation
Step 1 — Extract Received Pilot Tones

Code:
rx_pilot_symbol[self.pilot_idx] => This selects only pilot subcarriers from the received OFDM symbol.
Example:
Received OFDM symbol:
[y0 y1 y2 y3 y4 y5 y6 y7]

Pilot positions: 0,2,4,6
Extracted: [y0 y2 y4 y6]

Step 2 — Divide by Known Pilots

Code:
h_pilot = rx_pilot_symbol[self.pilot_idx] / self.pilots

This computes:
H^LS[k] => for pilot locations only.

Step 3 — Interpolate Missing Subcarriers Because pilots exist only on some subcarriers, the estimator fills in the rest.
Code:
np.interp(...)

Interpolation assumes nearby subcarriers experience similar fading. So the channel between two pilot tones is approximated linearly.

Example: Pilot estimates:
H0 = 1.0
H2 = 0.8

Estimate H1 ≈ 0.9

Advantages:
LS Estimator is computationally cheap and Fast Suitable for:
a. real-time systems
b. hardware implementation
c. low-latency processing
d. No Channel Statistics Needed

LS does not need:
a. covariance matrices
b. channel models
c. correlation assumptions

Disadvantages
a. Sensitive to Noise
b. Noise directly corrupts:
    X[k]
    Y[k]
because no denoising is applied.
c. At low SNR, LS becomes unstable.
d. Interpolation Errors

Linear interpolation may fail if:
a. channel changes rapidly,
b. frequency selectivity is high.
c. No Statistical Optimization

LS treats each pilot independently. It ignores channel smoothness,correlation between neighboring subcarriers.

2. LMMSE(Linear Minimum Mean Square Error) Estimator improves LS by using:
    a. channel statistics,
    b. frequency correlation,
    c. noise information.
"Wireless channels are usually smooth in frequency". That means nearby subcarriers often experience similar fading.LMMSE exploits this property. Instead of trusting noisy LS estimates directly, it:
    a. smooths them,
    b. filters noise,
    c. uses correlation information.

LMMSE Equation: H^LMMSE=[Rhh(Rhh+I/SNR)−1] H^LS => This is the classical LMMSE estimator.
Meaning of Each Term "H^LS" Raw LS estimate at pilot positions.
Contains:
a. channel information
b. noise

Rhh => Channel correlation matrix.
Represents:
"How similar are channel responses between pilot subcarriers?"

Identity Matrix I Represents noise regularization. Helps stabilize inversion.

1/SNR Noise variance approximation. Higher SNR: trust LS estimates more.
Lower SNR: apply stronger smoothing.

Building the Correlation Matrix
code:
d = np.abs(np.subtract.outer(self.pilot_idx, self.pilot_idx))

Computes distance between pilots.
Example:
Pilot indices:
[0,2,4]

Distance matrix:
[[0,2,4],
 [2,0,2],
 [4,2,0]]

Then:
self._R_hh = np.exp(-d / coherence_bw) builds an exponential correlation model.

Meaning:
a. nearby pilots → highly correlated
b. distant pilots → weakly correlated

Coherence Bandwidth: Coherence bandwidth describes: how quickly the channel changes across frequency. Large coherence bandwidth:
a. smoother channel
b. neighboring subcarriers highly correlated
Small coherence bandwidth:
a. rapidly varying channel
b. frequency-selective fading

LMMSE Weight Matrix
Code:self._W = self._R_hh @ np.linalg.inv(
    self._R_hh + np.eye(self.n_pilots) / snr_lin
)
This computes a filtering matrix.

LMMSE Estimation Process
Step 1 — Compute LS Estimate
h_ls = rx_pilot_symbol[self.pilot_idx] / self.pilots

Step 2 — Apply Statistical Filtering
h_lmmse = self._W @ h_ls => This smooths noisy pilot estimates.

Step 3 — Interpolate
Same interpolation step as LS.

Why LMMSE Performs Better
LMMSE uses:
a. channel smoothness,
b. noise statistics,
c. pilot correlation.

This produces:
a. lower estimation error,
b. smoother channel response,
c. better equalization.

Especially at:
a. low SNR,
b. fading channels,
c. frequency-selective channels.

Computational Complexity
LS Complexity Very small: O(N)
Mostly:
a. divisions
b. interpolation

LMMSE Complexity
Higher complexity due to matrix inversion: O(N3)

for:
a. covariance matrix inversion.
In practical systems:
a. approximations,
b. low-rank methods,
c. FFT-based implementations
are often used.

Practical Interpretation
LS Estimator Thinks:
“I trust every pilot measurement directly.”

LMMSE Estimator Thinks:
“Pilot measurements are noisy, but neighboring subcarriers should behave similarly, so I’ll smooth intelligently.”

In Real 5G Systems
LS often used:
a. as initial estimate,
b. in simple receivers,
c. in low-complexity UE implementations.

LMMSE dsed in:
a. advanced receivers,
b. high-performance modems,
c. MIMO systems,
d. massive MIMO,
e. high-mobility scenarios.

Summary
LS Estimator
a. Direct pilot division
b. Simple and fast
c. Noise sensitive
d. Uses interpolation

LMMSE Estimator
a. Starts from LS
b. Uses channel correlation
c. Reduces noise statistically
d. More accurate but computationally heavier