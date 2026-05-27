"""
tput_analysis.py — Throughput Analysis & Visualisation
=======================================================
Explains and visualises the throughput vs MCS and throughput vs SNR curves.

Two representations:
  1. Peak TBS per MCS (spec-correct, inherently stepped)
  2. Shannon capacity bound (continuous reference curve)
  3. Effective throughput vs SNR with AMC (simulated)
  4. Per-MCS BLER waterfall (shows decoder operating region)

Run standalone:  python tput_analysis.py
"""

from __future__ import annotations
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from config.nr_tables  import get_tbs, get_mcs_params, MCS_TABLE
from config.sim_config  import SimConfig


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

N_PRB      = 52
MU         = 1          # 30 kHz SCS
SLOT_US    = 466.67     # μs per slot at μ=1
BW_HZ      = 20e6       # approximate 20 MHz bandwidth for spectral efficiency

# SNR → MCS threshold table (conservative, ~0% BLER target)
_MCS_THRESH = {
    0:-6.0, 1:-4.5, 2:-3.0, 3:-1.5, 4:0.0,  5:1.5,  6:3.0,  7:4.5,
    8:6.0,  9:7.5, 10:9.0, 11:10.5,12:12.0,13:13.5,14:15.0,15:16.5,
   16:18.0,17:19.5,18:21.0,19:22.5,20:24.0,21:25.5,22:27.0,23:28.5,
   24:30.0,25:31.5,26:33.0,27:34.5,28:36.0,
}

def select_mcs(snr_db: float) -> int:
    return max((m for m, t in _MCS_THRESH.items() if snr_db >= t), default=0)


# ─────────────────────────────────────────────────────────────────────────────
# Capacity and peak curves
# ─────────────────────────────────────────────────────────────────────────────

def shannon_capacity(snr_db: np.ndarray, bw_hz: float = BW_HZ) -> np.ndarray:
    """Shannon C = B * log2(1 + SNR) in Mbps."""
    snr_lin = 10 ** (snr_db / 10)
    return bw_hz * np.log2(1 + snr_lin) / 1e6


def amc_peak_curve(snr_db: np.ndarray) -> np.ndarray:
    """Peak TBS/slot for best MCS at each SNR (no BLER losses)."""
    return np.array([get_tbs(select_mcs(float(s)), N_PRB) / SLOT_US for s in snr_db])


def per_mcs_peak(n_prb: int = N_PRB) -> tuple[np.ndarray, np.ndarray]:
    """Return (mcs_indices, peak_throughput_mbps) for all 29 MCS."""
    mcs  = np.arange(29)
    peak = np.array([get_tbs(m, n_prb) / SLOT_US for m in mcs])
    return mcs, peak


# ─────────────────────────────────────────────────────────────────────────────
# Effective throughput with partial BLER
# Computed analytically: Tput_eff = TBS/slot * (1 - BLER_approx)
# BLER approximation: sigmoid transition centred at MCS threshold SNR
# ─────────────────────────────────────────────────────────────────────────────

def approx_bler(snr_db: float, mcs: int, width_db: float = 1.5) -> float:
    """
    Approximate BLER as a sigmoid centred at the MCS threshold.
    width_db controls steepness (real LDPC ~1 dB, conv codes ~3-5 dB).
    """
    thresh = _MCS_THRESH[mcs]
    x = (snr_db - thresh) / width_db
    return float(1 / (1 + np.exp(4 * x)))   # logistic: 0.98 at thresh-2, 0.02 at thresh+2


def effective_tput_curve(
    snr_db:   np.ndarray,
    channels: dict,          # {name: bler_offset_db}  — channel penalty shifts curve left
    width_db: float = 1.5,
) -> dict:
    """
    Compute effective throughput = TBS * (1 - BLER) for AMC selection.
    channel_offset shifts the effective SNR (models fading penalty).
    """
    results = {}
    for ch_name, snr_offset in channels.items():
        tput = np.zeros(len(snr_db))
        for i, snr in enumerate(snr_db):
            eff_snr = snr + snr_offset
            mcs     = select_mcs(float(eff_snr))
            tbs_mbps = get_tbs(mcs, N_PRB) / SLOT_US
            bler    = approx_bler(float(eff_snr), mcs, width_db)
            tput[i] = tbs_mbps * (1 - bler)
        results[ch_name] = tput
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_tput_analysis(save_path: str):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle('5G NR Throughput Analysis — 52 PRBs, μ=1, MCS 0–28',
                 fontsize=13, fontweight='bold')

    snr_db   = np.linspace(-8, 40, 500)
    snr_fine = np.arange(-8, 40, 0.1)

    # ── 1. Peak TBS per MCS (the staircase) ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mcs_idx, peak_mbps = per_mcs_peak()

    # Colour by modulation order
    qm_colors = {2: '#1f77b4', 4: '#ff7f0e', 6: '#2ca02c'}
    qm_labels = {2: 'QPSK', 4: '16-QAM', 6: '64-QAM'}
    plotted   = set()
    for m in mcs_idx:
        qm  = get_mcs_params(int(m))['Qm']
        col = qm_colors[qm]
        lbl = qm_labels[qm] if qm not in plotted else None
        ax1.bar(m, peak_mbps[m], color=col, alpha=0.8, edgecolor='white', width=0.7, label=lbl)
        plotted.add(qm)

    # Plateau lines
    for level, lbl in [(8.19,'8.2'), (17.55,'17.6'), (35.13,'35.1'),
                        (70.23,'70.2'), (140.52,'140.5')]:
        ax1.axhline(level, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax1.text(28.4, level+1, f'{lbl} Mbps', fontsize=7, color='gray', va='bottom')

    ax1.set_xlabel('MCS Index', fontsize=10)
    ax1.set_ylabel('Peak Throughput (Mbps)', fontsize=10)
    ax1.set_title('Peak TBS per MCS\n(spec-correct quantised steps)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-0.5, 28.5)
    ax1.grid(True, axis='y', alpha=0.3)

    # ── 2. Peak TBS vs SNR (AMC staircase) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    snr_step = np.arange(-8, 40, 0.2)
    peak_amc = amc_peak_curve(snr_step)
    cap      = shannon_capacity(snr_step)

    ax2.plot(snr_step, cap,      color='black', lw=2,   ls='--', label='Shannon capacity')
    ax2.step(snr_step, peak_amc, color='#1f77b4', lw=2, where='post', label='AMC peak (BLER=0)', alpha=0.9)
    ax2.fill_between(snr_step, peak_amc, color='#1f77b4', alpha=0.08, step='post')

    ax2.set_xlabel('SNR (dB)', fontsize=10)
    ax2.set_ylabel('Throughput (Mbps)', fontsize=10)
    ax2.set_title('Peak Throughput vs SNR\nStaircase = TBS quantisation', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-8, 40)
    ax2.set_ylim(0, cap.max() * 1.05)

    # ── 3. Effective throughput S-curves (with BLER model) ────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    channels = {
        'AWGN':      0.0,
        'Rayleigh': -3.0,
        'CDL-A':    -4.0,
        'CDL-B':    -6.0,
        'CDL-C':    -9.0,
    }
    ch_colors = {
        'AWGN':'#1f77b4','Rayleigh':'#ff7f0e',
        'CDL-A':'#2ca02c','CDL-B':'#d62728','CDL-C':'#9467bd'
    }
    eff = effective_tput_curve(snr_step, channels, width_db=1.5)

    ax3.plot(snr_step, shannon_capacity(snr_step), 'k--', lw=1.5, alpha=0.5,
             label='Shannon capacity')
    for ch, tput in eff.items():
        ax3.plot(snr_step, tput, color=ch_colors[ch], lw=2, label=ch)

    ax3.set_xlabel('SNR (dB)', fontsize=10)
    ax3.set_ylabel('Effective Throughput (Mbps)', fontsize=10)
    ax3.set_title('Effective Throughput = TBS×(1−BLER)\nGradual S-curve with BLER transitions', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-8, 40)
    ax3.set_ylim(0, shannon_capacity(np.array([40.0]))[0] * 1.05)

    # ── 4. BLER waterfall per MCS (4 key MCS shown) ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    snr_fine_w = np.linspace(-10, 42, 400)
    key_mcs = [0, 8, 16, 24]
    wf_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    for mcs_k, col in zip(key_mcs, wf_colors):
        bler = np.array([approx_bler(float(s), mcs_k) for s in snr_fine_w])
        p    = get_mcs_params(mcs_k)
        lbl  = f'MCS {mcs_k} ({p["modulation"]}, R={p["code_rate"]:.2f})'
        ax4.plot(snr_fine_w, bler, color=col, lw=2, label=lbl)

    ax4.axhline(0.1, color='black', lw=1.2, ls='--', alpha=0.7, label='10% BLER target')
    ax4.set_xlabel('SNR (dB)', fontsize=10)
    ax4.set_ylabel('BLER', fontsize=10)
    ax4.set_title('BLER Waterfall per MCS\n(sigmoid model, width=1.5 dB)', fontsize=10)
    ax4.legend(fontsize=7.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-10, 42)
    ax4.set_ylim(-0.02, 1.05)

    # ── 5. Spectral efficiency vs SNR ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    se_shannon = shannon_capacity(snr_step) * 1e6 / BW_HZ   # bits/s/Hz
    se_amc     = peak_amc * 1e6 / BW_HZ
    se_eff_awgn = eff['AWGN'] * 1e6 / BW_HZ

    ax5.plot(snr_step, se_shannon,   'k--', lw=2, alpha=0.7, label='Shannon limit')
    ax5.step(snr_step, se_amc,       color='#1f77b4', lw=2, where='post',
             alpha=0.7, label='AMC peak')
    ax5.plot(snr_step, se_eff_awgn,  color='#ff7f0e', lw=2, label='Effective (AWGN)')

    # MCS operating points
    for m in [0, 4, 8, 12, 16, 20, 24, 28]:
        p    = get_mcs_params(m)
        se   = p['spectral_efficiency']
        thresh = _MCS_THRESH[m]
        ax5.scatter([thresh], [se], s=40, color='#2ca02c', zorder=5)

    ax5.set_xlabel('SNR (dB)', fontsize=10)
    ax5.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=10)
    ax5.set_title('Spectral Efficiency vs SNR\nDots = MCS operating points', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-8, 40)

    # ── 6. Why the staircase exists — TBS quantisation explanation ────────────
    ax6 = fig.add_subplot(gs[1, 2])

    # Show N_info (continuous) vs TBS (quantised) for MCS 0-28
    n_re = min(156, 144) * N_PRB
    n_info_vals = np.array([
        n_re * (MCS_TABLE[m][1] / 1024) * MCS_TABLE[m][0]
        for m in range(29)
    ])
    tbs_vals = np.array([get_tbs(m, N_PRB) for m in range(29)])

    ax6.scatter(range(29), n_info_vals / 1000, color='#1f77b4', s=30,
                label='N_info (continuous)', zorder=3)
    ax6.step(range(29), tbs_vals / 1000, color='#d62728', lw=2, where='post',
             label='TBS (quantised)', alpha=0.9)

    ax6.set_xlabel('MCS Index', fontsize=10)
    ax6.set_ylabel('Bits (×10³)', fontsize=10)
    ax6.set_title('N_info vs TBS Quantisation\nSpec rounds to power-of-2 steps', fontsize=10)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.5, 28.5)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────

def print_tput_table():
    print(f'\n{"MCS":>4} | {"Mod":^8} | {"Rate":>6} | {"TBS":>8} | {"Peak(Mbps)":>10} | {"SNR thresh"}')
    print('─' * 62)
    for m in range(29):
        p    = get_mcs_params(m)
        tbs  = get_tbs(m, N_PRB)
        peak = tbs / SLOT_US
        snr  = _MCS_THRESH[m]
        print(f'{m:4d} | {p["modulation"]:^8} | {p["code_rate"]:6.4f} | {tbs:8d} | '
              f'{peak:10.2f} | {snr:+.1f} dB')


if __name__ == '__main__':
    OUT = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUT, exist_ok=True)

    print_tput_table()
    print('\nGenerating throughput analysis plot...')
    plot_tput_analysis(f'{OUT}/tput_analysis.png')
    print('Done.')