"""
run_sim.py — CLI Entry Point & Parameter Sweep
===============================================
Usage examples:
  python run_sim.py                                          # defaults
  python run_sim.py --snr 15 --channel CDL-A                # single point
  python run_sim.py --preset embb_high_tput                  # named preset
  python run_sim.py --sweep-snr --snr-min 0 --snr-max 30    # SNR sweep
  python run_sim.py --sweep-mcs --channel AWGN               # MCS sweep
  python run_sim.py --mcs-table qam256 --mcs 24              # 256-QAM
  python run_sim.py --bw 100 --numerology 1 --mcs 20         # 100 MHz
  python run_sim.py --plot --output results.json             # save + plot
  python run_sim.py --help
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.sim_config import SimConfig
from config.nr_tables  import max_mcs
from sim.link_sim      import LinkSimulator, SimResult


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='5G NR Link-Level Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Preset ────────────────────────────────────────────────────────────────
    p.add_argument('--preset', type=str, default=None,
                   choices=[
                       'nr_fr1_5mhz', 'nr_fr1_10mhz', 'nr_fr1_20mhz',
                       'nr_fr1_100mhz', 'nr_fr1_100mhz_256qam',
                       'nr_fr2_100mhz', 'embb_high_tput', 'urllc', 'iiot_low_snr',
                   ],
                   help='Named configuration preset (overrides individual flags)')

    # ── Numerology / BW ───────────────────────────────────────────────────────
    p.add_argument('--numerology', type=int, default=1, choices=[0,1,2,3,4],
                   help='μ: 0=15kHz, 1=30kHz, 2=60kHz, 3=120kHz, 4=240kHz')
    p.add_argument('--bw',         type=int, default=20,
                   help='Channel bandwidth in MHz (auto-selects n_prb)')
    p.add_argument('--n-prb',      type=int, default=None,
                   help='PRBs (overrides --bw auto-selection)')
    p.add_argument('--n-fft',      type=int, default=None,
                   help='FFT size (overrides auto-selection)')

    # ── Modulation ────────────────────────────────────────────────────────────
    p.add_argument('--mcs',       type=int,  default=16,
                   help='MCS index (0–28 for qam64; 0–27 for qam256)')
    p.add_argument('--mcs-table', type=str,  default='qam64',
                   choices=['qam64', 'qam256', 'qam64lr'],
                   help='MCS table: qam64 (default), qam256, qam64lr')
    p.add_argument('--n-layers',  type=int,  default=1,
                   help='Spatial layers (rank indicator)')

    # ── Channel ───────────────────────────────────────────────────────────────
    p.add_argument('--channel',   type=str,  default='CDL-A',
                   choices=['AWGN','Rayleigh',
                            'CDL-A','CDL-B','CDL-C',
                            'TDL-A','TDL-B','TDL-C','TDL-D','TDL-E'],
                   help='Channel model')
    p.add_argument('--snr',       type=float, default=15.0,  help='SNR in dB')
    p.add_argument('--velocity',  type=float, default=30.0,  help='UE velocity (km/h)')
    p.add_argument('--fc',        type=float, default=3.5,
                   help='Carrier frequency in GHz')
    p.add_argument('--delay-spread', type=float, default=100.0,
                   help='RMS delay spread in ns (for TDL models)')

    # ── MIMO ──────────────────────────────────────────────────────────────────
    p.add_argument('--n-tx',     type=int,  default=4,   help='TX antennas')
    p.add_argument('--n-rx',     type=int,  default=2,   help='RX antennas')
    p.add_argument('--detector', type=str,  default='MMSE',
                   choices=['ZF','MMSE','SIC'], help='Equaliser')
    p.add_argument('--correlation', type=float, default=0.0,
                   help='Antenna correlation coefficient (0–1)')

    # ── HARQ ──────────────────────────────────────────────────────────────────
    p.add_argument('--harq-procs', type=int, default=8,
                   help='HARQ processes (4 or 8)')
    p.add_argument('--max-harq',   type=int, default=4,
                   help='Max transmissions per TB (1=no HARQ)')
    p.add_argument('--harq-type',  type=str, default='IR',
                   choices=['CC','IR'], help='HARQ combining: Chase (CC) or IR')

    # ── AMC ───────────────────────────────────────────────────────────────────
    p.add_argument('--target-bler', type=float, default=0.1,
                   help='OLLA target BLER')
    p.add_argument('--no-olla',  action='store_true',
                   help='Disable outer-loop link adaptation')

    # ── Simulation control ────────────────────────────────────────────────────
    p.add_argument('--slots',     type=int,   default=500,  help='Slots per point')
    p.add_argument('--seed',      type=int,   default=42,   help='Random seed')
    p.add_argument('--quiet',     action='store_true',      help='Suppress slot output')

    # ── Sweeps ────────────────────────────────────────────────────────────────
    p.add_argument('--sweep-snr', action='store_true', help='Sweep SNR')
    p.add_argument('--snr-min',   type=float, default=-5,   help='Sweep SNR min (dB)')
    p.add_argument('--snr-max',   type=float, default=30,   help='Sweep SNR max (dB)')
    p.add_argument('--snr-step',  type=float, default=5,    help='Sweep SNR step (dB)')

    p.add_argument('--sweep-mcs', action='store_true',
                   help='Sweep all MCS indices for selected table')

    p.add_argument('--sweep-bw',  action='store_true',
                   help='Sweep bandwidths [5,10,20,50,100] MHz')

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument('--output', type=str, default=None, help='Save results to JSON')
    p.add_argument('--plot',   action='store_true',    help='Generate throughput plot')

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

def args_to_config(args, snr_db=None, mcs=None, bw_mhz=None) -> SimConfig:
    """Build SimConfig from parsed args with optional overrides."""
    if args.preset:
        overrides = {}
        if snr_db  is not None: overrides['snr_db']  = snr_db
        if mcs     is not None: overrides['mcs']      = mcs
        if bw_mhz  is not None: overrides['bw_mhz']  = bw_mhz
        return SimConfig.from_preset(args.preset, **overrides)

    return SimConfig(
        numerology         = args.numerology,
        bw_mhz             = bw_mhz or args.bw,
        n_prb              = args.n_prb,
        n_fft              = args.n_fft,
        mcs                = mcs if mcs is not None else args.mcs,
        mcs_table          = args.mcs_table,
        n_layers           = args.n_layers,
        channel_model      = args.channel,
        snr_db             = snr_db if snr_db is not None else args.snr,
        velocity_kmh       = args.velocity,
        carrier_freq_ghz   = args.fc,
        delay_spread_ns    = args.delay_spread,
        n_tx               = args.n_tx,
        n_rx               = args.n_rx,
        detector           = args.detector,
        antenna_correlation= args.correlation,
        harq_processes     = args.harq_procs,
        max_harq_tx        = args.max_harq,
        harq_combining     = args.harq_type,
        target_bler        = args.target_bler,
        olla_enabled       = not args.no_olla,
        seed               = args.seed,
        n_slots            = args.slots,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_result(r: SimResult, label: str = ''):
    tag = f'  [{label}]  ' if label else '  '
    print(f"\n{'─'*58}")
    print(f"{tag}SNR={r.snr_db:+.1f} dB  MCS={r.avg_mcs:.0f}  "
          f"Channel={r.channel}")
    print(f"  BER            : {r.ber:.3e}")
    print(f"  BLER           : {r.bler:.4f}")
    print(f"  Throughput     : {r.throughput:.3f} Mbps")
    print(f"  Peak throughput: {r.peak_tput:.3f} Mbps  (BLER=0)")
    print(f"  Efficiency     : {r.throughput/max(r.peak_tput,1e-9)*100:.1f}% of peak")
    print(f"  Spectral eff.  : {r.spectral_eff:.4f} bits/s/Hz")
    print(f"  HARQ retx      : {r.n_harq_retx} / {r.n_harq_tx}  "
          f"({r.n_harq_retx/max(r.n_harq_tx,1)*100:.1f}%)")
    print(f"  Slots simulated: {r.n_slots}")
    print(f"{'─'*58}")


def print_sweep_table(results: list[SimResult], sweep_var: str = 'SNR'):
    hdr = f"\n{'─'*72}\n"
    hdr += f"  {'Point':>12} {'BER':>10} {'BLER':>8} {'Tput(Mbps)':>12} "
    hdr += f"{'Peak(Mbps)':>12} {'Eff%':>7}\n"
    hdr += f"{'─'*72}"
    print(hdr)
    for r in results:
        label = (f"SNR={r.snr_db:+.1f}dB" if sweep_var == 'SNR'
                 else f"MCS={r.avg_mcs:.0f}" if sweep_var == 'MCS'
                 else f"BW={r.bw_mhz}MHz")
        eff = r.throughput / max(r.peak_tput, 1e-9) * 100
        print(f"  {label:>12} {r.ber:>10.3e} {r.bler:>8.4f} "
              f"{r.throughput:>12.3f} {r.peak_tput:>12.3f} {eff:>7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Optional plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list[SimResult], sweep_var: str, save_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'5G NR Simulator — {sweep_var} sweep', fontsize=12, fontweight='bold')

    if sweep_var == 'SNR':
        x     = [r.snr_db for r in results]
        xlabel = 'SNR (dB)'
    elif sweep_var == 'MCS':
        x     = [r.avg_mcs for r in results]
        xlabel = 'MCS Index'
    else:
        x     = [r.bw_mhz for r in results]
        xlabel = 'Bandwidth (MHz)'

    tput  = [r.throughput for r in results]
    bler  = [r.bler       for r in results]
    peak  = [r.peak_tput  for r in results]

    ax = axes[0]
    ax.plot(x, peak, 'k--', lw=1.5, alpha=0.5, label='Peak (BLER=0)')
    ax.plot(x, tput, 'o-',  lw=2,   ms=5, color='#1f77b4', label='Effective')
    ax.fill_between(x, tput, alpha=0.12, color='#1f77b4')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Throughput', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(x, np.clip(bler, 1e-4, 1), 's-', lw=2, ms=5, color='#d62728')
    if sweep_var == 'SNR':
        ax2.axhline(0.1, color='k', ls='--', lw=1.2, alpha=0.6, label='10% target')
        ax2.legend(fontsize=9)
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel('BLER', fontsize=11)
    ax2.set_title('Block Error Rate', fontsize=11)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_point(cfg: SimConfig, args, label: str = '') -> SimResult:
    sim = LinkSimulator(cfg)
    t0  = time.time()
    r   = sim.run(n_slots=cfg.n_slots, verbose=not args.quiet)
    dt  = time.time() - t0
    if not args.quiet:
        print(f"  ({dt:.1f}s)")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    np.random.seed(args.seed)

    print('=' * 58)
    print('  5G NR Link-Level Simulator')
    print('=' * 58)

    results:  list[SimResult] = []
    sweep_var = 'SNR'

    # ── Preset info ───────────────────────────────────────────────────────────
    if args.preset:
        print(f"\n  Preset: {args.preset}")

    # ── MCS sweep ─────────────────────────────────────────────────────────────
    if args.sweep_mcs:
        sweep_var = 'MCS'
        n_mcs = max_mcs(args.mcs_table) + 1
        print(f"\n  MCS sweep: 0–{n_mcs-1}  table={args.mcs_table}  "
              f"SNR={args.snr}dB  channel={args.channel}\n")
        for mcs in range(n_mcs):
            cfg = args_to_config(args, mcs=mcs)
            w   = cfg.validate_cp_vs_channel()
            if w:
                print(f"  ⚠ MCS {mcs:2d}: {w[0]}")
            if not args.quiet:
                print(f"\n▶ MCS {mcs}")
            r = run_point(cfg, args)
            results.append(r)
            print(f"  MCS {mcs:2d}  {r.throughput:7.2f} Mbps  BLER={r.bler:.4f}")
        print_sweep_table(results, 'MCS')

    # ── BW sweep ──────────────────────────────────────────────────────────────
    elif args.sweep_bw:
        sweep_var = 'BW'
        from config.sim_config import _N_PRB_TABLE
        bw_list = sorted(_N_PRB_TABLE.get(args.numerology, {}).keys())
        print(f"\n  BW sweep: {bw_list} MHz  μ={args.numerology}  "
              f"SNR={args.snr}dB\n")
        for bw in bw_list:
            cfg = args_to_config(args, bw_mhz=bw)
            if not args.quiet:
                print(f"\n▶ BW={bw} MHz  ({cfg.n_prb_eff} PRBs)")
            r = run_point(cfg, args)
            results.append(r)
            print(f"  BW={bw:4d}MHz  {cfg.n_prb_eff:3d}PRBs  "
                  f"{r.throughput:7.2f} Mbps  BLER={r.bler:.4f}")
        print_sweep_table(results, 'BW')

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    elif args.sweep_snr:
        sweep_var = 'SNR'
        snr_pts   = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
        cfg0      = args_to_config(args)
        print(f"\n  SNR sweep: {snr_pts[0]:.0f} → {snr_pts[-1]:.0f} dB  "
              f"(step={args.snr_step})  channel={args.channel}  "
              f"MCS={args.mcs}\n")
        for snr in snr_pts:
            cfg = args_to_config(args, snr_db=float(snr))
            if not args.quiet:
                print(f"\n▶ SNR = {snr:+.1f} dB")
            r = run_point(cfg, args)
            results.append(r)
            print_result(r)
        print_sweep_table(results, 'SNR')

    # ── Single point ──────────────────────────────────────────────────────────
    else:
        cfg = args_to_config(args)
        print(f"\n{cfg.summary()}")
        w = cfg.validate_cp_vs_channel()
        for warn in w:
            print(f"\n  ⚠ {warn}")
        print()
        r = run_point(cfg, args)
        results.append(r)
        print_result(r)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        out = []
        for r in results:
            d = {
                'snr_db':         r.snr_db,
                'channel':        r.channel,
                'mcs':            r.avg_mcs,
                'bw_mhz':         r.bw_mhz,
                'ber':            r.ber,
                'bler':           r.bler,
                'throughput_mbps':r.throughput,
                'peak_tput_mbps': r.peak_tput,
                'efficiency_pct': r.throughput / max(r.peak_tput, 1e-9) * 100,
                'spectral_eff':   r.spectral_eff,
                'n_harq_retx':    r.n_harq_retx,
                'n_slots':        r.n_slots,
            }
            out.append(d)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {args.output}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot and len(results) > 1:
        plot_path = (args.output.replace('.json', '.png')
                     if args.output else 'output/sweep_results.png')
        os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
        plot_results(results, sweep_var, plot_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
    