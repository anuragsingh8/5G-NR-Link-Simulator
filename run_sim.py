"""
run_sim.py — CLI Entry Point
=============================
Supports both CLI flags and YAML scenario files.

Usage
-----
  python run_sim.py                                     # defaults
  python run_sim.py --config scenarios/embb_cdla.yaml  # scenario file
  python run_sim.py --preset embb_high_tput             # named preset
  python run_sim.py --sweep-snr --channel TDL-A         # TDL sweep
  python run_sim.py --sweep-mcs --mcs-table qam256      # 256QAM MCS sweep
  python run_sim.py --sweep-bw                          # bandwidth sweep
  python run_sim.py --workers 4 --sweep-snr             # parallel
  python run_sim.py --help

Scenario YAML format
--------------------
  See scenarios/ directory for examples.
  Any SimConfig field can be a scalar (single run) or list (sweep).
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.sim_config import SimConfig
from config.nr_tables  import max_mcs
from sim.link_sim      import LinkSimulator, SimResult, run_parallel_sweep


# ─────────────────────────────────────────────────────────────────────────────
# YAML scenario loader
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(path: str) -> list[SimConfig]:
    """
    Load a YAML scenario file and return list of SimConfig objects.

    Scalar values → single run.
    List values under 'sweep' → one config per sweep point.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    sweep_key   = data.pop('sweep_param', None)
    sweep_values = data.pop('sweep_values', None)

    if sweep_key and sweep_values:
        configs = []
        for v in sweep_values:
            params = {**data, sweep_key: v}
            configs.append(SimConfig(**params))
        return configs
    else:
        return [SimConfig(**data)]


# ─────────────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='5G NR Link-Level Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Scenario / Preset ─────────────────────────────────────────────────────
    p.add_argument('--config', type=str, default=None,
                   help='YAML scenario file (overrides all other flags)')
    p.add_argument('--preset', type=str, default=None,
                   choices=[
                       'nr_fr1_5mhz', 'nr_fr1_10mhz', 'nr_fr1_20mhz',
                       'nr_fr1_100mhz', 'nr_fr1_100mhz_256qam',
                       'nr_fr2_100mhz', 'embb_high_tput', 'urllc', 'iiot_low_snr',
                   ],
                   help='Named configuration preset')

    # ── Numerology / BW ───────────────────────────────────────────────────────
    p.add_argument('--numerology', type=int, default=1, choices=[0,1,2,3,4])
    p.add_argument('--bw',         type=int, default=20, help='Bandwidth (MHz)')
    p.add_argument('--n-prb',      type=int, default=None)
    p.add_argument('--n-fft',      type=int, default=None)

    # ── Modulation ────────────────────────────────────────────────────────────
    p.add_argument('--mcs',       type=int, default=16)
    p.add_argument('--mcs-table', type=str, default='qam64',
                   choices=['qam64', 'qam256', 'qam64lr'])
    p.add_argument('--n-layers',  type=int, default=1)

    # ── Channel ───────────────────────────────────────────────────────────────
    p.add_argument('--channel',      type=str, default='CDL-A',
                   choices=['AWGN','Rayleigh',
                            'CDL-A','CDL-B','CDL-C',
                            'TDL-A','TDL-B','TDL-C','TDL-D','TDL-E'])
    p.add_argument('--snr',          type=float, default=15.0)
    p.add_argument('--velocity',     type=float, default=30.0)
    p.add_argument('--fc',           type=float, default=3.5, help='Carrier freq (GHz)')
    p.add_argument('--delay-spread', type=float, default=100.0, help='DS (ns) for TDL')

    # ── MIMO ──────────────────────────────────────────────────────────────────
    p.add_argument('--n-tx',       type=int,  default=4)
    p.add_argument('--n-rx',       type=int,  default=2)
    p.add_argument('--detector',   type=str,  default='MMSE', choices=['ZF','MMSE','SIC'])
    p.add_argument('--correlation',type=float,default=0.0)

    # ── HARQ ──────────────────────────────────────────────────────────────────
    p.add_argument('--harq-procs', type=int, default=8)
    p.add_argument('--max-harq',   type=int, default=4)
    p.add_argument('--harq-type',  type=str, default='IR', choices=['CC','IR'])

    # ── AMC ───────────────────────────────────────────────────────────────────
    p.add_argument('--target-bler', type=float, default=0.1)
    p.add_argument('--no-olla',     action='store_true')

    # ── Simulation ────────────────────────────────────────────────────────────
    p.add_argument('--slots',   type=int, default=500)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--quiet',   action='store_true')
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel workers for sweeps (default: 1 = serial)')

    # ── Sweeps ────────────────────────────────────────────────────────────────
    p.add_argument('--sweep-snr', action='store_true')
    p.add_argument('--snr-min',   type=float, default=-5)
    p.add_argument('--snr-max',   type=float, default=30)
    p.add_argument('--snr-step',  type=float, default=5)

    p.add_argument('--sweep-mcs', action='store_true')
    p.add_argument('--sweep-bw',  action='store_true')

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--output', type=str, default=None, help='Save JSON results')
    p.add_argument('--plot',   action='store_true',    help='Generate plots')

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

def args_to_config(args, snr_db=None, mcs=None, bw_mhz=None) -> SimConfig:
    if args.preset:
        overrides = {}
        if snr_db  is not None: overrides['snr_db']  = snr_db
        if mcs     is not None: overrides['mcs']      = mcs
        if bw_mhz  is not None: overrides['bw_mhz']  = bw_mhz
        return SimConfig.from_preset(args.preset, **overrides)

    return SimConfig(
        numerology          = args.numerology,
        bw_mhz              = bw_mhz or args.bw,
        n_prb               = args.n_prb,
        n_fft               = args.n_fft,
        mcs                 = mcs if mcs is not None else args.mcs,
        mcs_table           = args.mcs_table,
        n_layers            = args.n_layers,
        channel_model       = args.channel,
        snr_db              = snr_db if snr_db is not None else args.snr,
        velocity_kmh        = args.velocity,
        carrier_freq_ghz    = args.fc,
        delay_spread_ns     = args.delay_spread,
        n_tx                = args.n_tx,
        n_rx                = args.n_rx,
        detector            = args.detector,
        antenna_correlation = args.correlation,
        harq_processes      = args.harq_procs,
        max_harq_tx         = args.max_harq,
        harq_combining      = args.harq_type,
        target_bler         = args.target_bler,
        olla_enabled        = not args.no_olla,
        seed                = args.seed,
        n_slots             = args.slots,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def print_result(r: SimResult, label: str = ''):
    tag = f'  [{label}]' if label else ''
    print(f"\n{'─'*60}")
    print(f"{tag}  SNR={r.snr_db:+.1f}dB  MCS={r.avg_mcs:.0f}  "
          f"Channel={r.channel}  BW={r.bw_mhz}MHz")
    print(f"  BER            : {r.ber:.3e}")
    print(f"  BLER           : {r.bler:.4f}")
    print(f"  Throughput     : {r.throughput:.3f} Mbps")
    print(f"  Peak           : {r.peak_tput:.3f} Mbps  (BLER=0)")
    print(f"  Efficiency     : {r.throughput/max(r.peak_tput,1e-9)*100:.1f}% of peak")
    print(f"  Spectral eff.  : {r.spectral_eff:.4f} bits/s/Hz")
    print(f"  HARQ retx      : {r.n_harq_retx}/{r.n_harq_tx} "
          f"({r.n_harq_retx/max(r.n_harq_tx,1)*100:.1f}%)")
    print(f"  Slots          : {r.n_slots}")
    print(f"{'─'*60}")


def print_sweep_table(results: list[SimResult], sweep_var: str = 'SNR'):
    print(f"\n{'─'*75}")
    print(f"  {'Point':>12} {'BER':>10} {'BLER':>8} {'Tput(Mbps)':>12} "
          f"{'Peak(Mbps)':>12} {'Eff%':>7}")
    print(f"{'─'*75}")
    for r in results:
        if sweep_var == 'SNR':
            lbl = f"SNR={r.snr_db:+.1f}dB"
        elif sweep_var == 'MCS':
            lbl = f"MCS={r.avg_mcs:.0f}"
        else:
            lbl = f"BW={r.bw_mhz}MHz"
        eff = r.throughput / max(r.peak_tput, 1e-9) * 100
        print(f"  {lbl:>12} {r.ber:>10.3e} {r.bler:>8.4f} "
              f"{r.throughput:>12.3f} {r.peak_tput:>12.3f} {eff:>7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
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
    fig.suptitle(f'5G NR Simulator — {sweep_var} sweep  '
                 f'(ch={results[0].channel}, BW={results[0].bw_mhz}MHz)',
                 fontsize=12, fontweight='bold')

    x      = [r.snr_db for r in results] if sweep_var == 'SNR' \
             else [r.avg_mcs for r in results] if sweep_var == 'MCS' \
             else [r.bw_mhz for r in results]
    xlabel = 'SNR (dB)' if sweep_var == 'SNR' \
             else 'MCS Index' if sweep_var == 'MCS' else 'Bandwidth (MHz)'
    tput   = [r.throughput for r in results]
    peak   = [r.peak_tput  for r in results]
    bler   = [r.bler        for r in results]

    ax = axes[0]
    ax.plot(x, peak, 'k--', lw=1.5, alpha=0.5, label='Peak (BLER=0)')
    ax.plot(x, tput, 'o-',  lw=2, ms=5, color='#1f77b4', label='Effective')
    ax.fill_between(x, tput, alpha=0.12, color='#1f77b4')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Throughput', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(x, np.clip(bler, 1e-4, 1.1), 's-', lw=2, ms=5, color='#d62728')
    if sweep_var == 'SNR':
        ax2.axhline(0.1, color='k', ls='--', lw=1.2, alpha=0.6, label='10% target')
        ax2.legend(fontsize=9)
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel('BLER', fontsize=11)
    ax2.set_title('Block Error Rate', fontsize=11)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    np.random.seed(args.seed)

    print('=' * 60)
    print('  5G NR Link-Level Simulator')
    print('=' * 60)

    results:   list[SimResult] = []
    sweep_var: str             = 'SNR'

    # ── YAML scenario ─────────────────────────────────────────────────────────
    if args.config:
        print(f"\n  Scenario: {args.config}")
        configs = load_scenario(args.config)
        sweep_var = 'SNR' if len(configs) > 1 else 'single'
        t0 = time.time()
        if args.workers > 1 and len(configs) > 1:
            results = run_parallel_sweep(configs, args.slots, args.workers)
        else:
            for cfg in configs:
                sim = LinkSimulator(cfg)
                r   = sim.run(n_slots=cfg.n_slots, verbose=not args.quiet)
                results.append(r)
                if not args.quiet:
                    print_result(r)
        print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ── MCS sweep ─────────────────────────────────────────────────────────────
    elif args.sweep_mcs:
        sweep_var = 'MCS'
        n_mcs = max_mcs(args.mcs_table) + 1
        print(f"\n  MCS sweep: 0–{n_mcs-1}  table={args.mcs_table}  "
              f"SNR={args.snr}dB  channel={args.channel}")
        configs = [args_to_config(args, mcs=m) for m in range(n_mcs)]
        t0      = time.time()
        if args.workers > 1:
            results = run_parallel_sweep(configs, args.slots, args.workers)
        else:
            for m, cfg in enumerate(configs):
                sim = LinkSimulator(cfg)
                r   = sim.run(n_slots=args.slots, verbose=False)
                results.append(r)
                print(f"  MCS {m:2d}  {r.throughput:7.2f} Mbps  BLER={r.bler:.4f}")
        print(f"\n  Completed in {time.time()-t0:.1f}s")
        print_sweep_table(results, 'MCS')

    # ── BW sweep ──────────────────────────────────────────────────────────────
    elif args.sweep_bw:
        sweep_var = 'BW'
        from config.sim_config import _N_PRB_TABLE
        bw_list = sorted(_N_PRB_TABLE.get(args.numerology, {}).keys())
        print(f"\n  BW sweep: {bw_list} MHz  μ={args.numerology}")
        configs = [args_to_config(args, bw_mhz=bw) for bw in bw_list]
        t0      = time.time()
        if args.workers > 1:
            results = run_parallel_sweep(configs, args.slots, args.workers)
        else:
            for bw, cfg in zip(bw_list, configs):
                sim = LinkSimulator(cfg)
                r   = sim.run(n_slots=args.slots, verbose=False)
                results.append(r)
                print(f"  BW={bw:4d}MHz  {cfg.n_prb_eff:3d}PRBs  "
                      f"{r.throughput:7.2f} Mbps  BLER={r.bler:.4f}")
        print(f"\n  Completed in {time.time()-t0:.1f}s")
        print_sweep_table(results, 'BW')

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    elif args.sweep_snr:
        sweep_var = 'SNR'
        snr_pts   = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
        print(f"\n  SNR sweep: {snr_pts[0]:.0f}→{snr_pts[-1]:.0f}dB  "
              f"channel={args.channel}  MCS={args.mcs}  workers={args.workers}")
        configs   = [args_to_config(args, snr_db=float(s)) for s in snr_pts]
        t0        = time.time()
        if args.workers > 1:
            print(f"  Running {len(configs)} points in parallel ({args.workers} workers)...")
            results = run_parallel_sweep(configs, args.slots, args.workers)
        else:
            for snr, cfg in zip(snr_pts, configs):
                if not args.quiet:
                    print(f"\n▶ SNR = {snr:+.1f} dB")
                sim = LinkSimulator(cfg)
                r   = sim.run(n_slots=args.slots, verbose=not args.quiet)
                results.append(r)
                if args.quiet:
                    print(f"  SNR={snr:+.1f}dB  BLER={r.bler:.4f}  "
                          f"Tput={r.throughput:.2f}Mbps")
                else:
                    print_result(r)
        print(f"\n  Completed in {time.time()-t0:.1f}s")
        print_sweep_table(results, 'SNR')

    # ── Single point ──────────────────────────────────────────────────────────
    else:
        sweep_var = 'single'
        cfg = args_to_config(args)
        print(f"\n{cfg.summary()}")
        for w in cfg.validate_cp_vs_channel():
            print(f"\n  ⚠ {w}")
        print()
        t0  = time.time()
        sim = LinkSimulator(cfg)
        r   = sim.run(n_slots=args.slots, verbose=not args.quiet)
        dt  = time.time() - t0
        results.append(r)
        print_result(r)
        print(f"  Wall time: {dt:.1f}s")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\n  Results → {args.output}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot and len(results) > 1:
        plot_path = (args.output.replace('.json', '.png')
                     if args.output else 'results/plots/sweep.png')
        plot_results(results, sweep_var, plot_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
    