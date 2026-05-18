"""
run_sim.py — CLI Entry Point & Parameter Sweep
Usage:
  python run_sim.py                          # default config
  python run_sim.py --snr 10                 # single SNR point
  python run_sim.py --sweep-snr              # SNR sweep (-5 to 30 dB)
  python run_sim.py --channel CDL-A --slots 500
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
from sim.link_sim      import LinkSimulator, SimResult


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="5G NR Link-Level Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # OFDM
    p.add_argument('--numerology', type=int, default=1, choices=[0, 1, 2],
                   help='Subcarrier spacing: 0=15kHz, 1=30kHz, 2=60kHz')
    p.add_argument('--n-prb',     type=int, default=52,   help='Number of PRBs')
    p.add_argument('--n-fft',     type=int, default=2048, help='FFT size')

    # Modulation
    p.add_argument('--mcs',      type=int,   default=16,  help='MCS index (0-28)')
    p.add_argument('--n-layers', type=int,   default=1,   help='Spatial layers (RI)')

    # Channel
    p.add_argument('--channel',  type=str,   default='CDL-A',
                   choices=['AWGN', 'Rayleigh', 'CDL-A', 'CDL-B', 'CDL-C'],
                   help='Channel model')
    p.add_argument('--snr',      type=float, default=15.0, help='SNR in dB')
    p.add_argument('--velocity', type=float, default=30.0, help='UE velocity (km/h)')

    # MIMO
    p.add_argument('--n-tx',     type=int,  default=4,    help='TX antennas')
    p.add_argument('--n-rx',     type=int,  default=2,    help='RX antennas')
    p.add_argument('--detector', type=str,  default='MMSE',
                   choices=['ZF', 'MMSE', 'SIC'], help='Equaliser type')

    # Simulation control
    p.add_argument('--slots',     type=int,  default=500,  help='Slots per SNR point')
    p.add_argument('--seed',      type=int,  default=42,   help='Random seed')
    p.add_argument('--sweep-snr', action='store_true',     help='Run SNR sweep')
    p.add_argument('--snr-min',   type=float, default=-5,  help='Sweep SNR min (dB)')
    p.add_argument('--snr-max',   type=float, default=30,  help='Sweep SNR max (dB)')
    p.add_argument('--snr-step',  type=float, default=5,   help='Sweep SNR step (dB)')
    p.add_argument('--output',    type=str,   default=None, help='Save results to JSON')
    p.add_argument('--quiet',     action='store_true',     help='Suppress slot-level output')

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def args_to_config(args: argparse.Namespace, snr_db: float | None = None) -> SimConfig:
    return SimConfig(
        numerology    = args.numerology,
        n_prb         = args.n_prb,
        n_fft         = args.n_fft,
        mcs           = args.mcs,
        n_layers      = args.n_layers,
        channel_model = args.channel,
        snr_db        = snr_db if snr_db is not None else args.snr,
        velocity_kmh  = args.velocity,
        n_tx          = args.n_tx,
        n_rx          = args.n_rx,
        detector      = args.detector,
    )


def print_result(r: SimResult):
    print(f"\n{'─'*55}")
    print(f"  SNR            : {r.snr_db:+.1f} dB")
    print(f"  BER            : {r.ber:.3e}")
    print(f"  BLER           : {r.bler:.4f}")
    print(f"  Throughput     : {r.throughput:.3f} Mbps")
    print(f"  Avg MCS        : {r.avg_mcs:.1f}")
    print(f"  Slots simulated: {r.n_slots}")
    print(f"  Total HARQ Tx  : {r.n_harq_tx}")
    print(f"{'─'*55}")


def print_sweep_table(results: list[SimResult]):
    print(f"\n{'SNR(dB)':>8} {'BER':>12} {'BLER':>10} {'Tput(Mbps)':>12} {'AvgMCS':>8}")
    print("─" * 55)
    for r in results:
        print(f"{r.snr_db:>8.1f} {r.ber:>12.3e} {r.bler:>10.4f} "
              f"{r.throughput:>12.3f} {r.avg_mcs:>8.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 55)
    print("  5G NR Link-Level Simulator")
    print("=" * 55)

    results: list[SimResult] = []

    if args.sweep_snr:
        snr_points = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
        print(f"\n  SNR sweep: {snr_points[0]:.0f} → {snr_points[-1]:.0f} dB "
              f"(step={args.snr_step} dB, {len(snr_points)} points)")
        print(f"  Channel={args.channel}  MCS={args.mcs}  "
              f"Detector={args.detector}  Slots/point={args.slots}\n")

        for snr in snr_points:
            print(f"\n▶ SNR = {snr:+.1f} dB")
            cfg = args_to_config(args, snr_db=float(snr))
            sim = LinkSimulator(cfg)
            t0  = time.time()
            r   = sim.run(n_slots=args.slots, verbose=not args.quiet)
            dt  = time.time() - t0
            results.append(r)
            print_result(r)
            print(f"  (wall time: {dt:.1f}s)")

        print_sweep_table(results)

    else:
        print(f"\n  Single point: SNR={args.snr} dB  Channel={args.channel}  "
              f"MCS={args.mcs}  Detector={args.detector}")
        cfg = args_to_config(args)
        print(cfg.summary())
        sim = LinkSimulator(cfg)
        t0  = time.time()
        r   = sim.run(n_slots=args.slots, verbose=not args.quiet)
        dt  = time.time() - t0
        results.append(r)
        print_result(r)
        print(f"  Wall time: {dt:.1f}s")

    # Optionally save to JSON
    if args.output:
        out = [
            {
                "snr_db": r.snr_db, "ber": r.ber, "bler": r.bler,
                "throughput_mbps": r.throughput, "avg_mcs": r.avg_mcs,
                "n_slots": r.n_slots,
            }
            for r in results
        ]
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
