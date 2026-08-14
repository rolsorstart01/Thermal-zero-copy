#!/usr/bin/env python3
"""
ThermoCache Simulation Runner

This script runs a comparison between the baseline scheduler and ThermoCache,
demonstrating the benefits of thermal-aware scheduling with context deduplication.

Usage:
    python simulate.py [--requests NUM] [--sharing RATIO] [--gpu NUM_GPUS]
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime

from simulator.cluster import GPUClusterSimulator, SimulationRunner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_metrics(title: str, metrics):
    """Print system metrics in a formatted way."""
    print(f"\n📊 {title}")
    print("-" * 40)
    print(f"  Average Temperature:   {metrics.average_temperature:.2f}°C")
    print(f"  Peak Temperature:      {metrics.peak_temperature:.2f}°C")
    print(f"  VRAM Utilization:      {metrics.vram_utilization_percent:.1f}%")
    print(f"  Total VRAM Used:       {metrics.total_vram_used:.2f} GB")
    print(f"  Context Hit Rate:      {metrics.context_hit_rate * 100:.1f}%")
    print(f"  KV Cache Saved:        {metrics.kv_cache_memory_saved:.2f} GB")
    print(f"  Active GPUs:           {metrics.active_gpus}/{metrics.total_gpus}")
    print(f"  Thermal Hotspots:      {len(metrics.thermal_hotspots)} GPUs")
    if metrics.thermal_hotspots:
        print(f"    Hot GPU IDs:         {metrics.thermal_hotspots}")


def print_comparison(comparison: dict):
    """Print comparison results."""
    print(f"\n🏆 Improvement Summary (ThermoCache vs Baseline)")
    print("-" * 40)
    
    temp_reduction = comparison.get('temperature_reduction', 0)
    peak_reduction = comparison.get('peak_temperature_reduction', 0)
    vram_savings = comparison.get('vram_savings_percent', 0)
    reuse_rate = comparison.get('context_reuse_rate', 0)
    dist_improvement = comparison.get('thermal_distribution_improvement', 0)
    
    print(f"  Average Temp Reduction:     {temp_reduction:+.2f}°C")
    print(f"  Peak Temp Reduction:        {peak_reduction:+.2f}°C")
    print(f"  VRAM Savings:               {vram_savings:+.1f}%")
    print(f"  Context Reuse Rate:         {reuse_rate * 100:.1f}%")
    print(f"  Thermal Distribution Gain:  {dist_improvement:+.1f}%")
    
    # Qualitative assessment
    print("\n  Assessment:")
    if temp_reduction > 0:
        print(f"    ✓ Better thermal distribution")
    if peak_reduction > 2:
        print(f"    ✓ Significant peak temperature reduction")
    if vram_savings > 5:
        print(f"    ✓ Meaningful VRAM savings from deduplication")
    if reuse_rate > 0.5:
        print(f"    ✓ High context reuse efficiency")


async def main():
    parser = argparse.ArgumentParser(description='Run ThermoCache simulation')
    parser.add_argument('--requests', type=int, default=1000,
                        help='Number of requests to simulate')
    parser.add_argument('--sharing', type=float, default=0.7,
                        help='Context sharing ratio (0-1)')
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    
    args = parser.parse_args()
    
    print_separator("THERMOCACHE SIMULATION")
    print(f"\nConfiguration:")
    print(f"  GPUs:                 {args.gpus}")
    print(f"  Requests:             {args.requests}")
    print(f"  Context Sharing:      {args.sharing * 100:.0f}%")
    print(f"  Start Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize cluster and runner
    cluster = GPUClusterSimulator(num_gpus=args.gpus, num_racks=args.gpus // 2)
    runner = SimulationRunner(cluster, context_sharing_ratio=args.sharing, seed=42)
    
    # Run comparison
    print_separator("RUNNING SIMULATIONS")
    print("\nRunning baseline and ThermoCache simulations with identical workloads...")
    
    results = await runner.run_comparison(num_requests=args.requests)
    
    baseline = results['baseline']
    thermocache = results['thermocache']
    comparison = results['comparison']
    
    if args.json:
        # Output JSON for programmatic use
        output = {
            "configuration": {
                "num_gpus": args.gpus,
                "num_requests": args.requests,
                "context_sharing_ratio": args.sharing
            },
            "baseline": {
                "average_temperature": baseline.average_temperature,
                "peak_temperature": baseline.peak_temperature,
                "vram_utilization_percent": baseline.vram_utilization_percent,
                "total_vram_used": baseline.total_vram_used,
                "context_hit_rate": baseline.context_hit_rate
            },
            "thermocache": {
                "average_temperature": thermocache.average_temperature,
                "peak_temperature": thermocache.peak_temperature,
                "vram_utilization_percent": thermocache.vram_utilization_percent,
                "total_vram_used": thermocache.total_vram_used,
                "context_hit_rate": thermocache.context_hit_rate
            },
            "improvement": comparison
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print_metrics("BASELINE SCHEDULER", baseline)
        print_metrics("THERMOCACHE", thermocache)
        print_comparison(comparison)
        
        print_separator("SIMULATION COMPLETE")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nKey Insights:")
        print(f"  • ThermoCache reduces average temperature by {comparison.get('temperature_reduction', 0):.2f}°C")
        print(f"  • Peak temperatures are {comparison.get('peak_temperature_reduction', 0):.2f}°C lower")
        print(f"  • Context reuse rate: {comparison.get('context_reuse_rate', 0) * 100:.1f}%")
        print(f"\nThe prototype demonstrates that combining context deduplication")
        print(f"with thermal-aware scheduling can improve both memory efficiency")
        print(f"and thermal distribution across the GPU cluster.")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
