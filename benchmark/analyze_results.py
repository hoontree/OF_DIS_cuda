#!/usr/bin/env python3
"""
Analyze DIS Optical Flow Baseline Performance Results
Generates summary statistics and comparison tables
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# --- added: schema helpers ---
REQUIRED_COLUMNS = {
    "timestamp",
    "build_type",
    "image_pair",
    "resolution",
    "oppoint",
    "time_total_ms",
    "time_pyramid_ms",
    "time_oflow_ms",
    "time_save_ms",
}
TIME_COLUMNS = ["time_total_ms", "time_pyramid_ms", "time_oflow_ms", "time_save_ms"]

def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce time columns to numeric (robust against strings)
    for c in TIME_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic cleanup
    df = df.dropna(subset=["build_type", "oppoint", "time_total_ms"])
    df["build_type"] = df["build_type"].astype(str)
    df["oppoint"] = df["oppoint"].astype(str)
    df["resolution"] = df["resolution"].astype(str)
    df["image_pair"] = df["image_pair"].astype(str)
    return df

def analyze_baseline(csv_file):
    """Analyze baseline performance CSV"""

    if not Path(csv_file).exists():
        print(f"ERROR: {csv_file} not found")
        return

    df = pd.read_csv(csv_file)
    try:
        df = _validate_and_normalize(df)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    print("=" * 70)
    print("DIS Optical Flow CPU Baseline Performance Analysis")
    print("=" * 70)
    print()

    # Overall summary by build type
    print("1. Overall Performance by Build Configuration")
    print("-" * 70)
    summary = df.groupby('build_type').agg({
        'time_total_ms': ['mean', 'std', 'min', 'max', 'count'],
        'time_pyramid_ms': 'mean',
        'time_oflow_ms': 'mean',
        'time_save_ms': 'mean',
    }).round(2)
    print(summary)
    print()

    # Performance by operating point
    print("2. Performance by Operating Point")
    print("-" * 70)
    op_summary = df.groupby(['build_type', 'oppoint']).agg({
        'time_total_ms': ['mean', 'std'],
    }).round(2)
    print(op_summary)
    print()

    # Speedup analysis (multi-build, e.g., CPU-noMP/CPU-OMP/CUDA)
    builds = df['build_type'].unique()
    if len(builds) >= 2:
        print("3. Build Speedup Analysis (vs baseline)")
        print("-" * 70)

        # Pick a baseline build in a stable, meaningful order
        baseline_candidates = ["CPU-noMP", "CPU-OMP"]
        baseline_build = next((b for b in baseline_candidates if b in set(builds)), builds[0])

        # OpPoint별 평균 총시간 피벗
        times_pivot = (
            df.groupby(["oppoint", "build_type"])["time_total_ms"]
            .mean()
            .unstack("build_type")
        )

        # 출력 순서: baseline 먼저, 나머지는 이름순(재현성)
        ordered_builds = [baseline_build] + sorted([b for b in times_pivot.columns if b != baseline_build])

        for oppoint, row in times_pivot.iterrows():
            base = row.get(baseline_build, np.nan)
            if not (np.isfinite(base) and base > 0):
                continue

            parts = [f"  OpPoint {oppoint} | baseline={baseline_build} ({base:.2f} ms)"]
            for b in ordered_builds:
                t = row.get(b, np.nan)
                if np.isfinite(t) and t > 0:
                    parts.append(f"{b}: {t:.2f} ms ({base/t:.2f}x)")
            print("  " + " | ".join(parts))
        print()

        # 전체 평균(모든 OpPoint 합) 기준 speedup도 한 번 더 요약
        print("  Overall (mean of all rows):")
        overall = df.groupby("build_type")["time_total_ms"].mean()
        if baseline_build in overall.index and np.isfinite(overall.loc[baseline_build]) and overall.loc[baseline_build] > 0:
            base_all = float(overall.loc[baseline_build])
            for b in [baseline_build] + sorted([b for b in overall.index if b != baseline_build]):
                t = float(overall.loc[b])
                if np.isfinite(t) and t > 0:
                    print(f"    {b:12s}: {t:8.2f} ms  ({base_all/t:5.2f}x vs {baseline_build})")
        print()

    # Breakdown by resolution (provided in schema)
    print("4. Performance by Image Resolution")
    print("-" * 70)
    res_summary = df.groupby(['resolution', 'build_type']).agg({
        'time_total_ms': 'mean',
    }).round(2)
    print(res_summary)
    print()

    # Optional: breakdown by image pair (useful when pairs differ a lot)
    print("5. Performance by Image Pair (top 10 slowest mean total)")
    print("-" * 70)
    pair_summary = (
        df.groupby(['image_pair', 'build_type'])['time_total_ms']
        .mean()
        .sort_values(ascending=False)
        .groupby(level=1, group_keys=False)
        .head(10)
        .round(2)
    )
    print(pair_summary)
    print()

    # Stage timing breakdown
    print("6. Time Breakdown by Stage (% of total)")
    print("-" * 70)
    for build in builds:
        df_build = df[df['build_type'] == build]
        denom = df_build['time_total_ms'].replace(0, np.nan)
        pyramid_pct = (df_build['time_pyramid_ms'] / denom * 100).mean()
        oflow_pct = (df_build['time_oflow_ms'] / denom * 100).mean()
        save_pct = (df_build['time_save_ms'] / denom * 100).mean()
        print(f"  {build:12s}: Pyramid {pyramid_pct:5.1f}%, OFlow {oflow_pct:5.1f}%, Save {save_pct:5.1f}%")
    print()

    # Recommendations
    print("7. Optimization Opportunities")
    print("-" * 70)

    avg_pyramid = df['time_pyramid_ms'].mean()
    avg_oflow = df['time_oflow_ms'].mean()
    avg_save = df['time_save_ms'].mean()

    # Bottleneck hint (largest stage dominates)
    stage_avgs = {
        "pyramid": avg_pyramid,
        "oflow": avg_oflow,
        "save": avg_save,
    }
    primary = max(stage_avgs, key=stage_avgs.get)

    if primary == "oflow":
        print("  - OFlow computation is the primary bottleneck (CUDA/parallelism priority)")
    elif primary == "pyramid":
        print("  - Pyramid/Gradient generation is significant (consider CUDA / SIMD / threading)")
    else:
        print("  - Save/IO time is significant (batch writes, async IO, disable saving during benchmarks)")

    # CPU threading benefit: only when both CPU-noMP and CPU-OMP exist
    overall_means = df.groupby('build_type')['time_total_ms'].mean()
    if "CPU-noMP" in overall_means.index and "CPU-OMP" in overall_means.index:
        base = float(overall_means.loc["CPU-noMP"])
        fast = float(overall_means.loc["CPU-OMP"])
        if np.isfinite(base) and np.isfinite(fast) and fast > 0:
            speedup = base / fast
            if speedup < 1.5:
                print(f"  - Limited OpenMP speedup ({speedup:.2f}x) suggests memory-bound or serial sections")

    # CUDA benefit: compare against best available CPU baseline
    if "CUDA" in overall_means.index:
        cpu_baseline = None
        for b in ["CPU-OMP", "CPU-noMP"]:
            if b in overall_means.index:
                cpu_baseline = b
                break
        if cpu_baseline:
            cpu_t = float(overall_means.loc[cpu_baseline])
            cuda_t = float(overall_means.loc["CUDA"])
            if np.isfinite(cpu_t) and np.isfinite(cuda_t) and cuda_t > 0:
                print(f"  - CUDA speedup vs {cpu_baseline}: {cpu_t/cuda_t:.2f}x (lower is better ms/frame)")

    print("=" * 70)

def generate_latex_table(csv_file, output_file=None):
    """Generate LaTeX table for report"""
    df = pd.read_csv(csv_file)
    df = _validate_and_normalize(df)

    # Create summary table (total ms/frame)
    summary = df.groupby(['build_type', 'oppoint']).agg({
        'time_total_ms': 'mean',
    }).round(1).reset_index()

    pivot = summary.pivot(index='oppoint', columns='build_type', values='time_total_ms')

    latex = pivot.to_latex(caption="CPU Baseline Performance (ms/frame)",
                           label="tab:baseline")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to {output_file}")
    else:
        print("\nLaTeX Table:")
        print(latex)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <csv_file> [--latex output.tex]")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_baseline(csv_file)

    if "--latex" in sys.argv:
        idx = sys.argv.index("--latex")
        if idx + 1 < len(sys.argv):
            generate_latex_table(csv_file, sys.argv[idx + 1])
        else:
            generate_latex_table(csv_file)
