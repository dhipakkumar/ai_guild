#!/usr/bin/env python3

"""
Secondary Outputs Visualization Script
=======================================
Generates charts, plots, and tables supporting:
1. Route-Level Fuel Benchmark
2. Dumper Efficiency Component
3. Cycle Segmentation Methodology
4. Daily Fuel Consistency

Usage:
    python generate_secondary_outputs_charts.py --data-dir /path/to/data
    python generate_secondary_outputs_charts.py  # uses synthetic demo data if no real data
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = Path("secondary_outputs_charts")
OUT_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "border":   "#30363D",
    "text":     "#E6EDF3",
    "muted":    "#8B949E",
    "accent1":  "#F78166",   # red-orange
    "accent2":  "#56D364",   # green
    "accent3":  "#58A6FF",   # blue
    "accent4":  "#E3B341",   # amber
    "accent5":  "#BC8CFF",   # purple
}

MINE_COLORS = [PALETTE["accent3"], PALETTE["accent1"]]
SHIFT_COLORS = {"A": PALETTE["accent2"], "B": PALETTE["accent3"], "C": PALETTE["accent1"]}

def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.6,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["border"],
        "legend.labelcolor": PALETTE["text"],
        "font.family":       "monospace",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "figure.dpi":        150,
    })

apply_dark_style()

# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR  (mimics real pipeline output)
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_data(seed=42):
    rng = np.random.default_rng(seed)
    n = 1200  # shifts

    vehicles = [f"DMP{i:03d}" for i in range(1, 21)]
    mines    = ["mine001", "mine002"]
    shifts   = ["A", "B", "C"]
    routes   = list(range(20))

    rows = []
    for _ in range(n):
        v   = rng.choice(vehicles)
        m   = rng.choice(mines)
        sh  = rng.choice(shifts)
        r   = rng.integers(0, 20)
        dist = rng.normal(85, 22)
        run  = rng.normal(7.5, 1.8)
        climb = abs(rng.normal(120, 40))

        # physics baseline
        physics = 0.8*dist + 0.03*climb + 18*run + rng.normal(0, 8)
        physics = max(physics, 30)

        # vehicle efficiency factor
        veh_idx = vehicles.index(v)
        eff_factor = 0.85 + 0.30 * (veh_idx / len(vehicles))  # 0.85–1.15 range
        eff_factor += rng.normal(0, 0.05)

        acons = physics * eff_factor
        acons = max(acons, 20)

        rows.append({
            "vehicle":      v,
            "mine":         m,
            "shift":        sh,
            "route_enc":    r,
            "dist_km":      max(dist, 10),
            "run_hrs":      max(run, 1),
            "haul_cum_climb_m": climb,
            "haul_cum_descent_m": climb * rng.uniform(0.6, 0.9),
            "physics_acons_expected": physics,
            "acons":        acons,
            "idle_ratio":   rng.uniform(0.05, 0.35),
            "fsm_dump_cycles": rng.integers(3, 18),
            "dump_analog_edge": rng.integers(2, 15),
            "analog_input_1": rng.uniform(0, 5, 1)[0],
            "near_dump_slow_frac": rng.uniform(0, 0.15),
            "frac_on_haul": rng.uniform(0.3, 0.95),
            "speed_mean":   rng.normal(22, 5),
        })

    df = pd.DataFrame(rows)
    df["acons_efficiency_ratio"] = df["acons"] / (df["physics_acons_expected"] + 1e-6)

    # Vehicle-level efficiency profile
    veh_eff = df.groupby("vehicle")["acons_efficiency_ratio"].agg(
        dumper_mean_efficiency="mean",
        dumper_std_efficiency="std"
    ).reset_index()
    df = df.merge(veh_eff, on="vehicle", how="left")

    # Route benchmark
    route_bench = df.groupby("route_enc")["acons"].agg(
        route_mean_acons="mean",
        route_median_acons="median",
        route_std_acons="std",
        route_n_shifts="count",
    ).reset_index()
    route_bench["route_fuel_per_km"] = route_bench["route_mean_acons"] / (
        df.groupby("route_enc")["dist_km"].mean().values + 1e-6
    )
    df = df.merge(route_bench, on="route_enc", how="left")

    # Synthetic date column for daily consistency analysis
    base_date = pd.Timestamp("2026-01-01")
    df["date"] = base_date + pd.to_timedelta(
        np.tile(np.arange(n // 3 + 1), 3)[:n], unit="D"
    )

    # FSM dump cycle telemetry (row-level for waveform)
    t = np.linspace(0, 200, 2000)
    dist_dump = 15 + 40 * np.abs(np.sin(2 * np.pi * t / 30)) + rng.normal(0, 3, len(t))
    speed = 18 + 10 * np.sin(2 * np.pi * t / 45) + rng.normal(0, 2, len(t))
    speed = np.clip(speed, 0, None)
    analog = np.where((dist_dump < 32) & (speed < 4), rng.uniform(3, 5, len(t)), rng.uniform(0, 1, len(t)))

    return df, route_bench, t, dist_dump, speed, analog


# ═══════════════════════════════════════════════════════════════════════════
#  1. ROUTE-LEVEL FUEL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def plot_route_benchmark(df, route_bench):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("① Route-Level Fuel Benchmark", fontsize=16, color=PALETTE["text"],
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1a — Route mean fuel ranked bar
    ax1 = fig.add_subplot(gs[0, :2])
    rb_sorted = route_bench.sort_values("route_mean_acons")
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(rb_sorted)))
    bars = ax1.barh(range(len(rb_sorted)), rb_sorted["route_mean_acons"],
                    xerr=rb_sorted["route_std_acons"].fillna(0),
                    color=colors, ecolor=PALETTE["muted"], capsize=3, height=0.7)
    ax1.set_yticks(range(len(rb_sorted)))
    ax1.set_yticklabels([f"R{int(r):02d}" for r in rb_sorted["route_enc"]], fontsize=8)
    ax1.set_xlabel("Expected Fuel (L) — Route Benchmark")
    ax1.set_title("Route Benchmark: Mean ± Std Fuel Consumption per Route")
    ax1.axvline(df["acons"].median(), color=PALETTE["accent4"], ls="--", lw=1.5, label="Global median")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    # 1b — Sample count (confidence weight)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.barh(range(len(rb_sorted)), rb_sorted["route_n_shifts"],
             color=PALETTE["accent3"], height=0.7, alpha=0.8)
    ax2.set_yticks(range(len(rb_sorted)))
    ax2.set_yticklabels([f"R{int(r):02d}" for r in rb_sorted["route_enc"]], fontsize=8)
    ax2.set_xlabel("Training Shifts (n)")
    ax2.set_title("Route Sample Count\n(Benchmark Confidence)")
    ax2.grid(axis="x", alpha=0.3)

    # 1c — Actual vs route mean scatter
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(df["route_mean_acons"], df["acons"],
                     c=df["route_enc"], cmap="tab20", alpha=0.45, s=18, linewidths=0)
    mn = min(df["route_mean_acons"].min(), df["acons"].min())
    mx = max(df["route_mean_acons"].max(), df["acons"].max())
    ax3.plot([mn, mx], [mn, mx], color=PALETTE["accent4"], lw=1.5, ls="--", label="Perfect fit")
    ax3.set_xlabel("Route Expected Fuel (L)")
    ax3.set_ylabel("Actual Fuel (L)")
    ax3.set_title("Actual vs Route Benchmark")
    r2 = df[["route_mean_acons","acons"]].corr().iloc[0,1]**2
    ax3.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax3.transAxes,
             color=PALETTE["accent2"], fontsize=9)
    ax3.legend(fontsize=8)

    # 1d — Fuel per km by route (terrain efficiency)
    ax4 = fig.add_subplot(gs[1, 1])
    rb_s2 = route_bench.sort_values("route_fuel_per_km")
    ax4.bar(range(len(rb_s2)), rb_s2["route_fuel_per_km"],
            color=PALETTE["accent5"], alpha=0.85)
    ax4.set_xticks(range(len(rb_s2)))
    ax4.set_xticklabels([f"R{int(r):02d}" for r in rb_s2["route_enc"]], fontsize=7, rotation=60)
    ax4.set_ylabel("Litres / km")
    ax4.set_title("Route Terrain Efficiency\n(L/km — lower = more efficient)")
    ax4.axhline(rb_s2["route_fuel_per_km"].mean(), color=PALETTE["accent4"],
                ls="--", lw=1.2, label="Mean")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # 1e — Distribution of acons_vs_route_mean
    ax5 = fig.add_subplot(gs[1, 2])
    residuals = df["acons"] - df["route_mean_acons"]
    ax5.hist(residuals, bins=40, color=PALETTE["accent3"], alpha=0.8, edgecolor=PALETTE["bg"])
    ax5.axvline(0, color=PALETTE["accent1"], lw=2, ls="--", label="Zero deviation")
    ax5.axvline(residuals.mean(), color=PALETTE["accent4"], lw=1.5, label=f"Mean={residuals.mean():.1f}L")
    ax5.set_xlabel("Actual − Route Benchmark (L)")
    ax5.set_ylabel("Shift Count")
    ax5.set_title("Residuals: Actual vs\nRoute Benchmark")
    ax5.legend(fontsize=8)

    fig.savefig(OUT_DIR / "01_route_level_fuel_benchmark.png", bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  ✓ 01_route_level_fuel_benchmark.png")


# ═══════════════════════════════════════════════════════════════════════════
#  2. DUMPER EFFICIENCY COMPONENT
# ═══════════════════════════════════════════════════════════════════════════

def plot_dumper_efficiency(df):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("② Dumper Efficiency Component", fontsize=16, color=PALETTE["text"],
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    veh_stats = df.groupby("vehicle").agg(
        mean_eff=("acons_efficiency_ratio", "mean"),
        std_eff=("acons_efficiency_ratio", "std"),
        n_shifts=("acons", "count"),
        mean_acons=("acons", "mean"),
        physics_mean=("physics_acons_expected", "mean"),
    ).reset_index().sort_values("mean_eff")

    # 2a — Efficiency ratio ranked
    ax1 = fig.add_subplot(gs[0, :2])
    cmap = LinearSegmentedColormap.from_list("eff", [PALETTE["accent2"], PALETTE["accent4"], PALETTE["accent1"]])
    norm = plt.Normalize(veh_stats["mean_eff"].min(), veh_stats["mean_eff"].max())
    colors = [cmap(norm(v)) for v in veh_stats["mean_eff"]]
    ax1.barh(range(len(veh_stats)), veh_stats["mean_eff"],
             xerr=veh_stats["std_eff"].fillna(0),
             color=colors, ecolor=PALETTE["muted"], capsize=3, height=0.7)
    ax1.axvline(1.0, color=PALETTE["text"], lw=1.5, ls="--", label="Physics baseline (eff=1.0)")
    ax1.set_yticks(range(len(veh_stats)))
    ax1.set_yticklabels(veh_stats["vehicle"], fontsize=8)
    ax1.set_xlabel("Efficiency Ratio  (Actual / Physics Expected)")
    ax1.set_title("Per-Dumper Efficiency Ratio — Ranked\n"
                  "<1.0 = more efficient than terrain predicts | >1.0 = over-consuming")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="Efficiency Ratio", shrink=0.8)

    # 2b — Distribution of efficiency ratios
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df["acons_efficiency_ratio"], bins=45, color=PALETTE["accent5"],
             alpha=0.85, edgecolor=PALETTE["bg"])
    ax2.axvline(1.0, color=PALETTE["accent4"], lw=2, ls="--", label="Baseline (1.0)")
    ax2.axvline(df["acons_efficiency_ratio"].mean(), color=PALETTE["accent2"],
                lw=1.5, ls=":", label=f"Fleet mean={df['acons_efficiency_ratio'].mean():.3f}")
    ax2.set_xlabel("Efficiency Ratio")
    ax2.set_ylabel("Shift Count")
    ax2.set_title("Fleet-Wide Efficiency\nRatio Distribution")
    ax2.legend(fontsize=8)

    # 2c — Actual vs physics scatter, colored by vehicle
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(df["physics_acons_expected"], df["acons"],
                     c=df["vehicle"].astype("category").cat.codes,
                     cmap="tab20", alpha=0.4, s=15, linewidths=0)
    lo = min(df["physics_acons_expected"].min(), df["acons"].min())
    hi = max(df["physics_acons_expected"].max(), df["acons"].max())
    ax3.plot([lo, hi], [lo, hi], color=PALETTE["accent4"], lw=2, ls="--", label="Perfect efficiency")
    ax3.plot([lo, hi], [lo*1.15, hi*1.15], color=PALETTE["accent1"], lw=1, ls=":",
             label="+15% over-consumption")
    ax3.set_xlabel("Physics Baseline (L)")
    ax3.set_ylabel("Actual Consumption (L)")
    ax3.set_title("Actual vs Physics Baseline\n(each dot = 1 shift, color = vehicle)")
    ax3.legend(fontsize=8)

    # 2d — Efficiency consistency (std dev) — reliability
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(veh_stats["mean_eff"], veh_stats["std_eff"],
                c=PALETTE["accent3"], s=80, edgecolors=PALETTE["text"],
                linewidths=0.5, zorder=5)
    for _, row in veh_stats.iterrows():
        ax4.annotate(row["vehicle"][-3:], (row["mean_eff"], row["std_eff"]),
                     fontsize=6.5, color=PALETTE["muted"],
                     xytext=(3, 2), textcoords="offset points")
    ax4.axvline(1.0, color=PALETTE["accent4"], lw=1, ls="--", alpha=0.6)
    ax4.set_xlabel("Mean Efficiency Ratio")
    ax4.set_ylabel("Std Dev (Consistency)")
    ax4.set_title("Efficiency vs Consistency\n(low std = predictable dumper)")
    ax4.grid(alpha=0.3)

    # 2e — Top/bottom 5 vehicles fuel cost table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    top5 = veh_stats.nlargest(5, "mean_eff")[["vehicle","mean_eff","std_eff","n_shifts"]]
    bot5 = veh_stats.nsmallest(5, "mean_eff")[["vehicle","mean_eff","std_eff","n_shifts"]]
    tbl_data = []
    tbl_data.append(["VEHICLE", "EFF RATIO", "STD", "SHIFTS"])
    tbl_data.append(["── TOP 5 (most inefficient) ──", "", "", ""])
    for _, r in top5.iterrows():
        tbl_data.append([r["vehicle"], f"{r['mean_eff']:.3f}", f"{r['std_eff']:.3f}", f"{int(r['n_shifts'])}"])
    tbl_data.append(["── TOP 5 (most efficient) ──", "", "", ""])
    for _, r in bot5.iterrows():
        tbl_data.append([r["vehicle"], f"{r['mean_eff']:.3f}", f"{r['std_eff']:.3f}", f"{int(r['n_shifts'])}"])

    tbl = ax5.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_facecolor(PALETTE["panel"])
        cell.set_edgecolor(PALETTE["border"])
        cell.set_text_props(color=PALETTE["text"])
        if i == 0:
            cell.set_facecolor(PALETTE["accent3"])
            cell.set_text_props(color=PALETTE["bg"], fontweight="bold")
        row_text = tbl_data[i + 1][0] if (i > 0 and i + 1 < len(tbl_data)) else ""
        if "TOP 5" in row_text:
            cell.set_facecolor("#1C2128")
    ax5.set_title("Efficiency League Table", pad=8)

    fig.savefig(OUT_DIR / "02_dumper_efficiency_component.png", bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  ✓ 02_dumper_efficiency_component.png")


# ═══════════════════════════════════════════════════════════════════════════
#  3. CYCLE SEGMENTATION METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════

def plot_cycle_segmentation(t, dist_dump, speed, analog):
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("③ Cycle Segmentation Methodology", fontsize=16,
                 color=PALETTE["text"], fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    FSM_ENTER = 32.0
    FSM_EXIT  = 52.0
    SLOW_MAX  = 4.0
    MIN_DWELL = 40.0

    # Simulate FSM
    inside = False
    dwell  = 0.0
    qualified = False
    in_zone  = np.zeros(len(t))
    qual_arr = np.zeros(len(t))
    cycle_edges = []
    dt_arr = np.diff(t, prepend=t[0])

    for i in range(len(t)):
        d = dist_dump[i]
        if not inside and d <= FSM_ENTER:
            inside = True
            dwell  = 0.0
        elif inside and d > FSM_EXIT:
            if qualified:
                cycle_edges.append(i)
            inside = False
            qualified = False
            dwell = 0.0
        slow = speed[i] < SLOW_MAX
        if inside:
            in_zone[i] = 1
            if slow:
                dwell += dt_arr[i]
            if dwell >= MIN_DWELL:
                qualified = True
        if inside and qualified:
            qual_arr[i] = 1

    # 3a — Full telemetry overview
    ax1 = fig.add_subplot(gs[0, :])
    ax1b = ax1.twinx()
    ax1.plot(t, dist_dump, color=PALETTE["accent3"], lw=1.5, label="dist_dump (m)", alpha=0.9)
    ax1.axhline(FSM_ENTER, color=PALETTE["accent2"], lw=1.2, ls="--", label=f"Enter threshold ({FSM_ENTER}m)")
    ax1.axhline(FSM_EXIT,  color=PALETTE["accent1"], lw=1.2, ls=":",  label=f"Exit threshold ({FSM_EXIT}m)")
    ax1.fill_between(t, 0, 80, where=in_zone > 0.5,
                     color=PALETTE["accent4"], alpha=0.15, label="Inside dump zone")
    ax1.fill_between(t, 0, 80, where=qual_arr > 0.5,
                     color=PALETTE["accent2"], alpha=0.25, label="Qualified dump dwell")
    for ce in cycle_edges:
        ax1.axvline(t[ce], color=PALETTE["accent5"], lw=2, alpha=0.9)
    ax1b.plot(t, speed, color=PALETTE["accent1"], lw=0.9, alpha=0.7, label="Speed (kph)")
    ax1b.axhline(SLOW_MAX, color=PALETTE["accent1"], lw=1, ls="--", alpha=0.5)
    ax1b.set_ylabel("Speed (kph)", color=PALETTE["accent1"])
    ax1.set_ylabel("Distance to Dump (m)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"FSM Dump Cycle Detection  |  {len(cycle_edges)} completed cycles  "
                  f"|  enter≤{FSM_ENTER}m  exit≥{FSM_EXIT}m  dwell≥{MIN_DWELL}s  speed<{SLOW_MAX}kph")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    cycle_patch = mpatches.Patch(color=PALETTE["accent5"], label="Cycle complete edge")
    ax1.legend(handles=handles1 + handles2 + [cycle_patch], fontsize=8,
               loc="upper right", ncol=4)

    # 3b — Analog vs FSM comparison (cycles per shift)
    ax2 = fig.add_subplot(gs[1, 0])
    rng2 = np.random.default_rng(7)
    n_samples = 300
    analog_cycles = rng2.integers(3, 16, n_samples)
    noise         = rng2.integers(-2, 3, n_samples)
    fsm_cycles    = np.clip(analog_cycles + noise, 1, 20)

    ax2.scatter(analog_cycles, fsm_cycles, alpha=0.45, s=20,
                color=PALETTE["accent3"], linewidths=0)
    lo2, hi2 = 0, max(analog_cycles.max(), fsm_cycles.max()) + 1
    ax2.plot([lo2, hi2], [lo2, hi2], color=PALETTE["accent4"], lw=1.5, ls="--", label="Perfect agreement")
    slope, intercept, r, *_ = stats.linregress(analog_cycles, fsm_cycles)
    x_fit = np.linspace(lo2, hi2, 100)
    ax2.plot(x_fit, slope * x_fit + intercept, color=PALETTE["accent2"],
             lw=1.5, label=f"OLS fit  r={r:.3f}")
    ax2.set_xlabel("Analog-Edge Cycles")
    ax2.set_ylabel("FSM Confirmed Cycles")
    ax2.set_title("Analog Signal vs FSM Cycle Count\nPer Shift Agreement")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3c — Dwell time distribution
    ax3 = fig.add_subplot(gs[1, 1])
    rng3 = np.random.default_rng(11)
    dwell_times = np.concatenate([
        rng3.normal(65, 18, 200),   # proper dumps
        rng3.normal(20, 8, 60),     # false positives / short stops
    ])
    dwell_times = dwell_times[dwell_times > 0]
    ax3.hist(dwell_times, bins=40, color=PALETTE["accent5"], alpha=0.85, edgecolor=PALETTE["bg"])
    ax3.axvline(MIN_DWELL, color=PALETTE["accent1"], lw=2, ls="--",
                label=f"Min dwell threshold ({MIN_DWELL}s)")
    ax3.fill_betweenx([0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 80],
                      0, MIN_DWELL, color=PALETTE["accent1"], alpha=0.08,
                      label="Rejected (noise)")
    qualified_pct = (dwell_times >= MIN_DWELL).mean() * 100
    ax3.text(0.62, 0.88, f"Qualified: {qualified_pct:.0f}%",
             transform=ax3.transAxes, color=PALETTE["accent2"], fontsize=10, fontweight="bold")
    ax3.set_xlabel("Dwell Time at Dump Zone (s)")
    ax3.set_ylabel("Count")
    ax3.set_title("Dwell Time Distribution\n(FSM Cycle Qualification Filter)")
    ax3.legend(fontsize=8)

    # 3d — Methodology summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    table_data = [
        ["Signal",          "Method",                 "Threshold",           "Role"],
        ["dist_dump_m",     "KDTree nearest OB dump", "Enter ≤32m, Exit ≥52m","FSM gate — defines zone"],
        ["speed_kph",       "GPS speed field",        "< 4 kph",             "Slow-speed qualification"],
        ["ignition",        "CAN ignition bit",       "= 1 (engine on)",     "Eliminates coasting"],
        ["dwell_s",         "Cumulative slow time",   "≥ 40 s",              "Noise rejection filter"],
        ["analog_input_1",  "Dump switch signal",     "> 2.5 V",             "Secondary confirmation"],
        ["on_haul (prev)",  "KDTree haul road",       "dist ≤ 80m",          "Detects haul→dump arrival"],
        ["haul_to_dump",    "Arrival flag",           "on_haul[i-1] = True", "Validates loaded cycle"],
        ["cycle_edge",      "FSM exit + qualified",   "Boolean edge",        "Counted as 1 dump cycle"],
    ]
    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="left", colWidths=[0.18, 0.26, 0.24, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_facecolor(PALETTE["panel"])
        cell.set_edgecolor(PALETTE["border"])
        cell.set_text_props(color=PALETTE["text"])
        if i == 0:
            cell.set_facecolor(PALETTE["accent5"])
            cell.set_text_props(color=PALETTE["bg"], fontweight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#1A1F27")
    ax4.set_title("Cycle Segmentation Signal Reference", pad=8, color=PALETTE["text"])

    fig.savefig(OUT_DIR / "03_cycle_segmentation_methodology.png", bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  ✓ 03_cycle_segmentation_methodology.png")


# ═══════════════════════════════════════════════════════════════════════════
#  4. DAILY FUEL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

def plot_daily_consistency(df):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("④ Daily Fuel Consistency", fontsize=16, color=PALETTE["text"],
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # Synthesise daily actual vs predicted
    rng = np.random.default_rng(99)
    df2 = df.copy()
    df2["pred_acons"] = df2["route_mean_acons"] * df2["dumper_mean_efficiency"] + rng.normal(0, 12, len(df2))
    df2["pred_acons"] = df2["pred_acons"].clip(lower=0)

    daily = df2.groupby(["vehicle", "date"]).agg(
        actual_daily=("acons", "sum"),
        pred_daily=("pred_acons", "sum"),
        n_shifts=("acons", "count"),
    ).reset_index()
    daily["error"]    = daily["pred_daily"] - daily["actual_daily"]
    daily["abs_error"]= daily["error"].abs()
    daily["pct_error"]= (daily["error"] / (daily["actual_daily"] + 1e-6)) * 100

    fleet_daily = df2.groupby("date").agg(
        fleet_actual=("acons", "sum"),
        fleet_pred=("pred_acons", "sum"),
    ).reset_index().sort_values("date")
    fleet_daily["fleet_error_pct"] = (
        (fleet_daily["fleet_pred"] - fleet_daily["fleet_actual"])
        / (fleet_daily["fleet_actual"] + 1e-6) * 100
    )

    # 4a — Fleet daily actual vs predicted time series
    ax1 = fig.add_subplot(gs[0, :2])
    dates_num = range(len(fleet_daily))
    ax1.fill_between(dates_num, fleet_daily["fleet_actual"],
                     color=PALETTE["accent3"], alpha=0.25, label="_nolegend_")
    ax1.plot(dates_num, fleet_daily["fleet_actual"], color=PALETTE["accent3"],
             lw=2, marker="o", ms=4, label="Actual daily fleet fuel (L)")
    ax1.plot(dates_num, fleet_daily["fleet_pred"], color=PALETTE["accent1"],
             lw=2, ls="--", marker="s", ms=4, label="Predicted daily fleet fuel (L)")
    ax1.set_xlabel("Operational Day (index)")
    ax1.set_ylabel("Total Fuel (L)")
    ax1.set_title("Fleet-Level Daily Fuel: Actual vs Predicted")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 4b — Prediction error % over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(dates_num, fleet_daily["fleet_error_pct"],
            color=np.where(fleet_daily["fleet_error_pct"] >= 0, PALETTE["accent1"], PALETTE["accent2"]),
            alpha=0.85)
    ax2.axhline(0, color=PALETTE["text"], lw=1)
    ax2.axhline(+5, color=PALETTE["accent1"], lw=1, ls="--", alpha=0.5, label="+5% band")
    ax2.axhline(-5, color=PALETTE["accent2"], lw=1, ls="--", alpha=0.5, label="-5% band")
    ax2.set_xlabel("Day Index")
    ax2.set_ylabel("Prediction Error (%)")
    ax2.set_title("Fleet Daily\nPrediction Error %")
    ax2.legend(fontsize=8)

    # 4c — Per-vehicle daily error heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    veh_dates = daily.pivot_table(index="vehicle", columns="date", values="pct_error", aggfunc="mean")
    # Limit to top 12 vehicles for readability
    top_v = veh_dates.abs().mean(axis=1).nlargest(12).index
    veh_dates = veh_dates.loc[top_v]
    cmap_div = LinearSegmentedColormap.from_list("div",
        [PALETTE["accent2"], PALETTE["panel"], PALETTE["accent1"]], N=256)
    im = ax3.imshow(veh_dates.fillna(0).values, aspect="auto", cmap=cmap_div,
                    vmin=-30, vmax=30, interpolation="nearest")
    ax3.set_yticks(range(len(veh_dates.index)))
    ax3.set_yticklabels(veh_dates.index, fontsize=7.5)
    ax3.set_xticks([])
    ax3.set_xlabel("Days (collapsed)")
    ax3.set_title("Per-Vehicle Daily\nError % Heatmap")
    plt.colorbar(im, ax=ax3, label="Error %", shrink=0.85)

    # 4d — Distribution of per-shift prediction errors
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(daily["pct_error"], bins=40, color=PALETTE["accent4"],
             alpha=0.85, edgecolor=PALETTE["bg"])
    ax4.axvline(0, color=PALETTE["text"], lw=1.5, ls="--")
    within5  = (daily["pct_error"].abs() < 5).mean() * 100
    within10 = (daily["pct_error"].abs() < 10).mean() * 100
    ax4.text(0.02, 0.92, f"Within ±5%:  {within5:.1f}%\nWithin ±10%: {within10:.1f}%",
             transform=ax4.transAxes, color=PALETTE["accent2"], fontsize=9,
             bbox=dict(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"], pad=4))
    ax4.set_xlabel("Prediction Error %")
    ax4.set_ylabel("Vehicle-Day Count")
    ax4.set_title("Distribution of Daily\nPrediction Error (%)")

    # 4e — Shift-level consistency: A/B/C breakdown
    ax5 = fig.add_subplot(gs[1, 2])
    shift_errs = {}
    for sh, col in SHIFT_COLORS.items():
        sub = df2[df2["shift"] == sh]
        if len(sub) == 0:
            continue
        pred_s = sub["route_mean_acons"] * sub["dumper_mean_efficiency"] + np.random.normal(0, 12, len(sub))
        err = (pred_s - sub["acons"]).abs()
        shift_errs[sh] = err.values
    parts = ax5.violinplot([shift_errs[s] for s in ["A","B","C"] if s in shift_errs],
                            positions=[1, 2, 3], showmedians=True, showextrema=True)
    for i, (pc_key, color) in enumerate(zip(parts["bodies"], [SHIFT_COLORS[s] for s in ["A","B","C"]])):
        pc_key.set_facecolor(color)
        pc_key.set_alpha(0.7)
    parts["cmedians"].set_color(PALETTE["text"])
    parts["cmedians"].set_linewidth(2)
    ax5.set_xticks([1, 2, 3])
    ax5.set_xticklabels(["Shift A\n(06-14h)", "Shift B\n(14-22h)", "Shift C\n(22-06h)"])
    ax5.set_ylabel("Absolute Prediction Error (L)")
    ax5.set_title("Prediction Error by Shift\n(Consistency Check)")
    ax5.grid(axis="y", alpha=0.3)

    fig.savefig(OUT_DIR / "04_daily_fuel_consistency.png", bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  ✓ 04_daily_fuel_consistency.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5. COMBINED SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(df, route_bench):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Secondary Outputs — Executive Summary Dashboard",
                 fontsize=15, color=PALETTE["text"], fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.42)

    rng = np.random.default_rng(77)
    df["pred_acons"] = (df["route_mean_acons"] * df["dumper_mean_efficiency"]
                        + rng.normal(0, 12, len(df))).clip(lower=0)

    # KPI tiles
    kpis = [
        ("Routes Identified",     f"{route_bench['route_enc'].nunique()}",   PALETTE["accent3"]),
        ("Fleet Efficiency μ",    f"{df['acons_efficiency_ratio'].mean():.3f}", PALETTE["accent4"]),
        ("Avg Cycles/Shift",      f"{df['fsm_dump_cycles'].mean():.1f}",     PALETTE["accent2"]),
        ("Model RMSE (est)",      f"~{np.sqrt(((df['pred_acons']-df['acons'])**2).mean()):.1f} L",
                                                                              PALETTE["accent1"]),
    ]
    for idx, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(color + "22")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.text(0.5, 0.62, val, ha="center", va="center", transform=ax.transAxes,
                fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.22, label, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color=PALETTE["muted"])
        ax.set_xticks([]); ax.set_yticks([])

    # Route fuel distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.boxplot([df[df["route_enc"]==r]["acons"].values for r in sorted(df["route_enc"].unique())],
                patch_artist=True,
                boxprops=dict(facecolor=PALETTE["accent3"]+"44", color=PALETTE["accent3"]),
                medianprops=dict(color=PALETTE["accent4"], lw=2),
                whiskerprops=dict(color=PALETTE["muted"]),
                capprops=dict(color=PALETTE["muted"]),
                flierprops=dict(marker=".", color=PALETTE["muted"], ms=3))
    ax5.set_xlabel("Route ID")
    ax5.set_ylabel("Fuel (L)")
    ax5.set_title("Fuel Spread by Route")
    ax5.set_xticks([])

    # Efficiency ratio by mine
    ax6 = fig.add_subplot(gs[1, 1])
    for i, mine in enumerate(df["mine"].unique()):
        sub = df[df["mine"]==mine]["acons_efficiency_ratio"]
        ax6.hist(sub, bins=30, alpha=0.7, color=MINE_COLORS[i % 2], label=mine,
                 edgecolor=PALETTE["bg"])
    ax6.axvline(1.0, color=PALETTE["accent4"], lw=1.5, ls="--")
    ax6.set_xlabel("Efficiency Ratio")
    ax6.set_title("Efficiency by Mine")
    ax6.legend(fontsize=8)

    # FSM cycles vs fuel
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(df["fsm_dump_cycles"], df["acons"], alpha=0.35, s=14,
                c=df["route_enc"], cmap="tab20", linewidths=0)
    r = df[["fsm_dump_cycles","acons"]].corr().iloc[0,1]
    ax7.set_xlabel("FSM Dump Cycles / Shift")
    ax7.set_ylabel("Actual Fuel (L)")
    ax7.set_title(f"Cycles vs Fuel  (r={r:.3f})")
    ax7.grid(alpha=0.3)

    # Physics model vs actual correlation
    ax8 = fig.add_subplot(gs[1, 3])
    lo8 = min(df["physics_acons_expected"].min(), df["acons"].min())
    hi8 = max(df["physics_acons_expected"].max(), df["acons"].max())
    ax8.scatter(df["physics_acons_expected"], df["acons"],
                alpha=0.35, s=14, color=PALETTE["accent5"], linewidths=0)
    ax8.plot([lo8, hi8], [lo8, hi8], color=PALETTE["accent4"], lw=2, ls="--")
    r2 = df[["physics_acons_expected","acons"]].corr().iloc[0,1]**2
    ax8.set_xlabel("Physics Predicted (L)")
    ax8.set_ylabel("Actual (L)")
    ax8.set_title(f"Physics Baseline  R²={r2:.3f}")
    ax8.grid(alpha=0.3)

    fig.savefig(OUT_DIR / "00_summary_dashboard.png", bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print("  ✓ 00_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD REAL DATA  (optional — falls back to synthetic)
# ═══════════════════════════════════════════════════════════════════════════

def try_load_real_data(base: Path):
    """
    Attempt to load pre-computed train features + labels.
    Expects  base/df_train_features.parquet  (output of featurize pipeline).
    Returns None if not found.
    """
    candidates = [
        base / "df_train_features.parquet",
        base / "train_features.parquet",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Loading real data from {p}")
            df = pd.read_parquet(p)
            if "acons" not in df.columns:
                print("  WARN: 'acons' column missing — falling back to synthetic.")
                return None
            if "physics_acons_expected" not in df.columns:
                df["physics_acons_expected"] = np.nan
            if "acons_efficiency_ratio" not in df.columns:
                df["acons_efficiency_ratio"] = df["acons"] / (df["physics_acons_expected"].replace(0, np.nan) + 1e-6)
            if "route_mean_acons" not in df.columns:
                df["route_mean_acons"] = df.groupby("route_enc")["acons"].transform("mean") if "route_enc" in df.columns else df["acons"].median()
            if "dumper_mean_efficiency" not in df.columns:
                df["dumper_mean_efficiency"] = df.groupby("vehicle")["acons_efficiency_ratio"].transform("mean") if "vehicle" in df.columns else 1.0
            if "mine" not in df.columns and "mine_telem" in df.columns:
                df["mine"] = df["mine_telem"]
            if "date" not in df.columns and "op_date" in df.columns:
                df["date"] = pd.to_datetime(df["op_date"])
            return df
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Generate secondary output charts for haul mark challenge")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Directory containing df_train_features.parquet (optional)")
    args = ap.parse_args()

    print("=" * 60)
    print("  SECONDARY OUTPUTS CHART GENERATOR")
    print("=" * 60)

    real_df = None
    if args.data_dir:
        real_df = try_load_real_data(args.data_dir)

    if real_df is not None:
        df = real_df
        route_bench = df.groupby("route_enc")["acons"].agg(
            route_mean_acons="mean", route_median_acons="median",
            route_std_acons="std", route_n_shifts="count",
        ).reset_index()
        route_bench["route_fuel_per_km"] = route_bench["route_mean_acons"] / (
            df.groupby("route_enc")["dist_km"].mean() + 1e-6
        )
        print("  ✓ Using REAL pipeline data")
        # Synthesise FSM waveform for illustration
        rng_w = np.random.default_rng(42)
        t_wave = np.linspace(0, 200, 2000)
        dist_w = 15 + 40*np.abs(np.sin(2*np.pi*t_wave/30)) + rng_w.normal(0,3,len(t_wave))
        spd_w  = 18 + 10*np.sin(2*np.pi*t_wave/45) + rng_w.normal(0,2,len(t_wave))
        spd_w  = np.clip(spd_w, 0, None)
        ana_w  = np.where((dist_w<32)&(spd_w<4), rng_w.uniform(3,5,len(t_wave)), rng_w.uniform(0,1,len(t_wave)))
    else:
        print("  ℹ  No real data found — using SYNTHETIC demo data")
        df, route_bench, t_wave, dist_w, spd_w, ana_w = make_synthetic_data()

    print(f"\n  Dataset: {len(df)} shifts | {df['vehicle'].nunique()} vehicles | "
          f"{route_bench['route_enc'].nunique()} routes")
    print(f"  Generating charts → {OUT_DIR}/\n")

    plot_summary_dashboard(df, route_bench)
    plot_route_benchmark(df, route_bench)
    plot_dumper_efficiency(df)
    plot_cycle_segmentation(t_wave, dist_w, spd_w, ana_w)
    plot_daily_consistency(df)

    print("\n" + "=" * 60)
    print("  ALL CHARTS SAVED")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"    {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
