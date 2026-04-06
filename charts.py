"""
eda_haulmark.py
---------------
Run this in Kaggle to generate EDA plots for the HaulMark Challenge dataset.
Saves all figures to /kaggle/working/ as PNG files.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

BASE = Path("/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge")
OUT  = Path("/kaggle/working")

# ── Load summary CSVs ──────────────────────────────────────────────────────────
jan  = pd.read_csv(BASE / "smry_jan_train_ordered.csv")
feb  = pd.read_csv(BASE / "smry_feb_train_ordered.csv")
mar  = pd.read_csv(BASE / "smry_mar_train_ordered.csv")
df   = pd.concat([jan, feb, mar], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

fleet   = pd.read_csv(BASE / "fleet.csv")
id_map  = pd.read_csv(BASE / "id_mapping_new.csv")
id_map["date"] = pd.to_datetime(id_map["date"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total training records  : {len(df)}")
print(f"  Jan: {len(jan)}  |  Feb: {len(feb)}  |  Mar: {len(mar)}")
print(f"Test records (id_map)   : {len(id_map)}")
print(f"Unique trucks (train)   : {df['vehicle'].nunique()}")
print(f"Unique trucks (fleet)   : {len(fleet)}")
print(f"Mines                   : {fleet['mine_anon'].nunique()} ({fleet.groupby('mine_anon').size().to_dict()})")
print(f"Train date range        : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Test  date range        : {id_map['date'].min()} → {id_map['date'].max()}")
print(f"Shifts                  : {df['shift'].value_counts().to_dict()}")
print()
print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()
print("acons summary:")
print(df["acons"].describe().round(2))
print(f"Zero acons rows         : {(df['acons']==0).sum()}")
print(f"Zero runhrs rows        : {(df['runhrs']==0).sum()}")


# ── FIGURE 1: Records / trucks / time coverage ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 1 — Dataset Coverage", fontsize=14, fontweight="bold")

# 1a: records per month
month_counts = {"Jan": len(jan), "Feb": len(feb), "Mar": len(mar)}
axes[0].bar(month_counts.keys(), month_counts.values(), color=["#4C72B0","#DD8452","#55A868"])
axes[0].set_title("Records per Month")
axes[0].set_ylabel("Count")
for i, (k, v) in enumerate(month_counts.items()):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=11)

# 1b: trucks per mine
mine_trucks = fleet.groupby("mine_anon")["vehicle"].count()
axes[1].bar(mine_trucks.index, mine_trucks.values, color=["#4C72B0","#DD8452"])
axes[1].set_title("Trucks per Mine (Fleet)")
axes[1].set_ylabel("Count")
for i, v in enumerate(mine_trucks.values):
    axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=11)

# 1c: shifts per day (heatmap proxy — records per date)
daily = df.groupby("date").size().reset_index(name="count")
axes[2].plot(daily["date"], daily["count"], marker="o", markersize=3, linewidth=1.5)
axes[2].set_title("Daily Record Count (Train)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Records")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT / "fig1_coverage.png", dpi=150)
plt.close()
print("Saved fig1_coverage.png")


# ── FIGURE 2: Target (acons) distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 2 — Target: acons (Fuel Consumption, Litres)", fontsize=14, fontweight="bold")

# 2a: full histogram
axes[0].hist(df["acons"], bins=60, color="#4C72B0", edgecolor="white")
axes[0].axvline(df["acons"].mean(), color="red", linestyle="--", label=f"Mean={df['acons'].mean():.1f}")
axes[0].axvline(df["acons"].median(), color="orange", linestyle="--", label=f"Median={df['acons'].median():.1f}")
axes[0].set_title("acons Distribution (Full)")
axes[0].set_xlabel("Litres")
axes[0].legend()

# 2b: non-zero only
nz = df[df["acons"] > 0]["acons"]
axes[1].hist(nz, bins=60, color="#55A868", edgecolor="white")
axes[1].axvline(nz.mean(), color="red", linestyle="--", label=f"Mean={nz.mean():.1f}")
axes[1].axvline(nz.median(), color="orange", linestyle="--", label=f"Median={nz.median():.1f}")
axes[1].set_title(f"acons Distribution (Non-zero, n={len(nz)})")
axes[1].set_xlabel("Litres")
axes[1].legend()

# 2c: box by shift
df.boxplot(column="acons", by="shift", ax=axes[2], grid=True)
axes[2].set_title("acons by Shift")
axes[2].set_xlabel("Shift")
axes[2].set_ylabel("Litres")
plt.suptitle("")  # suppress auto boxplot title

plt.tight_layout()
plt.savefig(OUT / "fig2_acons_distribution.png", dpi=150)
plt.close()
print("Saved fig2_acons_distribution.png")


# ── FIGURE 3: Zero / idle shift analysis ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 3 — Zero / Idle Shift Analysis", fontsize=14, fontweight="bold")

# 3a: pie of zero vs non-zero acons
zero_ct = (df["acons"] == 0).sum()
nz_ct   = len(df) - zero_ct
axes[0].pie([zero_ct, nz_ct], labels=[f"acons=0\n({zero_ct})", f"acons>0\n({nz_ct})"],
            autopct="%1.1f%%", colors=["#DD8452","#4C72B0"], startangle=90)
axes[0].set_title("Zero vs Non-zero acons")

# 3b: runhrs distribution for zero-acons rows
zero_df = df[df["acons"] == 0]
axes[1].hist(zero_df["runhrs"], bins=40, color="#DD8452", edgecolor="white")
axes[1].set_title("runhrs for Zero-acons Rows")
axes[1].set_xlabel("Run Hours")
axes[1].set_ylabel("Count")

# 3c: acons vs runhrs scatter (sample 1000 pts for speed)
sample = df.sample(min(1500, len(df)), random_state=42)
sc = axes[2].scatter(sample["runhrs"], sample["acons"], alpha=0.4, c=sample["acons"],
                     cmap="viridis", s=15)
plt.colorbar(sc, ax=axes[2], label="acons")
axes[2].set_title("acons vs runhrs")
axes[2].set_xlabel("Run Hours")
axes[2].set_ylabel("acons (Litres)")

plt.tight_layout()
plt.savefig(OUT / "fig3_zero_idle_analysis.png", dpi=150)
plt.close()
print("Saved fig3_zero_idle_analysis.png")


# ── FIGURE 4: acons by vehicle & mine ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig.suptitle("Fig 4 — Per-Vehicle & Per-Mine acons", fontsize=14, fontweight="bold")

# 4a: per-vehicle median acons (sorted)
veh_med = df[df["acons"] > 0].groupby("vehicle")["acons"].median().sort_values()
axes[0].barh(range(len(veh_med)), veh_med.values, color="#4C72B0")
axes[0].set_yticks(range(len(veh_med)))
axes[0].set_yticklabels(veh_med.index, fontsize=7)
axes[0].set_title("Median acons per Vehicle (non-zero shifts)")
axes[0].set_xlabel("Litres")
axes[0].axvline(veh_med.median(), color="red", linestyle="--", label="Fleet median")
axes[0].legend()

# 4b: acons by mine
df_mine = df.merge(fleet[["vehicle","mine_anon"]], on="vehicle", how="left")
df_mine[df_mine["acons"] > 0].boxplot(column="acons", by="mine_anon", ax=axes[1])
axes[1].set_title("acons by Mine (non-zero)")
axes[1].set_xlabel("Mine")
axes[1].set_ylabel("Litres")
plt.suptitle("")

plt.tight_layout()
plt.savefig(OUT / "fig4_vehicle_mine_acons.png", dpi=150)
plt.close()
print("Saved fig4_vehicle_mine_acons.png")


# ── FIGURE 5: lph analysis (outliers / inf) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 5 — lph (Litres per Hour) Analysis", fontsize=14, fontweight="bold")

lph_valid = df[(df["lph"].notna()) & np.isfinite(df["lph"]) & (df["lph"] > 0)]
lph_nan   = df["lph"].isna().sum()
lph_inf   = np.isinf(df["lph"]).sum()

# 5a: status pie
axes[0].pie([len(lph_valid), lph_nan, lph_inf],
            labels=[f"Valid\n({len(lph_valid)})", f"NaN\n({lph_nan})", f"Inf\n({lph_inf})"],
            autopct="%1.1f%%", colors=["#55A868","#DD8452","#C44E52"])
axes[0].set_title("lph Data Quality")

# 5b: distribution of valid lph
axes[1].hist(lph_valid["lph"], bins=60, color="#55A868", edgecolor="white")
p5, p95 = lph_valid["lph"].quantile(0.05), lph_valid["lph"].quantile(0.95)
axes[1].axvline(p5,  color="orange", linestyle="--", label=f"P5={p5:.1f}")
axes[1].axvline(p95, color="red",    linestyle="--", label=f"P95={p95:.1f}")
axes[1].set_title("lph Distribution (Valid only)")
axes[1].set_xlabel("L/hr")
axes[1].legend()

# 5c: lph vs runhrs (scatter)
sc2 = axes[2].scatter(lph_valid["runhrs"], lph_valid["lph"], alpha=0.3, s=10, c="#4C72B0")
axes[2].set_ylim(0, lph_valid["lph"].quantile(0.99) * 1.2)
axes[2].set_title("lph vs runhrs (capped at P99)")
axes[2].set_xlabel("Run Hours")
axes[2].set_ylabel("L/hr")

plt.tight_layout()
plt.savefig(OUT / "fig5_lph_analysis.png", dpi=150)
plt.close()
print("Saved fig5_lph_analysis.png")


# ── FIGURE 6: acons outlier detection ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 6 — acons Outlier Detection", fontsize=14, fontweight="bold")

mu, sd = df["acons"].mean(), df["acons"].std()

# 6a: Z-score distribution
zscore = (df["acons"] - mu) / sd
axes[0].hist(zscore, bins=60, color="#4C72B0", edgecolor="white")
axes[0].axvline(3,  color="red",    linestyle="--", label="+3σ")
axes[0].axvline(-3, color="orange", linestyle="--", label="-3σ")
axes[0].set_title("Z-score of acons")
axes[0].set_xlabel("Z-score")
axes[0].legend()
axes[0].text(0.98, 0.95, f"|z|>3: {(zscore.abs()>3).sum()} rows",
             transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="gray"))

# 6b: box per mine (with outlier dots)
df_mine[df_mine["acons"] > 0].boxplot(column="acons", by="mine_anon", ax=axes[1],
                                       flierprops=dict(marker="o", color="red", markersize=4))
axes[1].set_title("acons Boxplot per Mine (outliers in red)")
plt.suptitle("")
axes[1].set_xlabel("Mine")
axes[1].set_ylabel("Litres")

# 6c: time-series of daily max acons to spot anomalous days
daily_max = df.groupby("date")["acons"].max().reset_index()
axes[2].plot(daily_max["date"], daily_max["acons"], marker="o", markersize=3, linewidth=1.5)
axes[2].axhline(mu + 3*sd, color="red", linestyle="--", label=f"μ+3σ={mu+3*sd:.0f}")
axes[2].set_title("Daily Max acons Over Time")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Max acons (Litres)")
axes[2].legend()
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT / "fig6_outlier_detection.png", dpi=150)
plt.close()
print("Saved fig6_outlier_detection.png")


# ── FIGURE 7: Shift-level patterns ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 7 — Shift-level Patterns", fontsize=14, fontweight="bold")

# 7a: mean acons per shift per month
df["month_name"] = df["date"].dt.strftime("%b")
pivot = df[df["acons"] > 0].groupby(["month_name","shift"])["acons"].mean().unstack()
pivot = pivot.reindex(["Jan","Feb","Mar"])
pivot.plot(kind="bar", ax=axes[0], colormap="tab10")
axes[0].set_title("Mean acons by Shift & Month")
axes[0].set_xlabel("")
axes[0].set_ylabel("Mean acons (Litres)")
axes[0].legend(title="Shift")
axes[0].tick_params(axis="x", rotation=0)

# 7b: runhrs distribution per shift
for sh, grp in df.groupby("shift"):
    axes[1].hist(grp["runhrs"], bins=40, alpha=0.5, label=sh)
axes[1].set_title("runhrs Distribution by Shift")
axes[1].set_xlabel("Run Hours")
axes[1].legend(title="Shift")

# 7c: zero-acons rate per shift
zero_rate = df.groupby("shift").apply(lambda x: (x["acons"]==0).mean() * 100)
axes[2].bar(zero_rate.index, zero_rate.values, color=["#4C72B0","#DD8452","#55A868"])
axes[2].set_title("Zero-acons Rate by Shift (%)")
axes[2].set_ylabel("% of shifts")
for i, v in enumerate(zero_rate.values):
    axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "fig7_shift_patterns.png", dpi=150)
plt.close()
print("Saved fig7_shift_patterns.png")


# ── FIGURE 8: Tank capacity & fleet composition ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 8 — Fleet Composition & Tank Capacity", fontsize=14, fontweight="bold")

# 8a: tank cap counts
tc = fleet["tankcap"].value_counts().sort_index()
axes[0].bar([str(int(x)) for x in tc.index], tc.values, color="#4C72B0")
axes[0].set_title("Tank Capacity Distribution")
axes[0].set_xlabel("Litres")
axes[0].set_ylabel("# Trucks")
for i, v in enumerate(tc.values):
    axes[0].text(i, v + 0.1, str(v), ha="center")

# 8b: median acons by tank cap
df_tc = df[df["acons"] > 0].merge(fleet[["vehicle","tankcap"]], on="vehicle", how="left")
df_tc.boxplot(column="acons", by="tankcap", ax=axes[1])
axes[1].set_title("acons by Tank Capacity (non-zero)")
plt.suptitle("")
axes[1].set_xlabel("Tank Capacity (L)")
axes[1].set_ylabel("acons (Litres)")

# 8c: dump_switch distribution
ds = fleet["dump_switch"].value_counts()
axes[2].bar([str(int(x)) for x in ds.index], ds.values, color=["#4C72B0","#DD8452"])
axes[2].set_title("Dump Switch Flag Distribution")
axes[2].set_xlabel("dump_switch value")
axes[2].set_ylabel("# Trucks")

plt.tight_layout()
plt.savefig(OUT / "fig8_fleet_composition.png", dpi=150)
plt.close()
print("Saved fig8_fleet_composition.png")


# ── FIGURE 9: Correlation heatmap (summary numeric cols) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Fig 9 — Correlation Matrix (Summary Features)", fontsize=14, fontweight="bold")

num_cols = ["initlev","endlev","arefill","runhrs","acons","lph"]
corr_df  = df[num_cols].replace([np.inf, -np.inf], np.nan)
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Pearson Correlation")

plt.tight_layout()
plt.savefig(OUT / "fig9_correlation.png", dpi=150)
plt.close()
print("Saved fig9_correlation.png")


# ── FIGURE 10: arefill (refuel) analysis ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fig 10 — Refuel (arefill) Analysis", fontsize=14, fontweight="bold")

# 10a: arefill distribution
axes[0].hist(df[df["arefill"] > 0]["arefill"], bins=50, color="#C44E52", edgecolor="white")
axes[0].set_title(f"arefill Distribution (>0, n={(df['arefill']>0).sum()})")
axes[0].set_xlabel("Refuel Litres")
axes[0].set_ylabel("Count")

# 10b: arefill by shift
df.boxplot(column="arefill", by="shift", ax=axes[1])
axes[1].set_title("arefill by Shift")
plt.suptitle("")
axes[1].set_xlabel("Shift")
axes[1].set_ylabel("Litres Refuelled")

# 10c: zero-refuel rate per month
df["month_label"] = df["date"].dt.strftime("%b")
zero_ref = df.groupby("month_label").apply(lambda x: (x["arefill"]==0).mean()*100).reindex(["Jan","Feb","Mar"])
axes[2].bar(zero_ref.index, zero_ref.values, color=["#4C72B0","#DD8452","#55A868"])
axes[2].set_title("Zero-refuel Rate by Month (%)")
axes[2].set_ylabel("% of shifts")
for i, v in enumerate(zero_ref.values):
    axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "fig10_refuel_analysis.png", dpi=150)
plt.close()
print("Saved fig10_refuel_analysis.png")


print("\n✅ All EDA figures saved to /kaggle/working/")
print("Files: fig1_coverage.png through fig10_refuel_analysis.png")
