#!/usr/bin/env python3
"""
Approach 1: Uses BOTH analog_input_1 dump signal AND FSM cycle features.
LightGBM decides which signal is more predictive per context.
"""

from __future__ import annotations
import argparse, gc, re, warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

MINING_TZ = "Asia/Kolkata"
CRS_WGS84 = "EPSG:4326"
CRS_MINE = "EPSG:32645"
ON_HAUL_M = 80.0
LINE_SAMPLE_STEP_M = 35.0
ANALOG_DUMP_THRESH = 2.5
DUMP_NEAR_M = 60.0

FSM_DUMP_ENTER_M = 32.0
FSM_DUMP_EXIT_M = 52.0
FSM_DUMP_SLOW_MAX_KPH = 4.0
FSM_DUMP_MIN_DWELL_S = 40.0

BASE_COLS = [
    "vehicle_anon", "vehicle", "ts", "latitude", "longitude", "altitude",
    "speed", "ignition", "angle", "disthav", "mine_anon", "mine",
    "analog_input_1", "cumdist", "satellites", "gnss_hdop", "operator_id",
]


@dataclass
class MineSpatial:
    mine_anon: str
    tree_haul: cKDTree | None
    tree_dump: cKDTree | None
    tree_stock: cKDTree | None
    boundary: object | None


def resolve_base(explicit):
    if explicit is not None:
        return explicit.resolve()
    kaggle = Path("/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge")
    if kaggle.exists():
        return kaggle
    return Path(__file__).resolve().parent


def default_submission_path():
    kg = Path("/kaggle/working")
    if kg.exists():
        return kg / "submission_analog_and_fsm.csv"
    return Path(__file__).resolve().parent / "submission_analog_and_fsm.csv"


def mine_anon_from_gpkg_stem(stem):
    m = re.match(r"mine_(\d+)_anonymized", stem, re.I)
    return f"mine{int(m.group(1)):03d}" if m else ""


def sample_linestring_xy(geom, step_m=LINE_SAMPLE_STEP_M):
    pts = []
    if geom is None or geom.is_empty:
        return np.zeros((0, 2), dtype=np.float64)
    lines = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
    for line in lines:
        if line.length == 0:
            continue
        d = 0.0
        while d <= line.length:
            p = line.interpolate(d)
            pts.append([p.x, p.y])
            d += step_m
        c = line.coords
        pts.append([c[-1][0], c[-1][1]])
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), dtype=np.float64)


def layer_union_xy(gdf):
    if gdf is None or gdf.empty:
        return np.zeros((0, 2), dtype=np.float64)
    return sample_linestring_xy(unary_union(gdf.geometry.values))


def load_mine_spatial(gpkg_dir):
    out = {}
    if not gpkg_dir.is_dir():
        return out
    for path in sorted(gpkg_dir.glob("mine_*_anonymized.gpkg")):
        mid = mine_anon_from_gpkg_stem(path.stem)
        if not mid:
            continue
        try:
            haul = layer_union_xy(gpd.read_file(path, layer="haul_road"))
            dump = layer_union_xy(gpd.read_file(path, layer="ob_dump"))
            stock = layer_union_xy(gpd.read_file(path, layer="mineral_stock"))
            bnd_gdf = None
            try:
                bnd_gdf = gpd.read_file(path, layer="ml_boundary")
            except Exception:
                pass
            boundary = unary_union(bnd_gdf.geometry.values) if bnd_gdf is not None and not bnd_gdf.empty else None
            def _tree(arr):
                return cKDTree(arr) if arr.shape[0] >= 2 else None
            out[mid] = MineSpatial(mid, _tree(haul), _tree(dump), _tree(stock), boundary)
            print(f"  Spatial OK {mid}: haul={len(haul)} dump={len(dump)} stock={len(stock)}")
        except Exception as ex:
            print(f"  WARN: {path.name}: {ex}")
    return out


def add_op_date_shift(df, ts_col="ts"):
    ts = df[ts_col]
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_convert(MINING_TZ)
    hour = ts.dt.hour
    is_late = hour >= 22
    op_date = ts.dt.normalize()
    op_date = op_date.where(~is_late, op_date + pd.Timedelta(days=1))
    op_date = pd.to_datetime(op_date.dt.strftime("%Y-%m-%d"))
    shift = np.select([(hour >= 22) | (hour < 6), (hour >= 6) & (hour < 14)], ["C", "A"], default="B")
    return op_date, pd.Series(shift, index=df.index, dtype=object)


def dominant_operator_id(s):
    ser = pd.Series(s).dropna().astype(str).str.strip()
    ser = ser[(ser != "") & (ser.str.lower() != "nan") & (ser != "<NA>")]
    if ser.empty:
        return "unknown"
    mode = ser.mode()
    return str(mode.iloc[0]) if len(mode) else str(ser.iloc[0])


# ── FSM ──────────────────────────────────────────────────────────────────────
def fsm_dump_visit_arrays(dist_dump_m, speed_kph, ignition, dt_s, on_haul,
                           enter_m=FSM_DUMP_ENTER_M, exit_m=FSM_DUMP_EXIT_M,
                           slow_max_kph=FSM_DUMP_SLOW_MAX_KPH, min_dwell_s=FSM_DUMP_MIN_DWELL_S):
    n = len(dist_dump_m)
    cycle_complete_edge = np.zeros(n, dtype=np.float64)
    dwell_dt = np.zeros(n, dtype=np.float64)
    geom_enter = np.zeros(n, dtype=np.float64)
    haul_to_dump_arrival = np.zeros(n, dtype=np.float64)
    dump_qualified = np.zeros(n, dtype=np.float64)
    inside = prev_inside = qualified = False
    dwell = 0.0
    for i in range(n):
        di = dist_dump_m[i]
        if np.isfinite(di):
            if not inside and di <= enter_m:
                inside = True
            elif inside and di > exit_m:
                inside = False
        if prev_inside and not inside:
            if qualified:
                cycle_complete_edge[i] = 1.0
            qualified = False
            dwell = 0.0
        if inside and not prev_inside:
            dwell = 0.0
            qualified = False
            geom_enter[i] = 1.0
            if i > 0 and on_haul[i - 1] > 0.5:
                haul_to_dump_arrival[i] = 1.0
        slow = (speed_kph[i] < slow_max_kph) and (ignition[i] >= 0.5)
        if inside:
            if slow:
                dwell += dt_s[i]
                dwell_dt[i] = dt_s[i]
            if dwell >= min_dwell_s:
                qualified = True
        if inside and qualified:
            dump_qualified[i] = 1.0
        prev_inside = inside
    return cycle_complete_edge, dwell_dt, geom_enter, haul_to_dump_arrival, dump_qualified


def apply_fsm_dump_features(df):
    n = len(df)
    if n == 0:
        return
    dist = df["dist_dump_m"].to_numpy(dtype=np.float64, copy=False)
    spd = pd.to_numeric(df["speed"], errors="coerce").to_numpy()
    spd = np.where(np.isfinite(spd), spd, 999.0)
    ign = pd.to_numeric(df["ignition"], errors="coerce").to_numpy()
    ign = np.where(np.isfinite(ign), ign, 0.0)
    dt = df["dt"].to_numpy(dtype=np.float64, copy=False)
    haul = pd.to_numeric(df["on_haul"], errors="coerce").fillna(0.0).to_numpy()
    cyc = np.zeros(n); dw = np.zeros(n); ge = np.zeros(n); ha = np.zeros(n); dq = np.zeros(n)
    veh = df["vehicle"].to_numpy()
    start = 0
    while start < n:
        v = veh[start]
        end = start + 1
        while end < n and veh[end] == v:
            end += 1
        sl = slice(start, end)
        ce, ddt, g, h, q = fsm_dump_visit_arrays(dist[sl], spd[sl], ign[sl], dt[sl], haul[sl])
        cyc[sl]=ce; dw[sl]=ddt; ge[sl]=g; ha[sl]=h; dq[sl]=q
        start = end
    df["fsm_dump_cycle_edge"] = cyc
    df["fsm_dump_dwell_dt"] = dw
    df["fsm_dump_geom_enter"] = ge
    df["fsm_haul_to_dump_arrival"] = ha
    df["fsm_dump_qualified"] = dq
# ─────────────────────────────────────────────────────────────────────────────


def attach_projected_and_spatial(df, transformer, spatial):
    lat = pd.to_numeric(df.get("latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("longitude"), errors="coerce")
    valid = lat.notna() & lon.notna() & np.isfinite(lat) & np.isfinite(lon)
    xm = np.zeros(len(df)); ym = np.zeros(len(df))
    if valid.any():
        xx, yy = transformer.transform(lon[valid].to_numpy(), lat[valid].to_numpy())
        xm[np.where(valid)] = xx; ym[np.where(valid)] = yy
    df["_x"] = xm; df["_y"] = ym; df["_gps_ok"] = valid.astype(np.int8)
    mines = df["mine_anon"].astype(str).fillna("unknown").values
    n = len(df)
    d_haul = np.full(n, np.nan); d_dump = np.full(n, np.nan)
    d_stock = np.full(n, np.nan); inside = np.zeros(n)
    for m in pd.unique(mines):
        if m not in spatial:
            continue
        sp = spatial[m]
        idx = np.where(mines == m)[0]
        pts = np.column_stack([xm[idx], ym[idx]])
        mask = valid.to_numpy()[idx]
        if sp.tree_haul is not None and mask.any():
            d_haul[idx[mask]] = sp.tree_haul.query(pts[mask], k=1)[0]
        if sp.tree_dump is not None and mask.any():
            d_dump[idx[mask]] = sp.tree_dump.query(pts[mask], k=1)[0]
        if sp.tree_stock is not None and mask.any():
            d_stock[idx[mask]] = sp.tree_stock.query(pts[mask], k=1)[0]
        if sp.boundary is not None and mask.any():
            inside[idx[mask]] = contains_xy(sp.boundary, pts[mask,0], pts[mask,1]).astype(np.float64)
    df["dist_haul_m"] = d_haul; df["dist_dump_m"] = d_dump
    df["dist_stock_m"] = d_stock; df["inside_ml_boundary"] = inside
    df["on_haul"] = ((df["dist_haul_m"].notna()) & (df["dist_haul_m"] <= ON_HAUL_M)).astype(np.float64)
    df["near_dump"] = ((df["dist_dump_m"].notna()) & (df["dist_dump_m"] <= DUMP_NEAR_M)).astype(np.float64)


def featurize_file(fpath, transformer, spatial):
    avail = set(pq.read_schema(str(fpath)).names)
    cols = [c for c in BASE_COLS if c in avail]
    df = pd.read_parquet(fpath, columns=cols)
    if "vehicle_anon" in df.columns:
        df.rename(columns={"vehicle_anon": "vehicle"}, inplace=True)
    elif "vehicle" not in df.columns:
        raise ValueError(f"No vehicle column in {fpath}")
    if "mine_anon" not in df.columns and "mine" in df.columns:
        df.rename(columns={"mine": "mine_anon"}, inplace=True)
    df.sort_values(["vehicle", "ts"], inplace=True)
    df["op_date"], df["shift"] = add_op_date_shift(df)
    df["dt"] = df.groupby("vehicle")["ts"].diff().dt.total_seconds().clip(0,120).fillna(0).values
    for c in ["speed","ignition","altitude","disthav","analog_input_1","angle","latitude","longitude"]:
        if c not in df.columns:
            df[c] = np.nan if c in ("latitude","longitude") else 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in ("latitude","longitude") else df[c].fillna(0.0)
    for c in ("latitude","longitude"):
        df[c] = df.groupby("vehicle")[c].transform(lambda x: x.interpolate(limit=5))
        df[c] = df.groupby("vehicle")[c].transform(lambda x: x.ffill().bfill())
    if "mine_anon" not in df.columns:
        df["mine_anon"] = "unknown"
    if "operator_id" in df.columns:
        oid = df["operator_id"].astype(str).str.strip().replace({"nan":"","None":"","<NA>":""})
        df["_operator_id"] = oid.mask(oid == "", np.nan)
    else:
        df["_operator_id"] = np.nan
    df["gps_poor"] = (pd.to_numeric(df.get("satellites",0), errors="coerce").fillna(0) < 5).astype(np.float64)

    attach_projected_and_spatial(df, transformer, spatial)

    # ── Analog signal ──
    analog = pd.to_numeric(df["analog_input_1"], errors="coerce").fillna(0.0)
    df["dump_analog_high"] = (analog > ANALOG_DUMP_THRESH).astype(np.float64)
    analog_high = df["dump_analog_high"].values
    prev_high = np.concatenate([[0.0], analog_high[:-1]])
    same_veh = np.concatenate([[False], df["vehicle"].values[1:] == df["vehicle"].values[:-1]])
    df["dump_analog_edge"] = ((analog_high > 0.5) & (prev_high < 0.5) & same_veh).astype(np.float64)

    # ── FSM ──
    apply_fsm_dump_features(df)

    df["near_dump_slow"] = ((df["near_dump"] > 0.5) & (df["speed"] < 4)).astype(np.float64)

    ign = (df["ignition"] == 1).astype(float)
    df["run_s"] = ign * df["dt"]
    df["idle_s"] = ((df["ignition"] == 1) & (df["speed"] < 2)).astype(float) * df["dt"]
    df["move_s"] = ((df["ignition"] == 1) & (df["speed"] >= 2)).astype(float) * df["dt"]
    df["climb"] = df.groupby("vehicle")["altitude"].diff().clip(lower=0).fillna(0)
    df["heading_chg"] = df.groupby("vehicle")["angle"].diff().abs().clip(0,180).fillna(0)

    def p90(s): return float(np.nanpercentile(s, 90)) if len(s) else 0.0

    agg_kw = dict(
        run_hrs=("run_s", lambda x: x.sum()/3600),
        idle_hrs=("idle_s", lambda x: x.sum()/3600),
        move_hrs=("move_s", lambda x: x.sum()/3600),
        dist_km=("disthav", lambda x: x.sum()/1000),
        speed_mean=("speed","mean"), speed_std=("speed","std"), speed_max=("speed","max"),
        alt_mean=("altitude","mean"), alt_std=("altitude","std"),
        alt_range=("altitude", lambda x: x.max()-x.min()),
        # Analog features
        dump_count=("dump_analog_high","sum"),
        dump_events=("dump_analog_edge","sum"),
        # FSM features
        fsm_dump_cycles=("fsm_dump_cycle_edge","sum"),
        fsm_dump_dwell_hrs=("fsm_dump_dwell_dt", lambda x: x.sum()/3600),
        fsm_enter_dump_geom=("fsm_dump_geom_enter","sum"),
        fsm_arrival_haul_dump=("fsm_haul_to_dump_arrival","sum"),
        frac_fsm_dump_qualified=("fsm_dump_qualified","mean"),
        # Spatial
        near_dump_slow_frac=("near_dump_slow","mean"),
        frac_near_dump=("near_dump","mean"),
        frac_on_haul=("on_haul","mean"),
        frac_inside_ml=("inside_ml_boundary","mean"),
        haul_dist_mean=("dist_haul_m","mean"),
        haul_dist_p90=("dist_haul_m", p90),
        dump_dist_mean=("dist_dump_m","mean"),
        stock_dist_mean=("dist_stock_m","mean"),
        n_rec=("ts","count"),
        mine_telem=("mine_anon","first"),
        cum_climb_m=("climb","sum"),
        heading_chg_mean=("heading_chg","mean"),
        frac_gps_poor=("gps_poor","mean"),
        operator_shift=("_operator_id", dominant_operator_id),
    )
    if "cumdist" in df.columns:
        df["cumdist"] = df.groupby("vehicle")["cumdist"].ffill().fillna(0.0)
        agg_kw["cumdist_span_km"] = ("cumdist", lambda x: float(np.nanmax(x)-np.nanmin(x))/1000 if len(x) else 0.0)

    agg = df.groupby(["vehicle","op_date","shift"], sort=False).agg(**agg_kw).reset_index()
    agg["idle_ratio"] = agg["idle_hrs"] / (agg["run_hrs"] + 1e-6)
    agg["has_dump_signal"] = (agg["dump_events"] > 0).astype(np.float64)
    for c in ("speed_std","alt_std","haul_dist_mean","dump_dist_mean","stock_dist_mean"):
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    if "cumdist_span_km" in agg.columns:
        agg["cumdist_span_km"] = agg["cumdist_span_km"].fillna(0.0)
    del df; gc.collect()
    return agg


AGG_DICT = dict(
    run_hrs="sum", idle_hrs="sum", move_hrs="sum", dist_km="sum",
    speed_mean="mean", speed_std="mean", speed_max="max",
    alt_mean="mean", alt_std="mean", alt_range="max",
    dump_count="sum", dump_events="sum",
    fsm_dump_cycles="sum", fsm_dump_dwell_hrs="sum",
    fsm_enter_dump_geom="sum", fsm_arrival_haul_dump="sum",
    frac_fsm_dump_qualified="mean",
    near_dump_slow_frac="mean", frac_near_dump="mean",
    frac_on_haul="mean", frac_inside_ml="mean",
    haul_dist_mean="mean", haul_dist_p90="mean",
    dump_dist_mean="mean", stock_dist_mean="mean",
    n_rec="sum", mine_telem="first", operator_shift="first",
    idle_ratio="mean", cum_climb_m="sum", heading_chg_mean="mean",
    frac_gps_poor="mean", has_dump_signal="max",
    cumdist_span_km="max",
)


def merge_daily_rollups(shift_df):
    daily = shift_df.groupby(["vehicle","op_date"], sort=False).agg(
        daily_dist_km=("dist_km","sum"), daily_run_hrs=("run_hrs","sum")
    ).reset_index()
    return shift_df.merge(daily, on=["vehicle","op_date"], how="left")


def build_dumper_profile(tr):
    agg_kw = {}
    for col, mean_name, std_name in [
        ("dist_km","dumper_mean_dist_km","dumper_std_dist_km"),
        ("run_hrs","dumper_mean_run_hrs","dumper_std_run_hrs"),
        ("move_hrs","dumper_mean_move_hrs",None),
        ("idle_ratio","dumper_mean_idle_ratio","dumper_std_idle_ratio"),
        ("dump_events","dumper_mean_dump_events","dumper_std_dump_events"),
        ("fsm_dump_cycles","dumper_mean_fsm_cycles","dumper_std_fsm_cycles"),
        ("frac_on_haul","dumper_mean_frac_haul",None),
    ]:
        if col in tr.columns:
            agg_kw[mean_name] = (col,"mean")
            if std_name:
                agg_kw[std_name] = (col,"std")
    if "dist_km" in tr.columns:
        agg_kw["dumper_train_shifts"] = ("dist_km","count")
    if not agg_kw:
        return pd.DataFrame({"vehicle": pd.Series(dtype=object)})
    out = tr.groupby("vehicle", sort=False).agg(**agg_kw).reset_index()
    for c in out.columns:
        if "std" in c:
            out[c] = out[c].fillna(0.0)
    return out


def global_fallback(tr):
    fb = {}
    for col, key in [
        ("dist_km","dumper_mean_dist_km"), ("run_hrs","dumper_mean_run_hrs"),
        ("move_hrs","dumper_mean_move_hrs"), ("idle_ratio","dumper_mean_idle_ratio"),
        ("dump_events","dumper_mean_dump_events"), ("fsm_dump_cycles","dumper_mean_fsm_cycles"),
        ("frac_on_haul","dumper_mean_frac_haul"),
    ]:
        if col in tr.columns:
            fb[key] = float(np.nanmean(tr[col].to_numpy()))
    for col, key in [
        ("dist_km","dumper_std_dist_km"), ("run_hrs","dumper_std_run_hrs"),
        ("idle_ratio","dumper_std_idle_ratio"), ("dump_events","dumper_std_dump_events"),
        ("fsm_dump_cycles","dumper_std_fsm_cycles"),
    ]:
        if col in tr.columns:
            fb[key] = float(np.nanstd(tr[col].to_numpy()) or 0.0)
    if "dist_km" in tr.columns:
        fb["dumper_train_shifts"] = 1.0
    return fb


def attach_dumper_variation(df, prof, fb):
    has_prof = prof is not None and not prof.empty and "vehicle" in prof.columns
    out = df.merge(prof, on="vehicle", how="left") if has_prof else df.copy()
    for k, v in fb.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)
    pairs = [
        ("dist_km","dumper_mean_dist_km","dist_vs_dumper_mean"),
        ("run_hrs","dumper_mean_run_hrs","run_hrs_vs_dumper_mean"),
        ("idle_ratio","dumper_mean_idle_ratio","idle_ratio_vs_dumper_mean"),
        ("dump_events","dumper_mean_dump_events","dump_events_vs_dumper_mean"),
        ("fsm_dump_cycles","dumper_mean_fsm_cycles","fsm_cycles_vs_dumper_mean"),
        ("move_hrs","dumper_mean_move_hrs","move_hrs_vs_dumper_mean"),
        ("frac_on_haul","dumper_mean_frac_haul","frac_haul_vs_dumper_mean"),
    ]
    for col, mean_col, delta_col in pairs:
        if mean_col in out.columns and col in out.columns:
            out[delta_col] = out[col] - out[mean_col]
    if "tankcap" in out.columns and "dist_km" in out.columns:
        tc = pd.to_numeric(out["tankcap"], errors="coerce").replace(0, np.nan)
        out["dist_per_tankcap"] = out["dist_km"].astype(float) / (tc + 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--id-map", type=str, default="id_mapping_new.csv")
    ap.add_argument("--gpkg-dir", type=Path, default=None)
    ap.add_argument("--no-spatial", action="store_true")
    args = ap.parse_args()

    base = resolve_base(args.data_dir)
    gpkg_dir = args.gpkg_dir.resolve() if args.gpkg_dir else base
    out_path = default_submission_path() if args.output is None else (
        Path(args.output) if Path(args.output).suffix.lower() == ".csv" else Path(args.output) / "submission_analog_and_fsm.csv"
    )
    print(f"BASE={base}\nGPKG={gpkg_dir}\nOUT={out_path}")

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise SystemExit("pip install lightgbm") from e

    transformer = Transformer.from_crs(CRS_WGS84, CRS_MINE, always_xy=True)
    spatial = {} if args.no_spatial else load_mine_spatial(gpkg_dir)

    # ── RFID refuelling data ─────────────────────────────────────────────────
    print("Loading RFID refuelling data...")
    rfid_path = base / "rfid_refuels_2026-01-01_2026-02-28.parquet"
    rfid_shift_sorted = None
    rfid_veh_prof = None

    if rfid_path.exists():
        rfid = pd.read_parquet(rfid_path)
        rfid["litres"] = pd.to_numeric(rfid["litres"], errors="coerce").fillna(0.0)
        rfid["date_dpr"] = pd.to_datetime(rfid["date_dpr"])
        rfid["shift_dpr"] = rfid["shift_dpr"].astype(str).str.strip().str.upper()

        rfid_shift = (
            rfid.groupby(["vehicle", "date_dpr", "shift_dpr"])
            .agg(
                rfid_litres=("litres", "sum"),
                rfid_n_refuels=("litres", "count"),
                rfid_max_fill=("litres", "max"),
            )
            .reset_index()
            .rename(columns={"date_dpr": "rfid_date", "shift_dpr": "rfid_shift"})
        )

        rfid_veh_prof = (
            rfid_shift.groupby("vehicle")
            .agg(
                rfid_vehicle_mean_litres=("rfid_litres", "mean"),
                rfid_vehicle_std_litres=("rfid_litres", "std"),
                rfid_vehicle_mean_fills=("rfid_n_refuels", "mean"),
            )
            .reset_index()
        )
        rfid_veh_prof["rfid_vehicle_std_litres"] = rfid_veh_prof["rfid_vehicle_std_litres"].fillna(0.0)

        rfid_shift_sorted = rfid_shift.sort_values(["vehicle", "rfid_date", "rfid_shift"]).copy()
        rfid_shift_sorted["rfid_litres_lag1"] = (
            rfid_shift_sorted.groupby("vehicle")["rfid_litres"].shift(1)
        )

        # Dominant bowser per shift — proxy for mine sub-zone / route
        all_bowsers = sorted(rfid["bowser_anon"].dropna().unique().tolist())
        b2i = {b: i for i, b in enumerate(all_bowsers)}
        rfid["bowser_enc"] = rfid["bowser_anon"].map(b2i).fillna(-1).astype(int)
        rfid_bowser = (
            rfid.groupby(["vehicle", "date_dpr", "shift_dpr"])["bowser_enc"]
            .agg(lambda x: int(x.mode().iloc[0]) if len(x) else -1)
            .reset_index()
            .rename(columns={"date_dpr": "rfid_date", "shift_dpr": "rfid_shift"})
        )
        rfid_shift_sorted = rfid_shift_sorted.merge(
            rfid_bowser, on=["vehicle", "rfid_date", "rfid_shift"], how="left"
        )
        print(
            f"RFID loaded: {len(rfid)} transactions | "
            f"{rfid_shift_sorted['vehicle'].nunique()} vehicles | "
            f"{len(rfid_shift_sorted)} shift rows"
        )
    else:
        print("  WARN: RFID file not found — skipping RFID features.")

    def merge_rfid(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if rfid_shift_sorted is None:
            return df
        out = df.merge(
            rfid_shift_sorted[[
                "vehicle", "rfid_date", "rfid_shift",
                "rfid_litres", "rfid_n_refuels", "rfid_max_fill",
                "rfid_litres_lag1", "bowser_enc",
            ]],
            left_on=["vehicle", date_col, "shift"],
            right_on=["vehicle", "rfid_date", "rfid_shift"],
            how="left",
        )
        if rfid_veh_prof is not None:
            out = out.merge(rfid_veh_prof, on="vehicle", how="left")
        out["rfid_vs_vehicle_mean"] = out["rfid_litres"] - out.get("rfid_vehicle_mean_litres", np.nan)
        if "dist_km" in out.columns:
            out["rfid_fuel_per_km"] = out["rfid_litres"] / (out["dist_km"] + 1e-6)
        return out
    # ─────────────────────────────────────────────────────────────────────────

    smry = pd.concat([
        pd.read_csv(base / "smry_jan_train_ordered.csv"),
        pd.read_csv(base / "smry_feb_train_ordered.csv"),
        pd.read_csv(base / "smry_mar_train_ordered.csv"),
    ], ignore_index=True)
    smry["date"] = pd.to_datetime(smry["date"])
    smry["shift"] = smry["shift"].astype(str).str.strip().str.upper()
    smry = smry.dropna(subset=["acons"])
    smry = smry[np.isfinite(smry["acons"].astype(float))]
    smry = smry[~((smry["date"].dt.month == 3) & (smry["date"].dt.day > 11))]

    fleet = pd.read_csv(base / "fleet.csv")
    dumper_fleet = fleet[fleet["fleet"] == "Dumper"][["vehicle","tankcap","dump_switch","mine_anon"]].copy()

    id_map = pd.read_csv(base / args.id_map)
    id_map["date"] = pd.to_datetime(id_map["date"])
    id_map["shift"] = id_map["shift"].astype(str).str.strip().str.upper()

    train_files = [
        base / "telemetry_2026-01-01_2026-01-10.parquet",
        base / "telemetry_2026-01-11_2026-01-20.parquet",
        base / "telemetry_2026-02-01_2026-02-10.parquet",
        base / "telemetry_2026-02-11_2026-02-20.parquet",
        base / "telemetry_2026-03-01_2026-03-11.parquet",
    ]
    test_files = [
        base / "telemetry_2026-01-21_2026-01-31.parquet",
        base / "telemetry_2026-02-21_2026-02-28.parquet",
        base / "telemetry_2026-03-12_2026-03-20.parquet",
    ]
    for paths, label in [(train_files,"train"),(test_files,"test")]:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {label}: {missing}")

    print("Train telemetry...")
    tr_feats = pd.concat([featurize_file(f, transformer, spatial) for f in train_files], ignore_index=True)
    agg_d = {k:v for k,v in AGG_DICT.items() if k in tr_feats.columns}
    tr_feats = tr_feats.groupby(["vehicle","op_date","shift"], as_index=False).agg(agg_d)
    tr_feats = merge_daily_rollups(tr_feats)
    dumper_prof = build_dumper_profile(tr_feats)
    dumper_fb = global_fallback(tr_feats)
    print(f"Train rows: {len(tr_feats)}")

    print("Test telemetry...")
    te_feats = pd.concat([featurize_file(f, transformer, spatial) for f in test_files], ignore_index=True)
    agg_d_te = {k:v for k,v in AGG_DICT.items() if k in te_feats.columns}
    te_feats = te_feats.groupby(["vehicle","op_date","shift"], as_index=False).agg(agg_d_te)
    te_feats = merge_daily_rollups(te_feats)
    test_dates = set(pd.to_datetime(id_map["date"]).dt.normalize().unique())
    te_feats = te_feats[te_feats["op_date"].isin(test_dates)].reset_index(drop=True)
    print(f"Test rows: {len(te_feats)}")

    df_train = smry.merge(tr_feats, left_on=["vehicle","date","shift"], right_on=["vehicle","op_date","shift"], how="inner")
    for df_ in (df_train, te_feats):
        df_.merge(dumper_fleet[["vehicle","tankcap","dump_switch"]], on="vehicle", how="left")
    df_train = df_train.merge(dumper_fleet[["vehicle","tankcap","dump_switch"]], on="vehicle", how="left")
    te_feats = te_feats.merge(dumper_fleet[["vehicle","tankcap","dump_switch"]], on="vehicle", how="left")
    for df_ in (df_train, te_feats):
        df_["dump_switch"] = pd.to_numeric(df_["dump_switch"], errors="coerce").fillna(0)

    df_train = attach_dumper_variation(df_train, dumper_prof, dumper_fb)
    te_feats = attach_dumper_variation(te_feats, dumper_prof, dumper_fb)

    # Merge RFID features into train and test
    df_train = merge_rfid(df_train, "date")
    te_feats = merge_rfid(te_feats, "op_date")

    for df_, dcol in [(df_train,"date"),(te_feats,"op_date")]:
        df_[dcol] = pd.to_datetime(df_[dcol])
        df_["dow"] = df_[dcol].dt.dayofweek
        df_["month"] = df_[dcol].dt.month
        df_["day"] = df_[dcol].dt.day

    all_vehicles = sorted(set(df_train["vehicle"].tolist() + te_feats["vehicle"].tolist()))
    all_mines = sorted(set(df_train["mine"].fillna("unk").tolist() + te_feats["mine_telem"].fillna("unk").tolist()))
    v2i = {v:i for i,v in enumerate(all_vehicles)}
    m2i = {m:i for i,m in enumerate(all_mines)}
    shift2i = {"C":0,"A":1,"B":2}

    for df_ in (df_train, te_feats):
        df_["operator_shift"] = df_["operator_shift"].fillna("unknown").astype(str)
    all_ops = sorted(set(df_train["operator_shift"].tolist() + te_feats["operator_shift"].tolist()))
    o2i = {o:i for i,o in enumerate(all_ops)}
    df_train["operator_enc"] = df_train["operator_shift"].map(o2i).fillna(-1).astype(int)
    te_feats["operator_enc"] = te_feats["operator_shift"].map(o2i).fillna(-1).astype(int)
    df_train["vehicle_enc"] = df_train["vehicle"].map(v2i).fillna(-1).astype(int)
    df_train["mine_enc"] = df_train["mine"].fillna("unk").map(m2i).fillna(-1).astype(int)
    df_train["shift_enc"] = df_train["shift"].map(shift2i).fillna(-1).astype(int)
    te_feats["vehicle_enc"] = te_feats["vehicle"].map(v2i).fillna(-1).astype(int)
    te_feats["mine_enc"] = te_feats["mine_telem"].fillna("unk").map(m2i).fillna(-1).astype(int)
    te_feats["shift_enc"] = te_feats["shift"].map(shift2i).fillna(-1).astype(int)

    FEAT_COLS = [
        "run_hrs","idle_hrs","move_hrs","dist_km",
        "speed_mean","speed_std","speed_max",
        "alt_mean","alt_std","alt_range",
        # Both analog AND FSM
        "dump_count","dump_events","has_dump_signal",
        "fsm_dump_cycles","fsm_dump_dwell_hrs",
        "fsm_enter_dump_geom","fsm_arrival_haul_dump","frac_fsm_dump_qualified",
        "near_dump_slow_frac","frac_near_dump",
        "n_rec","idle_ratio","tankcap","dump_switch",
        "vehicle_enc","mine_enc","shift_enc","operator_enc",
        "dow","month","day",
        "cum_climb_m","heading_chg_mean","frac_gps_poor",
        "frac_on_haul","frac_inside_ml",
        "haul_dist_mean","haul_dist_p90","dump_dist_mean","stock_dist_mean",
        "daily_dist_km","daily_run_hrs",
        # Dumper profile
        "dumper_mean_dist_km","dumper_std_dist_km","dumper_train_shifts",
        "dumper_mean_run_hrs","dumper_std_run_hrs","dumper_mean_move_hrs",
        "dumper_mean_idle_ratio","dumper_std_idle_ratio",
        "dumper_mean_dump_events","dumper_std_dump_events",
        "dumper_mean_fsm_cycles","dumper_std_fsm_cycles",
        "dumper_mean_frac_haul",
        "dist_vs_dumper_mean","run_hrs_vs_dumper_mean",
        "idle_ratio_vs_dumper_mean","dump_events_vs_dumper_mean",
        "fsm_cycles_vs_dumper_mean","move_hrs_vs_dumper_mean",
        "frac_haul_vs_dumper_mean","dist_per_tankcap",
    ]
    RFID_FEATS = [
        "rfid_litres",
        "rfid_n_refuels",
        "rfid_max_fill",
        "rfid_litres_lag1",
        "rfid_vehicle_mean_litres",
        "rfid_vehicle_std_litres",
        "rfid_vehicle_mean_fills",
        "rfid_vs_vehicle_mean",
        "rfid_fuel_per_km",
        "bowser_enc",
    ]
    for c in RFID_FEATS:
        if c in df_train.columns:
            FEAT_COLS.append(c)
    if "cumdist_span_km" in tr_feats.columns:
        FEAT_COLS.append("cumdist_span_km")
    FEAT_COLS = [c for c in FEAT_COLS if c in df_train.columns and c in te_feats.columns]

    X_all = df_train[FEAT_COLS].fillna(-1)
    y_all = df_train["acons"].astype(float).values
    print(f"Features: {len(FEAT_COLS)} | Train samples: {len(X_all)}")

    df_train["date_"] = pd.to_datetime(df_train["date"])
    df_train["month_"] = df_train["date_"].dt.month
    df_train["day_"] = df_train["date_"].dt.day

    CV_FOLDS = [
        (lambda d: ((d["month_"]==1)&(d["day_"]<=10))|((d["month_"]==3)&(d["day_"]<=11)),
         lambda d: (d["month_"]==1)&(d["day_"]>10)),
        (lambda d: (d["month_"]==1)|((d["month_"]==2)&(d["day_"]<=10))|((d["month_"]==3)&(d["day_"]<=11)),
         lambda d: (d["month_"]==2)&(d["day_"]>10)),
    ]
    LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.02, max_depth=6, num_leaves=31,
                       min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbosity=-1)

    print("\n" + "═"*60)
    best_iters = []
    for fold_idx, (tr_fn, va_fn) in enumerate(CV_FOLDS, 1):
        tr_m, va_m = tr_fn(df_train), va_fn(df_train)
        Xtr, ytr = X_all.loc[tr_m], y_all[tr_m]
        Xva, yva = X_all.loc[va_m], y_all[va_m]
        print(f"Fold {fold_idx} | train={len(Xtr)} val={len(Xva)}")
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xtr, ytr, eval_set=[(Xva,yva)],
              callbacks=[lgb.early_stopping(150,verbose=False), lgb.log_evaluation(500)])
        preds = np.clip(m.predict(Xva), 0, None)
        rmse = float(np.sqrt(np.mean((yva-preds)**2)))
        best_n = int(m.best_iteration_ or m.n_estimators)
        best_iters.append(best_n)
        print(f"  best_iter={best_n}  RMSE={rmse:.2f} L")

    avg_best_n = max(1, int(np.median(best_iters)))
    model_final = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": avg_best_n})
    model_final.fit(X_all, y_all)
    fi = pd.Series(model_final.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    print("\nTop-12 features:\n", fi.head(12).to_string())

    X_te = te_feats[FEAT_COLS].fillna(-1)
    pred_raw = np.clip(model_final.predict(X_te), 0, None)
    cap = te_feats["tankcap"].astype(float).replace(0, np.nan)
    pred_acons = np.where(np.isfinite(cap), np.minimum(pred_raw, cap.values*1.05), pred_raw)

    fallback_vs = df_train.groupby(["vehicle","shift"])["acons"].median().to_dict()
    fallback_v = df_train.groupby("vehicle")["acons"].median().to_dict()
    global_med = float(np.median(y_all))

    te_pred = te_feats[["vehicle","op_date","shift"]].copy()
    te_pred["pred_acons"] = pred_acons
    sub = id_map.merge(te_pred, left_on=["vehicle","date","shift"], right_on=["vehicle","op_date","shift"], how="left")
    missing = sub["pred_acons"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows missing telemetry; using fallback.")
    sub["pred_acons"] = sub.apply(
        lambda r: r["pred_acons"] if pd.notna(r["pred_acons"])
        else fallback_vs.get((r["vehicle"],r["shift"]), fallback_v.get(r["vehicle"], global_med)),
        axis=1
    )
    sub["pred_acons"] = np.clip(sub["pred_acons"].astype(float), 0, None)
    sub[["id","pred_acons"]].rename(columns={"pred_acons":"fuel_consumption"}).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
