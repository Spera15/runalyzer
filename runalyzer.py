#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import gzip
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dateutil import tz as dateutil_tz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
import matplotlib.ticker as mticker

import gpxpy
from fitparse import FitFile


# ----------------------------
# Helpers
# ----------------------------

def safe_div(a, b):
    return a / b if b and b != 0 else np.nan

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def rolling_mean(series: pd.Series, window_s: int) -> pd.Series:
    return series.rolling(window=window_s, min_periods=max(1, window_s // 2)).mean()

def rolling_median(series: pd.Series, window_s: int) -> pd.Series:
    return series.rolling(window=window_s, min_periods=max(1, window_s // 2)).median()

def pace_from_speed(speed_mps: float) -> float:
    if speed_mps is None or np.isnan(speed_mps) or speed_mps <= 0:
        return np.nan
    return 1000.0 / speed_mps

def format_pace(s_per_km: float) -> str:
    if s_per_km is None or np.isnan(s_per_km) or s_per_km <= 0:
        return "n/a"
    m = int(s_per_km // 60)
    s = int(round(s_per_km - 60*m))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}/km"

def fmt_time(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return "n/a"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

def semicircles_to_deg(x):
    return x * (180.0 / 2**31)

def stem_name(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def sanitize_name(name: str) -> str:
    # Windows-safe
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return name[:120] if len(name) > 120 else name

def out_path(outdir: str, prefix: str, suffix: str, ext: str) -> str:
    # suffix like "analysis_summary", "records_1s", "overview", "intervals_gc"
    return os.path.join(outdir, f"{prefix}-{suffix}.{ext}")

def fmt_mmss(x_seconds: float) -> str:
    if x_seconds is None or np.isnan(x_seconds) or x_seconds < 0:
        return ""
    s = int(round(x_seconds))
    m = s // 60
    sec = s % 60
    return f"{m:02d}:{sec:02d}"

def pace_min_per_km_from_s_per_km(s_per_km: float) -> float:
    if s_per_km is None or np.isnan(s_per_km) or s_per_km <= 0:
        return np.nan
    return s_per_km / 60.0

PHASE_COLORS = {
    "Riscaldamento": "#ef4444",  # red-500
    "Corsa": "#3b82f6",          # blue-500
    "Recupero": "#94a3b8",       # slate-400 (grigio)
    "Defaticamento": "#22c55e",  # green-500
}

def phase_color(label: str) -> str:
    if not label:
        return "#94a3b8"
    key = str(label).strip()
    return PHASE_COLORS.get(key, "#94a3b8")
	
# ----------------------------
# GPX robust read
# ----------------------------

def read_gpx_bytes_maybe_gzip(path: str) -> bytes:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\x1f\x8b"):
        return gzip.decompress(raw)
    return raw

def decode_xml_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")

def parse_gpx(path: str) -> pd.DataFrame:
    raw = read_gpx_bytes_maybe_gzip(path)
    raw_lstrip = raw.lstrip()
    if not raw_lstrip.startswith(b"<"):
        raise RuntimeError("GPX does not look like XML. Check file path/extension.")
    xml_text = decode_xml_bytes(raw)
    gpx = gpxpy.parse(xml_text)

    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if not p.time:
                    continue
                points.append({
                    "timestamp": pd.to_datetime(p.time, utc=True),
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "alt_m": p.elevation if p.elevation is not None else np.nan,
                })

    df = pd.DataFrame(points)
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    # GPX typically lacks these
    df["hr_bpm"] = np.nan
    df["cad_spm"] = np.nan
    df["power_w"] = np.nan
    df["speed_mps"] = np.nan
    df["temp_c"] = np.nan

    return df[["timestamp","lat","lon","alt_m","hr_bpm","cad_spm","power_w","speed_mps","temp_c"]]


# ----------------------------
# FIT parsing + try weight
# ----------------------------

def parse_fit_records(path: str) -> pd.DataFrame:
    fit = FitFile(path)
    rows = []
    for msg in fit.get_messages("record"):
        d = {}
        for f in msg:
            d[f.name] = f.value
        if "timestamp" not in d:
            continue
        rows.append(d)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "position_lat" in df.columns and "position_long" in df.columns:
        df["lat"] = df["position_lat"].apply(lambda v: semicircles_to_deg(v) if pd.notna(v) else np.nan)
        df["lon"] = df["position_long"].apply(lambda v: semicircles_to_deg(v) if pd.notna(v) else np.nan)
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    if "enhanced_altitude" in df.columns:
        df["alt_m"] = pd.to_numeric(df["enhanced_altitude"], errors="coerce")
    elif "altitude" in df.columns:
        df["alt_m"] = pd.to_numeric(df["altitude"], errors="coerce")
    else:
        df["alt_m"] = np.nan

    df["hr_bpm"] = pd.to_numeric(df.get("heart_rate", np.nan), errors="coerce")
    df["cad_spm"] = pd.to_numeric(df.get("cadence", np.nan), errors="coerce")
    df["power_w"] = pd.to_numeric(df.get("power", np.nan), errors="coerce")

    if "enhanced_speed" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["enhanced_speed"], errors="coerce")
    elif "speed" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["speed"], errors="coerce")
    else:
        df["speed_mps"] = np.nan

    df["temp_c"] = pd.to_numeric(df.get("temperature", np.nan), errors="coerce")

    keep = ["timestamp","lat","lon","alt_m","hr_bpm","cad_spm","power_w","speed_mps","temp_c"]
    return df[keep].sort_values("timestamp").reset_index(drop=True)

def parse_fit_laps(path: str) -> pd.DataFrame:
    fit = FitFile(path)
    laps = []
    for msg in fit.get_messages("lap"):
        d = {}
        for f in msg:
            d[f.name] = f.value
        laps.append(d)
    if not laps:
        return pd.DataFrame()
    df = pd.DataFrame(laps)
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    if "timestamp" in df.columns:
        df["end_time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df

def try_extract_weight_from_fit(path: str) -> Optional[float]:
    """
    Not guaranteed. Some FITs have user_profile or athlete messages with weight.
    We'll try a few common message types/fields.
    """
    try:
        fit = FitFile(path)
    except Exception:
        return None

    candidates = []
    for msg_name in ["user_profile", "profile", "athlete", "device_info", "sport"]:
        try:
            for msg in fit.get_messages(msg_name):
                for f in msg:
                    if f.name in ("weight", "user_weight", "athlete_weight"):
                        val = f.value
                        if val is not None:
                            try:
                                candidates.append(float(val))
                            except Exception:
                                pass
        except Exception:
            pass

    # Also scan "session" fields sometimes include total_weight? rare
    try:
        for msg in fit.get_messages("session"):
            for f in msg:
                if f.name in ("weight", "user_weight"):
                    try:
                        candidates.append(float(f.value))
                    except Exception:
                        pass
    except Exception:
        pass

    # Sanity filter: typical human weight range
    candidates = [w for w in candidates if 30 <= w <= 200]
    return candidates[0] if candidates else None

def try_extract_training_effect_from_fit(path: str) -> Dict[str, Optional[float]]:
    """
    Estrae Training Effect se presente nel FIT (session message).
    Campi tipici:
      - total_training_effect
      - total_anaerobic_training_effect
    """
    out = {"aerobic_te": None, "anaerobic_te": None}
    try:
        fit = FitFile(path)
    except Exception:
        return out

    try:
        for msg in fit.get_messages("session"):
            for f in msg:
                if f.name == "total_training_effect":
                    try: out["aerobic_te"] = float(f.value)
                    except Exception: pass
                elif f.name == "total_anaerobic_training_effect":
                    try: out["anaerobic_te"] = float(f.value)
                    except Exception: pass
    except Exception:
        pass
    return out

def try_extract_thresholds_from_fit(path: str) -> Dict[str, float]:
    """
    Tenta di recuperare soglie da FIT (session/user_profile):
    - threshold_heart_rate / lthr
    - threshold_power / functional_threshold_power
    - threshold_speed (m/s) -> pace s/km
    """
    out: Dict[str, float] = {}
    try:
        fit = FitFile(path)
    except Exception:
        return out

    def pace_from_speed_field(val) -> Optional[float]:
        try:
            v = float(val)
            if v > 0:
                return 1000.0 / v
        except Exception:
            pass
        return None

    try:
        for msg in fit.get_messages("session"):
            for f in msg:
                if f.name in ("threshold_heart_rate", "lthr", "anaerobic_threshold_heart_rate"):
                    try:
                        out["hr_thr_bpm"] = float(f.value)
                    except Exception:
                        pass
                elif f.name in ("threshold_power", "functional_threshold_power"):
                    try:
                        out["pwr_thr_w"] = float(f.value)
                    except Exception:
                        pass
                elif f.name == "threshold_speed":
                    p = pace_from_speed_field(f.value)
                    if p:
                        out["pace_thr_s_per_km"] = p
    except Exception:
        pass

    try:
        for msg in fit.get_messages("user_profile"):
            for f in msg:
                if f.name == "lactate_threshold_heart_rate" and "hr_thr_bpm" not in out:
                    try:
                        out["hr_thr_bpm"] = float(f.value)
                    except Exception:
                        pass
    except Exception:
        pass

    return out


# ----------------------------
# Distance/speed + resample
# ----------------------------

def compute_distance_and_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt_s"] = df["timestamp"].diff().dt.total_seconds()
    df["dt_s"] = df["dt_s"].fillna(0).clip(lower=0)

    dist_step = [0.0]
    for i in range(1, len(df)):
        if pd.notna(df.loc[i-1,"lat"]) and pd.notna(df.loc[i,"lat"]) and pd.notna(df.loc[i-1,"lon"]) and pd.notna(df.loc[i,"lon"]):
            d = haversine_m(df.loc[i-1,"lat"], df.loc[i-1,"lon"], df.loc[i,"lat"], df.loc[i,"lon"])
        else:
            d = 0.0
        dist_step.append(d)

    df["d_m"] = dist_step
    df["cum_dist_m"] = df["d_m"].cumsum()

    if df["speed_mps"].isna().all():
        df["speed_mps"] = df.apply(lambda r: safe_div(r["d_m"], r["dt_s"]) if r["dt_s"] > 0 else np.nan, axis=1)

    return df

def resample_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().set_index("timestamp").sort_index()
    df_1s = df.resample("1s").mean(numeric_only=True)

    for col in ["lat","lon","alt_m","hr_bpm","cad_spm","power_w","speed_mps","temp_c"]:
        if col in df_1s.columns:
            df_1s[col] = df_1s[col].interpolate(limit=10, limit_direction="both")

    df_1s = df_1s.reset_index()
    df_1s = compute_distance_and_speed(df_1s)

    df_1s["pace_s_per_km"] = df_1s["speed_mps"].apply(pace_from_speed)
    df_1s["pace_10s"] = rolling_median(df_1s["pace_s_per_km"], 10)
    df_1s["pace_30s"] = rolling_mean(df_1s["pace_s_per_km"], 30)
    df_1s["hr_30s"] = rolling_mean(df_1s["hr_bpm"], 30) if df_1s["hr_bpm"].notna().any() else np.nan
    df_1s["pwr_30s"] = rolling_mean(df_1s["power_w"], 30) if df_1s["power_w"].notna().any() else np.nan
    return df_1s


# ----------------------------
# Interval detection (work vs recovery)
# ----------------------------

def build_intervals_from_laps(df_1s: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """
    Create official intervals using FIT lap boundaries.
    Returns a DataFrame with per-interval stats.
    Assumes df_1s has timestamp, d_m, pace_s_per_km, hr_bpm, power_w, cad_spm.
    """
    if laps is None or laps.empty or "start_time" not in laps.columns or "end_time" not in laps.columns:
        return pd.DataFrame()

    rows = []
    for i, r in laps.iterrows():
        t0 = r.get("start_time")
        t1 = r.get("end_time")
        if pd.isna(t0) or pd.isna(t1):
            continue

        seg = df_1s[(df_1s["timestamp"] >= t0) & (df_1s["timestamp"] <= t1)].copy()
        if seg.empty:
            continue

        dur_s = (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds()
        dist_m = float(seg["d_m"].sum())

        # Avg pace from dist/time (more stable than mean of pace samples)
        avg_speed = safe_div(dist_m, dur_s) if dur_s > 0 else np.nan
        avg_pace = pace_from_speed(avg_speed)

        hr_avg = float(seg["hr_bpm"].mean()) if seg["hr_bpm"].notna().any() else np.nan
        hr_max = float(seg["hr_bpm"].max()) if seg["hr_bpm"].notna().any() else np.nan

        pw_avg = float(seg["power_w"].mean()) if seg["power_w"].notna().any() else np.nan
        pw_max = float(seg["power_w"].max()) if seg["power_w"].notna().any() else np.nan

        cad_avg = float(seg["cad_spm"].mean()) if seg["cad_spm"].notna().any() else np.nan

        rows.append({
            "lap_index": int(i),
            "start": seg["timestamp"].iloc[0],
            "end": seg["timestamp"].iloc[-1],
            "duration_s": float(dur_s),
            "distance_m": dist_m,
            "avg_pace_s_per_km": float(avg_pace) if not np.isnan(avg_pace) else np.nan,
            "hr_avg_bpm": hr_avg,
            "hr_max_bpm": hr_max,
            "power_avg_w": pw_avg,
            "power_max_w": pw_max,
            "cad_avg_spm": cad_avg,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Heuristic labeling:
    # Work segments tend to have faster pace (lower s/km) and/or higher power than median of laps.
    # This labeling can be refined once you confirm your laps correspond to steps.
    pace_med = np.nanmedian(out["avg_pace_s_per_km"].to_numpy())
    pw_med = np.nanmedian(out["power_avg_w"].to_numpy()) if out["power_avg_w"].notna().any() else np.nan

    labels = []
    for _, rr in out.iterrows():
        is_work = False
        if not np.isnan(rr["avg_pace_s_per_km"]) and rr["avg_pace_s_per_km"] < 0.92 * pace_med:
            is_work = True
        if not np.isnan(pw_med) and not np.isnan(rr["power_avg_w"]) and rr["power_avg_w"] > 1.05 * pw_med:
            is_work = True
        labels.append("work" if is_work else "recovery")

    out["phase_type"] = labels
    return out

def detect_intervals(df_1s: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Heuristic segmentation into blocks based on intensity signals.
    Works best for workouts like: warmup -> repeats -> cooldown.
    Output columns: block_id, block_type (work/recovery/other)
    """
    df = df_1s.copy()

    # Choose intensity signal: prefer power (if threshold + data), else pace (if threshold + data), else speed.
    use = None
    if thresholds.get("pwr_thr_w") and df["power_w"].notna().any():
        # intensity = power / thr
        df["intensity"] = df["power_w"] / thresholds["pwr_thr_w"]
        use = "power"
    elif thresholds.get("pace_thr_s_per_km") and df["pace_s_per_km"].notna().any():
        # intensity = thr_pace / pace  ( >1 means faster than threshold )
        df["intensity"] = thresholds["pace_thr_s_per_km"] / df["pace_s_per_km"]
        use = "pace"
    else:
        df["intensity"] = df["speed_mps"]
        use = "speed"

    # Smooth intensity
    df["intensity_s"] = rolling_mean(df["intensity"], 15)

    # Decide a work threshold for intensity:
    # - power: work if >= 0.97 of threshold (approx threshold work)
    # - pace:  work if >= 0.98 (i.e., at/above threshold speed)
    # - speed: work if above 70th percentile of running speeds
    if use == "power":
        work_mask = df["intensity_s"] >= 0.97
    elif use == "pace":
        work_mask = df["intensity_s"] >= 0.98
    else:
        sp = df["speed_mps"].dropna()
        cut = float(np.nanpercentile(sp, 70)) if len(sp) else 0.0
        work_mask = df["speed_mps"] >= cut

    # Convert boolean series into segments with minimum duration
    df["is_work_raw"] = work_mask.fillna(False).astype(bool)

    # Clean up: remove tiny segments (<40s) and merge small gaps (<15s)
    # First pass: identify transitions
    isw = df["is_work_raw"].to_numpy(dtype=bool)
    n = len(isw)

    # Merge gaps shorter than 15s inside work
    gap = 0
    for i in range(n):
        if not isw[i]:
            gap += 1
        else:
            if 0 < gap <= 15:
                isw[i-gap:i] = True
            gap = 0

    # Remove short work bursts
    i = 0
    while i < n:
        if isw[i]:
            j = i
            while j < n and isw[j]:
                j += 1
            dur = j - i
            if dur < 40:
                isw[i:j] = False
            i = j
        else:
            i += 1

    df["is_work"] = isw

    # Label blocks
    block_id = np.full(n, -1, dtype=int)
    block_type = np.full(n, "other", dtype=object)

    cur = -1
    i = 0
    while i < n:
        j = i
        val = isw[i]
        while j < n and isw[j] == val:
            j += 1
        cur += 1
        block_id[i:j] = cur
        block_type[i:j] = "work" if val else "recovery"
        i = j

    df["block_id"] = block_id
    df["block_type"] = block_type

    # Summarize blocks to a separate DF
    blocks = []
    for bid in np.unique(block_id):
        sub = df[df["block_id"] == bid]
        if sub.empty:
            continue
        t0 = sub["timestamp"].iloc[0]
        t1 = sub["timestamp"].iloc[-1]
        dur_s = (t1 - t0).total_seconds()
        dist_m = float(sub["d_m"].sum())
        pace = safe_div(1000.0, safe_div(dist_m, dur_s)) if dur_s > 0 else np.nan
        blocks.append({
            "block_id": int(bid),
            "block_type": sub["block_type"].iloc[0],
            "start": t0,
            "end": t1,
            "duration_s": float(dur_s),
            "distance_m": dist_m,
            "avg_pace_s_per_km": pace,
            "hr_avg_bpm": float(sub["hr_bpm"].mean()) if sub["hr_bpm"].notna().any() else np.nan,
            "hr_max_bpm": float(sub["hr_bpm"].max()) if sub["hr_bpm"].notna().any() else np.nan,
            "power_avg_w": float(sub["power_w"].mean()) if sub["power_w"].notna().any() else np.nan,
        })

    blocks_df = pd.DataFrame(blocks)
    return df, blocks_df

def compute_phase_stats(df_1s: pd.DataFrame, intervals_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in intervals_ts.iterrows():
        seg = df_1s[(df_1s["timestamp"] >= r["start"]) & (df_1s["timestamp"] <= r["end"])].copy()
        if seg.empty:
            continue

        dur_s = float((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds())
        dist_m = float(seg["d_m"].sum())
        avg_speed = safe_div(dist_m, dur_s) if dur_s > 0 else np.nan
        avg_pace = pace_from_speed(avg_speed)

        rows.append({
            "phase_type": r.get("phase_type", r.get("phase_name", "phase")),
            "interval_index": r.get("interval_index", np.nan),
            "duration_s": dur_s,
            "distance_m": dist_m,
            "avg_pace_s_per_km": avg_pace,
            "hr_avg_bpm": float(seg["hr_bpm"].mean()) if seg["hr_bpm"].notna().any() else np.nan,
            "hr_max_bpm": float(seg["hr_bpm"].max()) if seg["hr_bpm"].notna().any() else np.nan,
            "power_avg_w": float(seg["power_w"].mean()) if seg["power_w"].notna().any() else np.nan,
            "power_max_w": float(seg["power_w"].max()) if seg["power_w"].notna().any() else np.nan,
            "cad_avg_spm": float(seg["cad_spm"].mean()) if seg["cad_spm"].notna().any() else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["avg_pace"] = out["avg_pace_s_per_km"].apply(format_pace)
    return out

def compute_interval_details(
    df_1s: pd.DataFrame,
    intervals_ts: pd.DataFrame,
    weight_kg: Optional[float] = None
) -> pd.DataFrame:
    rows = []

    for _, r in intervals_ts.iterrows():
        t0 = pd.to_datetime(r["start"], utc=True)
        t1 = pd.to_datetime(r["end"], utc=True)

        seg = df_1s[(df_1s["timestamp"] >= t0) & (df_1s["timestamp"] <= t1)].copy()
        if seg.empty:
            continue

        dur_s = float((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds())
        dist_m = float(seg["d_m"].sum())
        avg_speed = safe_div(dist_m, dur_s) if dur_s > 0 else np.nan
        avg_pace = pace_from_speed(avg_speed)

        pace_vals = seg["pace_s_per_km"].replace([np.inf, -np.inf], np.nan).dropna()
        hr_vals = seg["hr_bpm"].dropna()
        pw_vals = seg["power_w"].dropna()

        phase = str(r.get("phase_type", r.get("phase_name", r.get("block_type", "phase"))))
        rep = r.get("interval_index", np.nan)
        rep_int = None
        if pd.notna(rep):
            try:
                rep_int = int(rep)
            except Exception:
                rep_int = None

        rows.append({
            "Fase": phase,
            "Ripetuta": rep_int,
            "Durata": fmt_time(dur_s),
            "Durata_s": dur_s,
            "Distanza_km": dist_m/1000.0 if dist_m else np.nan,
            "Passo_med": format_pace(avg_pace),
            "Passo_min": format_pace(float(np.nanpercentile(pace_vals, 10))) if len(pace_vals) else "n/a",
            "Passo_max": format_pace(float(np.nanpercentile(pace_vals, 90))) if len(pace_vals) else "n/a",
            "FC_med": float(hr_vals.mean()) if len(hr_vals) else np.nan,
            "FC_min": float(np.nanpercentile(hr_vals, 10)) if len(hr_vals) else np.nan,
            "FC_max": float(np.nanpercentile(hr_vals, 90)) if len(hr_vals) else np.nan,
            "P_med": float(pw_vals.mean()) if len(pw_vals) else np.nan,
            "P_min": float(np.nanpercentile(pw_vals, 10)) if len(pw_vals) else np.nan,
            "P_max": float(np.nanpercentile(pw_vals, 90)) if len(pw_vals) else np.nan,
            "Wkg_med": (float(pw_vals.mean())/weight_kg) if (weight_kg and len(pw_vals)) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Totali (dove ha senso)
    tot = {
        "Fase": "Totale",
        "Ripetuta": None,
        "Durata": fmt_time(float(out["Durata_s"].sum())),
        "Durata_s": float(out["Durata_s"].sum()),
        "Distanza_km": float(out["Distanza_km"].sum()),
        "Passo_med": "—",
        "Passo_min": "—",
        "Passo_max": "—",
        "FC_med": float(np.nanmean(out["FC_med"])),
        "FC_min": float(np.nanmin(out["FC_min"])),
        "FC_max": float(np.nanmax(out["FC_max"])),
        "P_med": float(np.nanmean(out["P_med"])),
        "P_min": float(np.nanmin(out["P_min"])),
        "P_max": float(np.nanmax(out["P_max"])),
        "Wkg_med": float(np.nanmean(out["Wkg_med"])) if out["Wkg_med"].notna().any() else np.nan,
    }

    out = pd.concat([out, pd.DataFrame([tot])], ignore_index=True)
    return out

# ----------------------------
# Garmin CSV
# ----------------------------

def _parse_time_hms_ms(s: str) -> float:
    """
    Parses:
    - '8:01,2'  -> seconds (481.2)
    - '6:00'    -> 360
    - '51:20'   -> 3080
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    s = s.replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return np.nan

def _parse_float_it(s: str) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    # Italian decimal comma -> dot
    s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 and s.count(".") >= 1 else s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def read_garmin_intervals_csv(path: str) -> pd.DataFrame:
    """
    Reads Garmin Connect export (Ripetute table) in Italian format.
    Returns intervals with start_s/end_s, phase_type, interval_index.
    """
    # Try common separators
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if "Tipo di fase" in df.columns and "Tempo cumulato" in df.columns:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError("Unable to parse Garmin CSV. Check separator/encoding.")

    # Drop summary row if present
    df = df[~df["Intervallo"].astype(str).str.contains("Riepilogo", na=False)].copy()

    # Normalize key fields
    df["phase_name"] = df["Tipo di fase"].astype(str).str.strip()
    df["lap"] = df.get("Lap", "").astype(str)

    df["dur_s"] = df["Tempo"].apply(_parse_time_hms_ms)
    df["cum_s"] = df["Tempo cumulato"].apply(_parse_time_hms_ms)

    # start/end in seconds from activity start (using cumulative time)
    df["end_s"] = df["cum_s"]
    df["start_s"] = df["end_s"] - df["dur_s"]

    # Assign an interval index for "Corsa" blocks
    idx = 0
    interval_idx = []
    for name in df["phase_name"]:
        if name.lower() == "corsa":
            idx += 1
            interval_idx.append(idx)
        else:
            interval_idx.append(np.nan)
    df["interval_index"] = interval_idx

    # Clean types
    out = df[["phase_name","lap","start_s","end_s","dur_s","interval_index"]].copy()
    out = out.dropna(subset=["start_s","end_s"])
    return out.sort_values("start_s").reset_index(drop=True)

def intervals_seconds_to_timestamps(df_1s: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    t0 = df_1s["timestamp"].iloc[0]
    out = intervals.copy()
    out["duration_s"] = out["dur_s"]
    out["start"] = t0 + pd.to_timedelta(out["start_s"], unit="s")
    out["end"] = t0 + pd.to_timedelta(out["end_s"], unit="s")
    # phase_type label
    out["phase_type"] = out["phase_name"]
    return out


# ----------------------------
# Summary
# ----------------------------

def summarize(df_1s: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    total_time_s = (df_1s["timestamp"].iloc[-1] - df_1s["timestamp"].iloc[0]).total_seconds()
    total_dist_m = float(df_1s["cum_dist_m"].iloc[-1])
    avg_speed = safe_div(total_dist_m, total_time_s)
    avg_pace = pace_from_speed(avg_speed)

    out["total_time_s"] = float(total_time_s)
    out["total_dist_km"] = total_dist_m / 1000.0
    out["avg_pace_s_per_km"] = float(avg_pace) if not np.isnan(avg_pace) else np.nan

    out["hr_avg_bpm"] = float(df_1s["hr_bpm"].mean()) if df_1s["hr_bpm"].notna().any() else np.nan
    out["hr_max_bpm"] = float(df_1s["hr_bpm"].max()) if df_1s["hr_bpm"].notna().any() else np.nan

    out["power_avg_w"] = float(df_1s["power_w"].mean()) if df_1s["power_w"].notna().any() else np.nan
    out["power_max_w"] = float(df_1s["power_w"].max()) if df_1s["power_w"].notna().any() else np.nan

    out["cad_avg_spm"] = float(df_1s["cad_spm"].mean()) if df_1s["cad_spm"].notna().any() else np.nan

    if df_1s["alt_m"].notna().any():
        alt_series = pd.Series(df_1s["alt_m"], dtype=float).rolling(window=5, center=True, min_periods=1).median()
        dalt = np.diff(alt_series)
        dalt[np.abs(dalt) < 1.0] = 0.0  # filtra rumore <1m
        gain = float(np.nansum(dalt[dalt > 0]))
        loss = float(-np.nansum(dalt[dalt < 0]))
        out["elev_gain_m"] = gain
        out["elev_loss_m"] = loss
    else:
        out["elev_gain_m"] = np.nan
        out["elev_loss_m"] = np.nan

    # Drift proxy (pace-adjusted to avoid bias when pace varies). Excludes pace slower than 7:30/km.
    drift = {}
    if df_1s["hr_bpm"].notna().any() and df_1s["pace_s_per_km"].notna().any():
        filtered = df_1s[
            df_1s["pace_s_per_km"].notna()
            & df_1s["hr_bpm"].notna()
            & (df_1s["pace_s_per_km"] <= 450)  # <= 7:30 min/km
        ].copy()
        if len(filtered) > 180:
            try:
                slope, intercept = np.polyfit(filtered["pace_s_per_km"], filtered["hr_bpm"], 1)
                filtered["hr_pred"] = slope * filtered["pace_s_per_km"] + intercept
            except Exception:
                slope, intercept = np.nan, np.nan
                filtered["hr_pred"] = np.nan
            filtered = filtered.dropna(subset=["hr_pred"])
            if len(filtered) > 120:
                filtered["hr_residual"] = filtered["hr_bpm"] - filtered["hr_pred"]
                mid_t = filtered["timestamp"].iloc[0] + (filtered["timestamp"].iloc[-1] - filtered["timestamp"].iloc[0]) / 2
                first = filtered[filtered["timestamp"] <= mid_t]
                second = filtered[filtered["timestamp"] > mid_t]
                if len(first) > 30 and len(second) > 30:
                    res1 = float(first["hr_residual"].mean())
                    res2 = float(second["hr_residual"].mean())
                    ref_hr = float(np.nanmedian(filtered["hr_pred"]))
                    drift = {
                        "pace_adj_hr_residual_first_bpm": res1,
                        "pace_adj_hr_residual_second_bpm": res2,
                        "hr_drift_bpm": res2 - res1,
                        "hr_drift_pct": (res2 - res1) / ref_hr * 100.0 if ref_hr else np.nan,
                        "ref_hr_bpm": ref_hr,
                        "pace_model_slope": float(slope) if not np.isnan(slope) else np.nan,
                        "pace_model_intercept": float(intercept) if not np.isnan(intercept) else np.nan,
                        "median_pace_s_per_km": float(np.nanmedian(filtered["pace_s_per_km"])),
                        "samples_used": len(filtered),
                        "note": "Calcolato su residui HR vs passo (filtra >7:30/km)."
                    }
                else:
                    drift = {"note": "Non abbastanza dati dopo filtro ritmo>7:30/km per stimare drift."}
            else:
                drift = {"note": "Non abbastanza dati validi per stimare drift dopo regressione."}
        else:
            drift = {"note": "Dati ritmo/FC insufficienti dopo filtro >7:30 min/km."}
    else:
        drift = {"note": "HR o passo mancanti."}
    out["drift"] = drift

    # Efficiency
    eff = {}
    mask_run = df_1s["speed_mps"] > 1.5
    if mask_run.sum() > 120 and df_1s["hr_bpm"].notna().any():
        eff["hr_per_speed"] = float(np.nanmean(df_1s.loc[mask_run, "hr_bpm"] / df_1s.loc[mask_run, "speed_mps"]))
    else:
        eff["hr_per_speed"] = np.nan
    if mask_run.sum() > 120 and df_1s["power_w"].notna().any():
        eff["power_per_speed"] = float(np.nanmean(df_1s.loc[mask_run, "power_w"] / df_1s.loc[mask_run, "speed_mps"]))
        w = thresholds.get("weight_kg")
        eff["avg_w_per_kg_running"] = float(np.nanmean(df_1s.loc[mask_run, "power_w"] / w)) if w else np.nan
    else:
        eff["power_per_speed"] = np.nan
        eff["avg_w_per_kg_running"] = np.nan
    out["efficiency"] = eff

    return out


# ----------------------------
# Writers
# ----------------------------

def write_summary_md(summary: Dict[str, Any], thresholds: Dict[str, float], intervals: Optional[pd.DataFrame], activity_start_ts: pd.Timestamp, path: str, rpe: float) -> None:
    lines = []
    lines.append("# Running Training Activity Analysis\n\n")
    lines.append(f"- **Date:** {activity_start_ts.strftime('%Y-%m-%d %H:%M:%S')} \n")
    lines.append("\n")
    lines.append("# Activity Analysis Summary\n\n")
    lines.append("## Overall\n")
    lines.append(f"- Duration: **{fmt_time(summary['total_time_s'])}**\n")
    lines.append(f"- Distance: **{summary['total_dist_km']:.2f} km**\n")
    lines.append(f"- Avg pace: **{format_pace(summary['avg_pace_s_per_km'])}**\n")
    if not np.isnan(summary["hr_avg_bpm"]):
        lines.append(f"- HR avg / max: **{summary['hr_avg_bpm']:.0f} / {summary['hr_max_bpm']:.0f} bpm**\n")
    else:
        lines.append("- HR avg / max: **n/a**\n")
    if not np.isnan(summary["power_avg_w"]):
        lines.append(f"- Power avg / max: **{summary['power_avg_w']:.0f} / {summary['power_max_w']:.0f} W**\n")
    else:
        lines.append("- Power avg / max: **n/a**\n")
    lines.append(f"- Elev gain/loss: **{summary['elev_gain_m']:.0f} m / {summary['elev_loss_m']:.0f} m**\n" if not np.isnan(summary["elev_gain_m"]) else "- Elev gain/loss: **n/a**\n")
    if rpe is not None:
        lines.append(f"- **RPE:** {rpe}/10\n")
    lines.append("\n")

    lines.append("## Thresholds used\n")
    if thresholds.get("hr_thr_bpm") is not None:
        lines.append(f"- HR threshold: **{thresholds['hr_thr_bpm']:.0f} bpm**\n")
    if thresholds.get("pwr_thr_w") is not None:
        lines.append(f"- Power threshold: **{thresholds['pwr_thr_w']:.0f} W**\n")
    if thresholds.get("pace_thr_s_per_km") is not None:
        lines.append(f"- Pace threshold: **{format_pace(thresholds['pace_thr_s_per_km'])}**\n")
    if thresholds.get("weight_kg") is not None:
        lines.append(f"- Weight: **{thresholds['weight_kg']:.1f} kg**\n")
    lines.append("\n")

    lines.append("## Cardiac drift\n")
    drift = summary.get("drift", {})
    if "note" in drift:
        lines.append(f"- Note: {drift['note']}\n\n")
    else:
        lines.append(f"- Steady HR: **{drift['steady_hr_first_bpm']:.1f} → {drift['steady_hr_second_bpm']:.1f} bpm** "
                     f"(**Δ {drift['hr_drift_bpm']:.1f} bpm**, {drift['hr_drift_pct']:.1f}%)\n\n")

    lines.append("## Efficiency\n")
    eff = summary.get("efficiency", {})
    lines.append(f"- HR per speed: **{eff.get('hr_per_speed', np.nan):.2f} bpm/(m/s)**\n" if not np.isnan(eff.get("hr_per_speed", np.nan)) else "- HR per speed: **n/a**\n")
    lines.append(f"- Power per speed: **{eff.get('power_per_speed', np.nan):.1f} W/(m/s)**\n" if not np.isnan(eff.get("power_per_speed", np.nan)) else "- Power per speed: **n/a**\n")
    if not np.isnan(eff.get("avg_w_per_kg_running", np.nan)):
        lines.append(f"- Avg W/kg (running): **{eff['avg_w_per_kg_running']:.2f}**\n")
    lines.append("\n")

    if intervals is not None and not intervals.empty:
        lines.append("## Intervals\n")

        # Normalize duration column name across sources
        dur_col = None
        for c in ["duration_s", "dur_s"]:
            if c in intervals.columns:
                dur_col = c
                break

        # If missing, compute from start/end
        s = intervals.copy()
        if dur_col is None:
            if "start" in s.columns and "end" in s.columns:
                s["duration_s"] = (pd.to_datetime(s["end"], utc=True) - pd.to_datetime(s["start"], utc=True)).dt.total_seconds()
                dur_col = "duration_s"

        if dur_col is None:
            lines.append("- Note: intervals provided but no duration information available.\n\n")
        else:
            s = s[s[dur_col] >= 60].copy().head(20)

            # Print using whatever columns exist
            for _, r in s.iterrows():
                label = str(r.get("phase_type", r.get("block_type", r.get("phase_name", "phase"))))
                if "interval_index" in r and pd.notna(r["interval_index"]):
                    label = f"{label} {int(r['interval_index'])}"

                dist_m = r.get("distance_m", np.nan)
                pace = r.get("avg_pace_s_per_km", np.nan)
                hr = r.get("hr_avg_bpm", np.nan)

                line = f"- {label}: {fmt_time(float(r[dur_col]))}"
                if pd.notna(dist_m):
                    line += f", {float(dist_m)/1000:.2f} km"
                if pd.notna(pace):
                    line += f", pace {format_pace(float(pace))}"
                if pd.notna(hr):
                    line += f", HR avg {float(hr):.0f}"
                line += "\n"

                lines.append(line)

            lines.append("\n")


    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def plot_overview(df_1s: pd.DataFrame, intervals: pd.DataFrame, out_png: str) -> None:
    df = df_1s.copy()
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy()

    has_hr = df["hr_bpm"].notna().any()
    has_pw = df["power_w"].notna().any()
    has_pace = df["pace_s_per_km"].notna().any()

    panels = sum([has_hr, has_pace, has_pw])
    panels = max(panels, 1)

    fig, axes = plt.subplots(panels, 1, figsize=(14, 3.2 * panels), sharex=True)
    if panels == 1:
        axes = [axes]

    # --- Phase color mapping (Garmin-like)
    # Vertical lines
    PHASE_LINE = {
        "riscaldamento": "#ef4444",  # red-500
        "corsa": "#3b82f6",          # blue-500
        "recupero": "#9ca3af",       # gray-400
        "defaticamento": "#22c55e",  # green-500
    }
    # Background spans (very light)
    PHASE_SPAN = {
        "riscaldamento": "#fecaca",  # red-200
        "corsa": "#bfdbfe",          # blue-200
        "recupero": "#e5e7eb",       # gray-200
        "defaticamento": "#bbf7d0",  # green-200
    }

    def _phase_key(r) -> str:
        s = str(r.get("phase_type", r.get("phase_name", r.get("block_type", "")))).strip().lower()
        return s

    def add_phase_spans(ax):
        if intervals is None or intervals.empty:
            return
        for _, r in intervals.iterrows():
            if "start" not in r or "end" not in r:
                continue
            x0 = (pd.to_datetime(r["start"], utc=True) - df["timestamp"].iloc[0]).total_seconds()
            x1 = (pd.to_datetime(r["end"], utc=True) - df["timestamp"].iloc[0]).total_seconds()
            key = _phase_key(r)
            col = PHASE_SPAN.get(key, "#e5e7eb")
            ax.axvspan(x0, x1, alpha=0.22, color=col, linewidth=0)

    def add_phase_lines_and_labels(ax):
        if intervals is None or intervals.empty:
            return

        y0, y1 = ax.get_ylim()
        # label at bottom (so it doesn't cover the curve)
        ylab = y0 + 0.02 * (y1 - y0)

        for _, r in intervals.iterrows():
            if "start" not in r or "end" not in r:
                continue
            x0 = (pd.to_datetime(r["start"], utc=True) - df["timestamp"].iloc[0]).total_seconds()
            x1 = (pd.to_datetime(r["end"], utc=True) - df["timestamp"].iloc[0]).total_seconds()

            key = _phase_key(r)
            line_col = PHASE_LINE.get(key, "#94a3b8")

            ax.axvline(x0, linewidth=1.2, color=line_col)
            ax.axvline(x1, linewidth=1.2, color=line_col)

            label = str(r.get("phase_type", r.get("phase_name", key if key else "fase")))
            if "interval_index" in r and pd.notna(r["interval_index"]):
                try:
                    label = f"{label} {int(r['interval_index'])}"
                except Exception:
                    pass

            ax.text(x0 + 2, ylab, label, rotation=90, va="bottom", ha="right", fontsize=8, color="#334155")

    def style_axes(ax):
        # major/minor grid (light grey + very light grey)
        ax.grid(True, which="major", linewidth=0.8, color="#e5e7eb")
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.6, color="#f1f5f9")
        # X axis formatter: mm:ss
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: fmt_mmss(x)))

    k = 0

    # --- HR
    if has_hr:
        ax = axes[k]; k += 1
        add_phase_spans(ax)
        y = df["hr_bpm"].to_numpy(dtype=float)
        ax.plot(t, y, color="#ef4444", linewidth=1.2)
        ax.fill_between(t, y, np.nanmin(y[np.isfinite(y)]) if np.isfinite(y).any() else 0, color="#ef4444", alpha=0.18)
        ax.set_ylim(bottom=np.nanmin(y[np.isfinite(y)]) if np.isfinite(y).any() else 0)
        ax.set_ylabel("HR (bpm)")
        ax.set_title("Heart rate")
        style_axes(ax)
        add_phase_lines_and_labels(ax)

    # --- Pace (min/km)
    if has_pace:
        ax = axes[k]; k += 1
        add_phase_spans(ax)

        pace_s = df["pace_s_per_km"].to_numpy(dtype=float)
        pace_min = np.array([pace_min_per_km_from_s_per_km(v) for v in pace_s], dtype=float)

        # clamp: show details 3..10, saturate outside
        pace_min_clamped = np.clip(pace_min, 3.0, 10.0)

        ax.plot(t, pace_min_clamped, color="#0f172a", linewidth=1.1)  # neutral dark
        ax.set_ylabel("Pace (min/km)")
        ax.set_title("Pace")

        # pace axis: 3 at top, 10 at bottom (Garmin-like)
        ax.set_ylim(10.0, 3.0)
        style_axes(ax)
        add_phase_lines_and_labels(ax)

    # --- Power
    if has_pw:
        ax = axes[k]; k += 1
        add_phase_spans(ax)
        y = df["power_w"].to_numpy(dtype=float)
        ax.plot(t, y, color="#a21caf", linewidth=1.2)  # fuchsia-700
        ax.fill_between(t, y, 0, color="#a21caf", alpha=0.14)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Power (W)")
        ax.set_title("Power")
        style_axes(ax)
        add_phase_lines_and_labels(ax)

    axes[-1].set_xlabel("Time (mm:ss)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_hr_vs_pace_buckets(df_1s: pd.DataFrame, out_png: str) -> None:
    df = df_1s.copy()
    df = df[df["hr_bpm"].notna() & df["pace_s_per_km"].notna()]
    # escludi camminate lente
    df = df[df["pace_s_per_km"] <= 450]  # <= 7:30 min/km

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Dati insufficienti per FC vs passo", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return

    pace_min = df["pace_s_per_km"] / 60.0
    # bucket ogni 15s da 3:00 a 7:30
    edges = np.arange(3.0, 7.51, (1/6))
    labels = []
    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i+1]
        labels.append(f"{format_pace(a*60)} – {format_pace(b*60)}")

    df["bucket"] = pd.cut(pace_min, bins=edges, labels=labels, include_lowest=True, right=False)
    grp = df.groupby("bucket")["hr_bpm"].mean().reset_index()
    grp = grp.dropna(subset=["bucket", "hr_bpm"])

    # Ordina da lento (sinistra) a veloce (destra)
    order = list(reversed(labels))
    grp["bucket"] = pd.Categorical(grp["bucket"], categories=order, ordered=True)
    grp = grp.sort_values("bucket")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grp["bucket"], grp["hr_bpm"], color="#0ea5e9", marker="o", linewidth=1.6, markersize=4)
    y_min = float(np.nanmin(df["hr_bpm"])) if df["hr_bpm"].notna().any() else 0.0
    y_max = float(np.nanmax(df["hr_bpm"])) if df["hr_bpm"].notna().any() else 0.0
    ax.set_ylim(bottom=y_min - 2 if y_min else 0, top=y_max + 2 if y_max else None)
    ax.set_ylabel("FC media (bpm)")
    ax.set_xlabel("Passo (bucket 10s)")
    ax.set_title("FC vs passo (solo ritmo ≤ 7:30/km)")
    ax.grid(True, axis="y", linewidth=0.8, color="#e5e7eb")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_zones(
    df_1s: pd.DataFrame,
    thresholds: Dict[str, float],
    out_png: str,
) -> None:
    df = df_1s.copy()

    # Intensità (euristica): usa pace se disponibile, altrimenti HR
    intensity = None
    intensity_label = "Intensity (pace)"
    if df["pace_s_per_km"].notna().any() and thresholds.get("pace_thr_s_per_km"):
        intensity = thresholds["pace_thr_s_per_km"] / df["pace_s_per_km"]  # >1 = più veloce della soglia
        intensity_label = "Intensity (pace-based)"
    elif df["hr_bpm"].notna().any() and thresholds.get("hr_thr_bpm"):
        intensity = df["hr_bpm"] / thresholds["hr_thr_bpm"]
        intensity_label = "Intensity (HR-based)"

    # Potenza (euristica): power/thr
    pwr_rel = None
    if df["power_w"].notna().any() and thresholds.get("pwr_thr_w"):
        pwr_rel = df["power_w"] / thresholds["pwr_thr_w"]

    # Zone boundaries (5 zone semplici)
    bounds = [0.0, 0.80, 0.90, 1.00, 1.05, 10.0]
    zlabels = ["Z1", "Z2", "Z3", "Z4", "Z5"]

    def time_in_zones(rel: pd.Series) -> List[float]:
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
        if rel.empty:
            return [0]*5
        secs = []
        for i in range(5):
            a, b = bounds[i], bounds[i+1]
            secs.append(float(((rel >= a) & (rel < b)).sum()))
        return secs

    int_secs = time_in_zones(intensity) if intensity is not None else [0]*5
    pwr_secs = time_in_zones(pwr_rel) if pwr_rel is not None else [0]*5

    # plotting: 1 riga, 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    # sinistra: intensità
    axes[0].bar(zlabels, [s/60 for s in int_secs])
    axes[0].set_title(intensity_label)
    axes[0].set_ylabel("Minutes")
    axes[0].grid(True, which="major", linewidth=0.8, color="#e5e7eb")
    axes[0].grid(True, which="minor", linewidth=0.6, color="#f3f4f6")
    axes[0].yaxis.set_minor_locator(AutoMinorLocator(2))

    # destra: potenza
    axes[1].bar(zlabels, [s/60 for s in pwr_secs])
    axes[1].set_title("Power zones (power/thr)")
    axes[1].set_ylabel("Minutes")
    axes[1].grid(True, which="major", linewidth=0.8, color="#e5e7eb")
    axes[1].grid(True, which="minor", linewidth=0.6, color="#f3f4f6")
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

ZONE_COLORS = {
    1: "bg-gray-400",
    2: "bg-blue-500",
    3: "bg-lime-500",
    4: "bg-amber-500",
    5: "bg-red-600",
}

def compute_hr_zones(df_1s: pd.DataFrame, hr_thr: Optional[float]) -> List[Dict[str, Any]]:
    if hr_thr is None or not df_1s["hr_bpm"].notna().any():
        return []
    thr = float(hr_thr)

    # euristico su soglia HR
    edges = [
        (0.00*thr, 0.80*thr, 1, "Riscaldamento"),
        (0.80*thr, 0.90*thr, 2, "Facile"),
        (0.90*thr, 1.00*thr, 3, "Aerobico"),
        (1.00*thr, 1.05*thr, 4, "Soglia"),
        (1.05*thr, 10.0*thr, 5, "Massima"),
    ]

    total_s = float(df_1s["dt_s"].sum()) if "dt_s" in df_1s.columns else float(len(df_1s))
    hr = df_1s["hr_bpm"].to_numpy(dtype=float)
    dt = df_1s["dt_s"].to_numpy(dtype=float) if "dt_s" in df_1s.columns else np.ones_like(hr)

    rows = []
    for lo, hi, z, name in edges:
        mask = (hr >= lo) & (hr < hi)
        sec = float(np.nansum(dt[mask]))
        pct = (sec / total_s * 100.0) if total_s > 0 else 0.0
        rows.append({
            "z": z,
            "label": f"Zona {z}",
            "name": name,
            "range": f"{int(round(lo))}–{int(round(hi))} bpm" if z < 5 else f">{int(round(lo))} bpm",
            "sec": sec,
            "pct": pct,
            "color": ZONE_COLORS[z],
        })
    # Garmin: Z5 in alto
    return list(reversed(rows))

def compute_power_zones(df_1s: pd.DataFrame, pwr_thr: Optional[float]) -> List[Dict[str, Any]]:
    if pwr_thr is None or not df_1s["power_w"].notna().any():
        return []
    thr = float(pwr_thr)

    edges = [
        (0.00*thr, 0.80*thr, 1, "Facile"),
        (0.80*thr, 0.90*thr, 2, "Moderato"),
        (0.90*thr, 1.00*thr, 3, "Tempo"),
        (1.00*thr, 1.15*thr, 4, "Intervallo lungo"),
        (1.15*thr, 10.0*thr, 5, "Intervallo breve"),
    ]

    total_s = float(df_1s["dt_s"].sum()) if "dt_s" in df_1s.columns else float(len(df_1s))
    pw = df_1s["power_w"].to_numpy(dtype=float)
    dt = df_1s["dt_s"].to_numpy(dtype=float) if "dt_s" in df_1s.columns else np.ones_like(pw)

    rows = []
    for lo, hi, z, name in edges:
        mask = (pw >= lo) & (pw < hi)
        sec = float(np.nansum(dt[mask]))
        pct = (sec / total_s * 100.0) if total_s > 0 else 0.0
        rows.append({
            "z": z,
            "label": f"Zona {z}",
            "name": name,
            "range": f"{int(round(lo))}–{int(round(hi))} W" if z < 5 else f">{int(round(lo))} W",
            "sec": sec,
            "pct": pct,
            "color": ZONE_COLORS[z],
        })
    return list(reversed(rows))

def render_zones_block(zones: List[Dict[str, Any]]) -> str:
    if not zones:
        return "<div class='text-sm text-slate-500'>Dati non disponibili.</div>"
    out = []
    for z in zones:
        out.append(f"""
        <div class="space-y-1 mb-4">
          <div class="flex items-baseline justify-between">
            <div class="text-sm font-semibold">
              {z['label']} <span class="font-normal text-slate-600">· {z['range']} · {z['name']}</span>
            </div>
            <div class="text-sm tabular-nums">
              <span class="mr-2">{fmt_mmss(z['sec'])}</span>
              <span class="text-slate-500">{z['pct']:.0f}%</span>
            </div>
          </div>
          <div class="w-full h-3 rounded bg-slate-200 overflow-hidden">
            <div class="h-3 {z['color']}" style="width:{z['pct']:.2f}%;"></div>
          </div>
        </div>
        """)
    return "\n".join(out)


def write_report_html(
    template_path: str,
    out_path: str,
    base: str,
    meta: Dict[str, Any],
    thresholds: Dict[str, Any],
    summary: Dict[str, Any],
    intervals: Optional[pd.DataFrame],
    intervals_detail: Optional[pd.DataFrame],
    zones_png: str,
    overview_png: str,
    hr_pace_png: str,
    hr_zones_block: str,
    pwr_zones_block: str,
    training_effect: Dict[str, Optional[float]],
):
    with open(template_path, "r", encoding="utf-8") as f:
        tpl = f.read()

    def li(label, value):
        return f"<li><b>{label}:</b> {value}</li>"

    # SUMMARY
    summary_items = [
        li("Durata", fmt_time(summary["total_time_s"])),
        li("Distanza", f"{summary['total_dist_km']:.2f} km"),
        li("Passo medio", format_pace(summary["avg_pace_s_per_km"])),
        li("FC media / max", f"{summary['hr_avg_bpm']:.0f} / {summary['hr_max_bpm']:.0f} bpm"
           if not np.isnan(summary["hr_avg_bpm"]) else "n/a"),
        li("Potenza media / max", f"{summary['power_avg_w']:.0f} / {summary['power_max_w']:.0f} W"
           if not np.isnan(summary["power_avg_w"]) else "n/a"),
        li("Dislivello +/−", f"{summary.get('elev_gain_m', np.nan):.0f} / {summary.get('elev_loss_m', np.nan):.0f} m"
           if not np.isnan(summary.get("elev_gain_m", np.nan)) else "n/a"),
    ]
    summary_html = "\n".join(summary_items)

    # THRESHOLDS
    thr_items = []
    if thresholds.get("hr_thr_bpm"):
        thr_items.append(li("HR thr", f"{thresholds['hr_thr_bpm']:.0f} bpm"))
    if thresholds.get("pwr_thr_w"):
        thr_items.append(li("Power thr", f"{thresholds['pwr_thr_w']:.0f} W"))
    if thresholds.get("pace_thr_s_per_km"):
        thr_items.append(li("Pace thr", format_pace(thresholds["pace_thr_s_per_km"])))
    if thresholds.get("weight_kg"):
        thr_items.append(li("Peso", f"{thresholds['weight_kg']:.1f} kg"))
    thresholds_html = "\n".join(thr_items)

    # ---- Nerd metric gauges (euristici) ----
    def gauge_bands(
        label: str,
        value_str: str,
        value: Optional[float],
        *,
        vmin: float,
        vmax: float,
        ticks: List[str],
        bands: List[Tuple[float, float, str]],  # (from,to, tailwind_class)
        note: str
    ) -> str:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return f"""
            <div class="p-4 rounded-lg border bg-slate-50">
              <div class="flex items-baseline justify-between">
                <div class="font-semibold">{label}</div>
                <div class="text-sm text-slate-500">n/a</div>
              </div>
              <div class="text-xs text-slate-500 mt-1">{note}</div>
            </div>
            """

        v = float(value)
        v = max(vmin, min(vmax, v))
        span = (vmax - vmin) if vmax != vmin else 1.0
        pos = (v - vmin) / span * 100.0

        segs = []
        for a, b, cls in bands:
            a = max(vmin, a)
            b = min(vmax, b)
            if b <= a:
                continue
            left = (a - vmin) / span * 100.0
            width = (b - a) / span * 100.0
            segs.append(f"<div class='absolute inset-y-0 {cls}' style='left:{left:.4f}%; width:{width:.4f}%;'></div>")

        ticks_html = "".join([f"<div><span class='tabular-nums'>{t}</span></div>" for t in ticks])

        return f"""
        <div class="p-4 rounded-lg border bg-white">
          <div class="flex items-baseline justify-between">
            <div class="font-semibold">{label}</div>
            <div class="text-sm text-slate-800">{value_str}</div>
          </div>

          <div class="relative mt-3 h-3 rounded-full overflow-hidden bg-slate-200">
            {''.join(segs)}
            <div class="absolute -top-0.5" style="left:{pos:.2f}%; transform:translateX(-50%);">
              <div class="w-4 h-4 rounded-full bg-white border-2 border-slate-900 shadow"></div>
            </div>
          </div>

          <div class="flex justify-between text-[11px] text-slate-500 mt-1">{ticks_html}</div>
          <div class="text-xs text-slate-500 mt-2">{note}</div>
        </div>
        """

    drift = summary.get("drift", {})
    drift_pct = drift.get("hr_drift_pct", np.nan) if isinstance(drift, dict) else np.nan

    eff = summary.get("efficiency", {}) if isinstance(summary.get("efficiency", {}), dict) else {}
    hr_per_speed = eff.get("hr_per_speed", np.nan)
    pwr_per_speed = eff.get("power_per_speed", np.nan)
    wkg = eff.get("avg_w_per_kg_running", np.nan)

    nerd_cards = []
    nerd_cards.append(gauge_bands(
        "Cardiac drift",
        (f"{drift_pct:.1f}%" if not np.isnan(drift_pct) else "n/a"),
        (float(drift_pct) if not np.isnan(drift_pct) else None),
        vmin=0.0, vmax=10.0,
        ticks=["0%", "3%", "6%", "10%+"],
        bands=[
            (0, 2, "bg-violet-600"),  # eccellente
            (2, 4, "bg-blue-600"),
            (4, 6, "bg-green-500"),
            (6, 8, "bg-amber-500"),
            (8,10, "bg-red-600"),
        ],
        note="Più basso = meglio (a parità di intensità). >~5% spesso indica fatica/caldo/idrat. o base da consolidare."
    ))
    nerd_cards.append(gauge_bands(
    "HR per speed",
    (f"{hr_per_speed:.2f} bpm/(m/s)" if not np.isnan(hr_per_speed) else "n/a"),
    (float(hr_per_speed) if not np.isnan(hr_per_speed) else None),
    vmin=25.0, vmax=60.0,
    ticks=["25", "~42", "60"],
    bands=[
        (25, 33, "bg-violet-600"),
        (33, 40, "bg-blue-600"),
        (40, 47, "bg-green-500"),
        (47, 54, "bg-amber-500"),
        (54, 60, "bg-red-600"),
    ],
    note="Più basso = più efficiente (dipende da caldo, pendenze, stanchezza). Confronta allenamenti simili."
))
    nerd_cards.append(gauge_bands(
        "Power per speed",
        (f"{pwr_per_speed:.1f} W/(m/s)" if not np.isnan(pwr_per_speed) else "n/a"),
        (float(pwr_per_speed) if not np.isnan(pwr_per_speed) else None),
        vmin=120.0, vmax=260.0,
        ticks=["120", "~190", "260"],
        bands=[
            (120, 150, "bg-violet-600"),
            (150, 175, "bg-blue-600"),
            (175, 205, "bg-green-500"),
            (205, 235, "bg-amber-500"),
            (235, 260, "bg-red-600"),
        ],
        note="Più basso = più economico (ma vento/pendenza/superficie possono falsare)."
    ))
    nerd_cards.append(gauge_bands(
        "W/kg medio (running)",
        (f"{wkg:.2f} W/kg" if not np.isnan(wkg) else "n/a"),
        (float(wkg) if not np.isnan(wkg) else None),
        vmin=2.0, vmax=6.0,
        ticks=["2.0", "4.0", "6.0"],
        bands=[
            (2.0, 2.8, "bg-red-600"),
            (2.8, 3.4, "bg-amber-500"),
            (3.4, 4.2, "bg-green-500"),
            (4.2, 5.0, "bg-blue-600"),
            (5.0, 6.0, "bg-violet-600"),
        ],
        note="Più alto = meglio (speed>1.5 m/s). Utile per confronti interni e confronti tra allenamenti."
    ))

    nerd_html = f"""
    <div class="grid md:grid-cols-2 gap-4">
      {''.join(nerd_cards)}
    </div>
    """

    def te_band(te: Optional[float]) -> str:
        # colori come richiesto:
        # 0.0-0.9 dark gray, 1.0-1.9 light gray, 2.0-2.9 blue, 3.0-3.9 green, 4.0-4.9 amber, 5.0 red
        if te is None or (isinstance(te, float) and (np.isnan(te) or np.isinf(te))):
            return "n/a"
        v = float(te)
        if v < 1.0: return "bg-slate-700"
        if v < 2.0: return "bg-slate-300"
        if v < 3.0: return "bg-blue-500"
        if v < 4.0: return "bg-green-500"
        if v < 5.0: return "bg-amber-500"
        return "bg-red-600"

    def te_text(te: Optional[float], kind: str) -> str:
        # kind: "aerobic" | "anaerobic"
        if te is None or (isinstance(te, float) and (np.isnan(te) or np.isinf(te))):
            return "n/a"
        v = float(te)
        if v < 1.0: return "Nessun beneficio."
        if v < 2.0: return "Beneficio minimo."
        if v < 3.0: return "Mantiene l’attività fitness " + ("aerobica." if kind=="aerobic" else "anaerobica.")
        if v < 4.0: return "Influisce sul livello di fitness " + ("aerobico." if kind=="aerobic" else "anaerobico.")
        if v < 5.0: return "Influisce notevolmente sul livello di fitness " + ("aerobico." if kind=="aerobic" else "anaerobico.")
        return "Intensità troppo alta e potenzialmente dannosa senza un adeguato tempo di recupero."

    def te_bar(label: str, te: Optional[float], kind: str) -> str:
        val = "n/a" if te is None or (isinstance(te, float) and (np.isnan(te) or np.isinf(te))) else f"{float(te):.1f}"
        color = te_band(te)
        desc = te_text(te, kind)
        # barra 0..5: posizione marker in %
        if te is None or (isinstance(te, float) and (np.isnan(te) or np.isinf(te))):
            marker_left = 0
        else:
            marker_left = int(round(max(0.0, min(5.0, float(te))) / 5.0 * 100))

        # segmenti colorati come Garmin
        return f"""
        <div class="p-4 rounded-lg border bg-white">
          <div class="flex items-baseline justify-between">
            <div class="font-semibold">{label}</div>
            <div class="text-sm text-slate-700">{val}</div>
          </div>

		  <div class="relative">
            <div class="relative mt-3 h-3 rounded-full overflow-hidden bg-slate-200">
              <div class="absolute inset-y-0 left-0 w-[18%] bg-slate-700"></div>
              <div class="absolute inset-y-0 left-[18%] w-[20%] bg-slate-300"></div>
              <div class="absolute inset-y-0 left-[38%] w-[20%] bg-blue-500"></div>
              <div class="absolute inset-y-0 left-[58%] w-[20%] bg-green-500"></div>
              <div class="absolute inset-y-0 left-[78%] w-[20%] bg-amber-500"></div>
              <div class="absolute inset-y-0 left-[98%] w-[2%] bg-red-600"></div>            
            </div>
		  
		    <div class="absolute -top-0.5" style="left:{marker_left}%; transform:translateX(-50%);">
              <div class="w-4 h-4 rounded-full bg-white border-2 border-slate-900 shadow"></div>
            </div>
		  </div>

          <div class="text-xs text-slate-500 mt-2">{desc}</div>
        </div>
        """

    a_te = None
    an_te = None
    if isinstance(training_effect, dict):
        a_te = training_effect.get("aerobic_te", None)
        an_te = training_effect.get("anaerobic_te", None)

    training_effect_html = f"""
    <div class="grid md:grid-cols-2 gap-4">
      {te_bar("Beneficio aerobico", a_te, "aerobic")}
      {te_bar("Beneficio anaerobico", an_te, "anaerobic")}
    </div>
    """

    def row_bg_class(phase: str) -> str:
        # usa le tue label esatte (Riscaldamento/Corsa/Recupero/Defaticamento)
        p = (phase or "").strip().lower()
        if "riscald" in p: return "bg-red-50"
        if p == "corsa": return "bg-blue-50"
        if "recuper" in p: return "bg-slate-50"
        if "defatic" in p: return "bg-green-50"
        if p == "totale": return "bg-slate-200"
        return "bg-white"

    if intervals_detail is not None and not intervals_detail.empty:
        df = intervals_detail.copy()

        # drop Durata_s
        if "Durata_s" in df.columns:
            df = df.drop(columns=["Durata_s"])

        # format
        if "Ripetuta" in df.columns:
            df["Ripetuta"] = df["Ripetuta"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
        for c in ["FC_med","FC_min","FC_max"]:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")
        for c in ["P_med","P_min","P_max"]:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")
        if "Wkg_med" in df.columns:
            df["Wkg_med"] = df["Wkg_med"].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        if "Distanza_km" in df.columns:
            df["Distanza_km"] = df["Distanza_km"].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")

        # build table manually
        cols = list(df.columns)

        head = "".join([f"<th class='text-left font-semibold p-2 border-b border-slate-200 whitespace-nowrap'>{c}</th>" for c in cols])
        body_rows = []
        for _, r in df.iterrows():
            phase = str(r.get("Fase", ""))
            bg = row_bg_class(phase)
            tds = "".join([f"<td class='p-2 border-b border-slate-100 whitespace-nowrap'>{r.get(c,'')}</td>" for c in cols])
            body_rows.append(f"<tr class='{bg}'>{tds}</tr>")

        intervals_html = f"""
        <table class="min-w-full text-sm border border-slate-200 rounded-lg overflow-hidden">
          <thead class="bg-slate-50"><tr>{head}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
        """
    else:
        intervals_html = "<p class='text-sm text-slate-500'>Nessun intervallo disponibile.</p>"


    subtitle = f"Base: {meta.get('activity_base','')} · FIT: {meta.get('fit_file','')} · GPX: {meta.get('gpx_file','')}"
    html = (
        tpl
        .replace("{{TITLE}}", base)
        .replace("{{DATE}}", meta.get("activity_start_local",""))
        .replace("{{CATEGORY}}", meta.get("category",""))
        .replace("{{RPE}}", str(meta.get("rpe","n/a")) if meta.get("rpe","") != "" else "n/a")
        .replace("{{SUBTITLE}}", subtitle)
        .replace("{{OVERVIEW_PNG}}", overview_png)
        .replace("{{HR_PACE_PNG}}", hr_pace_png)
        .replace("{{ZONES_PNG}}", zones_png)
        .replace("{{SUMMARY_LIST}}", summary_html)
        .replace("{{THRESHOLDS_LIST}}", thresholds_html)
        .replace("{{NERD_METRICS}}", nerd_html)
        .replace("{{INTERVALS_TABLE}}", intervals_html)
        .replace("{{HR_ZONES_BLOCK}}", hr_zones_block)
        .replace("{{PWR_ZONES_BLOCK}}", pwr_zones_block)
        .replace("{{TRAINING_EFFECT}}", training_effect_html)
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def write_ai_txt(
    summary: Dict[str, Any],
    thresholds: Dict[str, Any],
    meta: Dict[str, Any],
    phase_stats: Optional[pd.DataFrame],
    intervals: Optional[pd.DataFrame],
    path: str
):
    lines = []

    lines.append("### META ###\n")
    lines.append(json.dumps(meta, indent=2, ensure_ascii=False))
    lines.append("\n\n### THRESHOLDS ###\n")
    lines.append(json.dumps(thresholds, indent=2, ensure_ascii=False))
    lines.append("\n\n### SUMMARY ###\n")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))

    if phase_stats is not None and not phase_stats.empty:
        lines.append("\n\n### REPEATS_REPORT ###\n")
        lines.append(phase_stats.to_csv(index=False))

    if intervals is not None and not intervals.empty:
        lines.append("\n\n### INTERVALS_USED ###\n")
        lines.append(intervals.to_csv(index=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    # POSIZIONALE obbligatorio: base name, es "20251215-SOG"
    ap.add_argument("base", type=str, help="Base name (no extension), e.g. 20251215-SOG")

    # input directory (dove stanno i file)
    ap.add_argument("--indir", type=str, default=".", help="Input directory containing .fit/.gpx/.csv")

    # output directory root
    ap.add_argument("--outdir", type=str, default="out_analysis")
    
    # category
    ap.add_argument("--cat", type=str, default=None)

    # soglie / opzioni
    ap.add_argument("--hr_thr", type=float, default=172)
    ap.add_argument("--pwr_thr", type=float, default=376)
    ap.add_argument("--pace_thr", type=str, default="5:36")
    ap.add_argument("--weight", type=float, default=None)
    ap.add_argument("--rpe", type=float, default=None)
    ap.add_argument("--no_plots", action="store_true")

    args = ap.parse_args()

    base = args.base.strip()
    in_dir = Path(args.indir)

    # Deriva automaticamente i nomi file
    fit_path = in_dir / f"{base}.fit"
    gpx_path = in_dir / f"{base}.gpx"
    gc_csv_path = in_dir / f"{base}.csv"  # Garmin intervals export (opzionale)

    # Esistenza: servono sia FIT che GPX
    has_fit = fit_path.exists()
    has_gpx = gpx_path.exists()

    if not (has_fit and has_gpx):
        raise SystemExit(
            f"No input. Both files are required:\n"
            f"- {fit_path}\n- {gpx_path}"
        )

    # CSV Garmin opzionale: se non c'è, si va su laps/autodetect
    has_gc_csv = gc_csv_path.exists()

    CAT_ALLOWED = {"REC","BAS","TMP","SOG","VO2","CAN","SPR"}
    cat = args.cat.strip().upper()
    if cat not in CAT_ALLOWED:
        raise SystemExit(f"--cat must be one of: {sorted(CAT_ALLOWED)}")

    # Parse thresholds
    hr_thr_default = ap.get_default("hr_thr")
    pwr_thr_default = ap.get_default("pwr_thr")
    pace_thr_default = ap.get_default("pace_thr")

    mm, ss = args.pace_thr.split(":")
    thresholds: Dict[str, float] = {
        "hr_thr_bpm": float(args.hr_thr),
        "pwr_thr_w": float(args.pwr_thr),
        "pace_thr_s_per_km": float(int(mm) * 60 + int(ss)),
    }

    # Weight: try FIT first (if not passed)
    if args.weight is not None:
        thresholds["weight_kg"] = float(args.weight)
    elif has_fit:
        w = try_extract_weight_from_fit(fit_path)
        if w is not None:
            thresholds["weight_kg"] = float(w)

    # Override thresholds from FIT if user kept defaults and FIT provides them
    if has_fit:
        fit_thr = try_extract_thresholds_from_fit(str(fit_path))
        if args.hr_thr == hr_thr_default and "hr_thr_bpm" in fit_thr:
            thresholds["hr_thr_bpm"] = float(fit_thr["hr_thr_bpm"])
        if args.pwr_thr == pwr_thr_default and "pwr_thr_w" in fit_thr:
            thresholds["pwr_thr_w"] = float(fit_thr["pwr_thr_w"])
        if args.pace_thr == pace_thr_default and "pace_thr_s_per_km" in fit_thr:
            thresholds["pace_thr_s_per_km"] = float(fit_thr["pace_thr_s_per_km"])

    # Read data
    df_fit = parse_fit_records(str(fit_path)) if fit_path else pd.DataFrame()
    df_gpx = parse_gpx(str(gpx_path)) if gpx_path else pd.DataFrame()

    # Prefer FIT; fill missing geo from GPX if needed
    if not df_fit.empty:
        df = df_fit.copy()
        if df["lat"].isna().all() and not df_gpx.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                df_gpx[["timestamp","lat","lon","alt_m"]].sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=2),
                suffixes=("","_gpx")
            )
            for col in ["lat","lon","alt_m"]:
                if f"{col}_gpx" in df.columns:
                    df[col] = df[col].fillna(df[f"{col}_gpx"])
                    df.drop(columns=[f"{col}_gpx"], inplace=True)
    else:
        df = df_gpx.copy()

    if df.empty:
        raise SystemExit("No data parsed. Check files.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = compute_distance_and_speed(df)
    df_1s = resample_to_1s(df)
	
    activity_date = df_1s["timestamp"].iloc[0].strftime("%Y%m%d")  # UTC
    prefix = f"{activity_date}-{cat}"
	
    activity_start = df_1s["timestamp"].iloc[0].strftime("%Y-%m-%d_%H-%M-%S")
    activity_outdir = os.path.join(
        args.outdir,
        prefix)
    os.makedirs(activity_outdir, exist_ok=True)

    summary = summarize(df_1s, thresholds)
    intervals_used = pd.DataFrame()
	
    hr_z = compute_hr_zones(df_1s, thresholds.get("hr_thr_bpm"))
    pw_z = compute_power_zones(df_1s, thresholds.get("pwr_thr_w"))
    hr_zones_block = render_zones_block(hr_z)
    pwr_zones_block = render_zones_block(pw_z)

    laps = pd.DataFrame()
    if fit_path:
        laps = parse_fit_laps(str(fit_path))

    # Gets training effect from FIT if available
    training_effect = {"aerobic_te": None, "anaerobic_te": None}
    if has_fit:
        training_effect = try_extract_training_effect_from_fit(str(fit_path))

    # 1) Prefer Garmin CSV if provided
    if has_gc_csv:
        intervals_gc = read_garmin_intervals_csv(str(gc_csv_path))
        intervals_used = intervals_seconds_to_timestamps(df_1s, intervals_gc)
        intervals_used.to_csv(out_path(activity_outdir, prefix, "intervals_gc", "csv"), index=False)

    # 2) Else fallback to FIT laps (se vuoi mantenerlo)
    elif fit_path:
        laps = parse_fit_laps(str(fit_path))
        intervals_official = build_intervals_from_laps(df_1s, laps)  # se ce l'hai ancora
        intervals_used = intervals_official

    # 3) Else fallback to autodetect
    else:
        df_labeled, blocks_df = detect_intervals(df_1s, thresholds)
        intervals_used = blocks_df

    # Plot overview + zones nella cartella attività
    overview_png_name = os.path.basename(out_path(activity_outdir, prefix, "overview", "png"))
    zones_png_name = os.path.basename(out_path(activity_outdir, prefix, "zones", "png"))
    hr_pace_png_name = os.path.basename(out_path(activity_outdir, prefix, "hr_pace", "png"))

    if not args.no_plots:
        plot_overview(
            df_1s,
            intervals_used,
            out_path(activity_outdir, prefix, "overview", "png"),
            #target_pace_min_s_per_km=target_min,
            #target_pace_max_s_per_km=target_max,
        )
        plot_zones(
            df_1s,
            thresholds,
            out_path(activity_outdir, prefix, "zones", "png"),
        )
        plot_hr_vs_pace_buckets(
            df_1s,
            out_path(activity_outdir, prefix, "hr_pace", "png"),
        )

    # Report ripetute
    phase_stats = compute_phase_stats(df_1s, intervals_used) if not intervals_used.empty else pd.DataFrame()
    if not phase_stats.empty:
        phase_stats.to_csv(out_path(activity_outdir, prefix, "repeats_report", "csv"), index=False)

    md_path = out_path(activity_outdir, prefix, "analysis_summary", "md")
    json_path = out_path(activity_outdir, prefix, "analysis_summary", "json")

    activity_start_ts = df_1s["timestamp"].iloc[0]
	
    # Salva sempre records_1s
    records_path = out_path(activity_outdir, prefix, "records_1s", "csv")
    df_1s.to_csv(records_path, index=False)

    # Salva anche un "intervals_used" normalizzato (se vuoi)
    if isinstance(intervals_used, pd.DataFrame) and not intervals_used.empty:
        intervals_path = out_path(activity_outdir, prefix, "intervals_used", "csv")
        intervals_used.to_csv(intervals_path, index=False)

	
    write_summary_md(
        summary=summary,
        thresholds=thresholds,
        intervals=intervals_used if isinstance(intervals_used, pd.DataFrame) else None,
        activity_start_ts=activity_start_ts,
        path=md_path,
		rpe=args.rpe
    )

    local_tz = dateutil_tz.tzlocal()
	
    meta = {
        "activity_start_utc": df_1s["timestamp"].iloc[0].isoformat(),
        "activity_start_local": df_1s["timestamp"].iloc[0].tz_convert(local_tz).isoformat()
            if df_1s["timestamp"].iloc[0].tzinfo else df_1s["timestamp"].iloc[0].isoformat(),
        "category": cat,
        "activity_base": base,
        "fit_file": str(fit_path),
        "gpx_file": str(gpx_path),
        "gc_csv": str(gc_csv_path) if has_gc_csv else "not available",
		"rpe": args.rpe if not args.rpe is None else ""

    }

    out_json = {
        "meta": meta,
        "thresholds": thresholds,
        "summary": summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
		
    ai_path = out_path(activity_outdir, prefix, "ai", "txt")

    write_ai_txt(
        summary=summary,
        thresholds=thresholds,
        meta=meta,
        phase_stats=phase_stats,
        intervals=intervals_used,
        path=ai_path
    )
	
    html_path = out_path(activity_outdir, prefix, "report", "html")

    # Dettaglio intervalli stile Garmin
    intervals_detail = None
    if isinstance(intervals_used, pd.DataFrame) and not intervals_used.empty and "start" in intervals_used.columns and "end" in intervals_used.columns:
        intervals_detail = compute_interval_details(df_1s, intervals_used, thresholds.get("weight_kg"))

    html_path = out_path(activity_outdir, prefix, "report", "html")
    write_report_html(
        template_path=os.path.join(os.path.dirname(__file__), "report_template.html"),
        out_path=html_path,
        base=prefix,
        meta=meta,
        thresholds=thresholds,
        summary=summary,
        intervals=intervals_used,
        intervals_detail=intervals_detail,
        zones_png=zones_png_name,
        overview_png=overview_png_name,
        hr_pace_png=hr_pace_png_name,
        hr_zones_block=hr_zones_block,
        pwr_zones_block=pwr_zones_block,
        training_effect=training_effect
    )

    print("Saved to:", activity_outdir)
    print("-", os.path.basename(records_path))
    print("-", os.path.basename(md_path))
    print("-", os.path.basename(json_path))
    print("-", os.path.basename(html_path))
    print("-", os.path.basename(ai_path))
    if not args.no_plots:
        print("-", os.path.basename(out_path(activity_outdir, prefix, "overview", "png")))
    if not phase_stats.empty:
        print("-", os.path.basename(out_path(activity_outdir, prefix, "repeats_report", "csv")))


if __name__ == "__main__":
    main()
