#!/usr/bin/env python3
"""
Evaluate trained LR model on a NEW video (no features CSV).

- Recomputes per-event features from video + ground_truth.csv
- Uses the trained shot_lr_model.joblib for classification

Edit VIDEO_PATH, GROUND_TRUTH_CSV, MODEL_PATH, and OUTPUT_CSV below,
then run:
    python eval_model_on_video.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ---------- USER CONFIG ----------
VIDEO_PATH = "test_video.mp4"
GROUND_TRUTH_CSV = "ground_truth/test_video.csv"
MODEL_PATH = "shot_lr_model.joblib"
OUTPUT_CSV = "event_features_with_preds.csv"

# Feature extraction params (same as training)
WINDOW = 10
FRAME_STRIDE = 1
DOWNSCALE_WIDTH = 640
DOWNSCALE_HEIGHT = 360
BLOCK_ROWS = 8
BLOCK_COLS = 14
BLOCK_DIFF_THRESH = 5.0
# ----------------------------------


# ---------- MODEL LOADING ----------

def load_model(path: str):
    bundle = joblib.load(path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    feat_names = bundle["features"]
    return model, scaler, imputer, feat_names


# ---------- FEATURE PIPELINE (same as training) ----------

def read_frame(cap, total_frames, idx):
    idx = idx * FRAME_STRIDE
    if idx < 0 or idx >= total_frames:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    if DOWNSCALE_WIDTH and DOWNSCALE_HEIGHT:
        frame = cv2.resize(frame, (DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT))
    return frame


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def hist_feature(frame, bins=32):
    gray = to_gray(frame)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def edge_map(frame):
    gray = to_gray(frame)
    return cv2.Canny(gray, 100, 200)


def edge_count(frame):
    e = edge_map(frame)
    return int(np.count_nonzero(e))


def phash(frame, hash_size=8, highfreq_factor=4):
    img = to_gray(frame)
    img = cv2.resize(img, (hash_size * highfreq_factor,
                           hash_size * highfreq_factor))
    img = np.float32(img)
    dct = cv2.dct(img)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    diff = dct_low > med
    return diff.flatten()


def phash_dist(f1, f2):
    h1 = phash(f1)
    h2 = phash(f2)
    return int(np.count_nonzero(h1 != h2))


def block_means(gray, rows=BLOCK_ROWS, cols=BLOCK_COLS):
    h, w = gray.shape
    bh = h // rows
    bw = w // cols
    means = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            means[r, c] = float(patch.mean())
    return means


def block_change_features(f1, f2, diff_thresh=BLOCK_DIFF_THRESH):
    g1 = to_gray(f1)
    g2 = to_gray(f2)
    m1 = block_means(g1)
    m2 = block_means(g2)
    diff = np.abs(m2 - m1)
    mask = diff > diff_thresh
    total_blocks = m1.size
    num_high = int(mask.sum())
    frac_high = num_high / total_blocks if total_blocks > 0 else np.nan
    if num_high == 0:
        return frac_high, np.nan, np.nan
    rs, cs = np.where(mask)
    cx = np.mean(cs / (BLOCK_COLS - 1)) if BLOCK_COLS > 1 else 0.5
    cy = np.mean(rs / (BLOCK_ROWS - 1)) if BLOCK_ROWS > 1 else 0.5
    return frac_high, cx, cy


def compute_frame_features(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_eff = total_frames // FRAME_STRIDE

    hist_diff = np.full(n_eff, np.nan, dtype=np.float32)
    edge_counts = np.full(n_eff, np.nan, dtype=np.float32)
    phash_d = np.full(n_eff, np.nan, dtype=np.float32)
    frac_high = np.full(n_eff, np.nan, dtype=np.float32)
    cx = np.full(n_eff, np.nan, dtype=np.float32)

    prev = None
    for i in range(n_eff):
        frame = read_frame(cap, total_frames, i)
        if frame is None:
            prev = None
            continue
        edge_counts[i] = edge_count(frame)
        if prev is not None:
            hist_diff[i] = float(np.sum(np.abs(hist_feature(prev) - hist_feature(frame))))
            phash_d[i] = phash_dist(prev, frame)
            fh, cx_i, _ = block_change_features(prev, frame)
            frac_high[i] = fh
            cx[i] = cx_i
        prev = frame

    cap.release()
    return hist_diff, edge_counts, phash_d, frac_high, cx


# ---------- GROUND TRUTH & EVENTS ----------

def load_ground_truth(path):
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["Frame_Index"])
            etype = row["type"].strip().lower()
            events.append((frame, etype))
    return events


def merge_intervals(events):
    merged = []
    pending_start = {}
    for frame, etype in sorted(events, key=lambda x: x[0]):
        eff = frame // FRAME_STRIDE
        if etype in ("cut", "fadein", "fadeout"):
            merged.append({"kind": etype, "start": eff, "end": eff})
        elif etype.endswith("_start"):
            base = etype.rsplit("_", 1)[0]
            pending_start[base] = eff
        elif etype.endswith("_end"):
            base = etype.rsplit("_", 1)[0]
            if base in pending_start:
                merged.append({"kind": base, "start": pending_start[base], "end": eff})
                del pending_start[base]
    return merged


def center_of_event(ev):
    return ev["start"] if ev["start"] == ev["end"] else (ev["start"] + ev["end"]) // 2


def summarize_window(idx, hist_diff, edge_counts, phash_d, frac_high, cx, window=WINDOW):
    n = len(hist_diff)

    def clip(a, b):
        return max(0, a), min(n, b)

    s_pre, e_pre = clip(idx - window, idx)
    s_post, e_post = clip(idx + 1, idx + window + 1)
    s_width, e_width = clip(idx - 5, idx + 6)

    max_hist = np.nanmax(hist_diff[s_width:e_width])
    thr = 0.5 * max_hist
    width_hist = int(np.sum(hist_diff[s_width:e_width] > thr))

    pre_edge = np.nanmean(edge_counts[s_pre:e_pre])
    post_edge = np.nanmean(edge_counts[s_post:e_post])
    pre_frac = np.nanmean(frac_high[s_pre:e_pre])
    post_frac = np.nanmean(frac_high[s_post:e_post])

    s_cx1, e_cx1 = clip(idx - 5, idx - 1)
    s_cx2, e_cx2 = clip(idx + 1, idx + 6)
    cx_pre = np.nanmean(cx[s_cx1:e_cx1])
    cx_post = np.nanmean(cx[s_cx2:e_cx2])
    cx_trend = cx_post - cx_pre

    return {
        "center": idx * FRAME_STRIDE,
        "max_hist": max_hist,
        "width_hist": width_hist,
        "pre_edge": pre_edge,
        "post_edge": post_edge,
        "pre_frac": pre_frac,
        "post_frac": post_frac,
        "pre_hist": np.nanmean(hist_diff[s_pre:e_pre]),
        "post_hist": np.nanmean(hist_diff[s_post:e_post]),
        "cx_trend": cx_trend,
        "phash": phash_d[idx],
    }


# ---------- EVAL PIPELINE ----------

def compute_metrics(df: pd.DataFrame):
    kinds = df["kind"].unique()
    rows = []
    for k in kinds:
        tp = ((df["kind"] == k) & (df["pred_kind"] == k)).sum()
        fp = ((df["pred_kind"] == k) & (df["kind"] != k)).sum()
        fn = ((df["kind"] == k) & (df["pred_kind"] != k)).sum()
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        rows.append({"kind": k, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def main():
    print("Computing per-frame features...")
    hist_diff, edge_counts, phash_d, frac_high, cx = compute_frame_features(VIDEO_PATH)

    print("Loading ground truth...")
    events_raw = load_ground_truth(GROUND_TRUTH_CSV)
    events = merge_intervals(events_raw)
    print(f"Total events: {len(events)}")

    # Build event-level feature rows
    rows = []
    for ev in events:
        center_eff = center_of_event(ev)
        feat = summarize_window(center_eff, hist_diff, edge_counts, phash_d, frac_high, cx)
        feat["kind"] = ev["kind"]
        rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No events to evaluate.")
        return

    # Derived features used in training
    df["edge_ratio_post_pre"] = df["post_edge"] / (df["pre_edge"] + 1e-6)
    df["edge_ratio_pre_post"] = df["pre_edge"] / (df["post_edge"] + 1e-6)
    df["frac_mean"] = (df["pre_frac"] + df["post_frac"]) / 2.0

    model, scaler, imputer, feat_names = load_model(MODEL_PATH)
    print(f"Using features: {feat_names}")

    X = df[feat_names].values.astype(np.float32)
    X = imputer.transform(X)
    X_sc = scaler.transform(X)
    y_true = df["kind"].values
    y_pred = model.predict(X_sc)

    df["pred_kind"] = y_pred
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Per-event predictions written to {OUTPUT_CSV}")

    print("\n=== Metrics on new video ===")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
