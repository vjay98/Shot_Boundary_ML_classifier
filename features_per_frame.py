import os
import csv
import cv2
import numpy as np

# ------------- CONFIG -------------
VIDEO_PATH = "video.mp4"
GROUND_TRUTH_CSV = "ground_truth.csv"
OUTPUT_CSV = "event_features.csv"   # change per stream
WINDOW = 10        # frames before/after center for summaries
BLOCK_ROWS = 8
BLOCK_COLS = 14
BLOCK_DIFF_THRESH = 5.0
# ---------------------------------


# ---------- Video access ----------
cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frame(idx):
    if idx < 0 or idx >= TOTAL_FRAMES:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------- Per-frame features ----------

def hist_feature(frame, bins=32):
    gray = to_gray(frame)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def edge_map(frame):
    gray = to_gray(frame)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def edge_count(frame):
    e = edge_map(frame)
    return int(np.count_nonzero(e))


def edge_change_ratio(f1, f2):
    e1 = edge_map(f1).astype(bool)
    e2 = edge_map(f2).astype(bool)
    lost = np.logical_and(e1, np.logical_not(e2))
    gained = np.logical_and(np.logical_not(e1), e2)
    denom1 = np.count_nonzero(e1) or 1
    denom2 = np.count_nonzero(e2) or 1
    ecr1 = np.count_nonzero(lost) / denom1
    ecr2 = np.count_nonzero(gained) / denom2
    return float(max(ecr1, ecr2))


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
    h1 = phash(f1, hash_size=8, highfreq_factor=4)
    h2 = phash(f2, hash_size=8, highfreq_factor=4)
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
    """
    Returns:
      frac_high: fraction of blocks whose mean intensity changed above threshold.
      cx: centroid x of high-change blocks in [0,1] (NaN if none).
      cy: centroid y of high-change blocks in [0,1] (NaN if none).
    """
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


def compute_frame_features():
    """
    Compute per-frame sequences:
      hist_diff[i] = hist diff between frame i-1 and i
      edge_count[i]
      ecr[i]        = edge change ratio between i-1 and i
      phash_diff[i]
      frac_high[i], cx[i]
    """
    n = TOTAL_FRAMES
    hist_diff = np.full(n, np.nan, dtype=np.float32)
    edge_counts = np.full(n, np.nan, dtype=np.float32)
    ecr = np.full(n, np.nan, dtype=np.float32)
    phash_d = np.full(n, np.nan, dtype=np.float32)
    frac_high = np.full(n, np.nan, dtype=np.float32)
    cx = np.full(n, np.nan, dtype=np.float32)

    prev_frame = None
    for i in range(n):
        frame = read_frame(i)
        if frame is None:
            prev_frame = None
            continue

        edge_counts[i] = edge_count(frame)

        if prev_frame is not None:
            hist_diff[i] = float(
                np.sum(np.abs(hist_feature(prev_frame) - hist_feature(frame)))
            )
            ecr[i] = edge_change_ratio(prev_frame, frame)
            phash_d[i] = phash_dist(prev_frame, frame)
            fh, cx_i, cy_i = block_change_features(prev_frame, frame)
            frac_high[i] = fh
            cx[i] = cx_i

        prev_frame = frame

    return hist_diff, edge_counts, ecr, phash_d, frac_high, cx


# ---------- Ground truth & centers ----------

def load_ground_truth(path):
    events = []
    with open(path, newline='') as f:
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
        if etype == "cut":
            merged.append({"kind": "cut", "start": frame, "end": frame})
        elif etype in ("fadein", "fadeout"):
            merged.append({"kind": etype, "start": frame, "end": frame})
        elif etype.endswith("_start"):
            base = etype.rsplit("_", 1)[0]  # e.g. dissolve, wipe
            pending_start[base] = frame
        elif etype.endswith("_end"):
            base = etype.rsplit("_", 1)[0]
            if base in pending_start:
                merged.append({
                    "kind": base,
                    "start": pending_start[base],
                    "end": frame
                })
                del pending_start[base]
    return merged


def center_of_event(ev):
    if ev["start"] == ev["end"]:
        return ev["start"]
    return (ev["start"] + ev["end"]) // 2


# ---------- Event-level summary ----------

def summarize_window(idx, hist_diff, edge_count, ecr,
                     phash_diff, frac_high, cx, window=WINDOW):
    n = len(hist_diff)

    def clip(a, b):
        return max(0, a), min(n, b)

    s_pre, e_pre = clip(idx - window, idx)
    s_post, e_post = clip(idx + 1, idx + window + 1)
    s_width, e_width = clip(idx - 5, idx + 6)

    max_hist = np.nanmax(hist_diff[s_width:e_width])
    thr = 0.5 * max_hist
    width_hist = int(np.sum(hist_diff[s_width:e_width] > thr))

    pre_edge = np.nanmean(edge_count[s_pre:e_pre])
    post_edge = np.nanmean(edge_count[s_post:e_post])
    pre_frac = np.nanmean(frac_high[s_pre:e_pre])
    post_frac = np.nanmean(frac_high[s_post:e_post])
    pre_hist = np.nanmean(hist_diff[s_pre:e_pre])
    post_hist = np.nanmean(hist_diff[s_post:e_post])

    s_cx1, e_cx1 = clip(idx - 5, idx - 1)
    s_cx2, e_cx2 = clip(idx + 1, idx + 6)
    cx_pre = np.nanmean(cx[s_cx1:e_cx1])
    cx_post = np.nanmean(cx[s_cx2:e_cx2])
    cx_trend = cx_post - cx_pre

    return {
        "center": idx,
        "max_hist": max_hist,
        "width_hist": width_hist,
        "pre_edge": pre_edge,
        "post_edge": post_edge,
        "pre_frac": pre_frac,
        "post_frac": post_frac,
        "pre_hist": pre_hist,
        "post_hist": post_hist,
        "phash": phash_diff[idx],
        "cx_trend": cx_trend,
    }


def export_event_features():
    # 1) per-frame features
    print("Computing per-frame features ...")
    hist_diff, edge_counts, ecr, phash_d, frac_high, cx = compute_frame_features()

    # 2) ground truth events
    print("Loading ground truth ...")
    events_raw = load_ground_truth(GROUND_TRUTH_CSV)
    events = merge_intervals(events_raw)

    print(f"Total events: {len(events)}")

    # 3) summarize per event
    rows = []
    for ev in events:
        center = center_of_event(ev)
        feat = summarize_window(center, hist_diff, edge_counts, ecr,
                                phash_d, frac_high, cx)
        feat["kind"] = ev["kind"]
        rows.append(feat)

    # 4) write CSV
    if not rows:
        print("No events to write.")
        return

    fieldnames = list(rows[0].keys())
    print(f"Writing {len(rows)} rows to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    export_event_features()
    cap.release()
