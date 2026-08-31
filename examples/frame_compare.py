#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import pandas as pd
import sys
import os
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.5

MIN_FACE_W, MIN_FACE_H = 40, 40
MIN_PERSON_W, MIN_PERSON_H = 100, 150

LABELS_TO_EVAL = {"person"}

# -------------------------------------------------------
# DEBUG LOGGING
# -------------------------------------------------------

DEBUG_LOG = True
DEBUG_MAX_LINES = 200000
DEBUG_FILE = "tracking_debug.log"

debug_lines = []

def log_debug(msg):
    if not DEBUG_LOG:
        return
    if len(debug_lines) < DEBUG_MAX_LINES:
        debug_lines.append(msg)


# -------------------------------------------------------
# IOU
# -------------------------------------------------------

def calculate_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    if xB <= xA or yB <= yA:
        return 0.0

    inter = (xB - xA) * (yB - yA)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# -------------------------------------------------------
# XML PARSER
# -------------------------------------------------------

def parse_xml(file):
    if not os.path.exists(file):
        return None

    tree = ET.parse(file)
    root = tree.getroot()

    data = {}
    total = 0
    filtered = 0
    skipped_label = 0

    for track in root.findall("track"):
        label = track.get("label", "").lower().strip()

        if label not in LABELS_TO_EVAL:
            skipped_label += 1
            continue

        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue

            total += 1

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            w = xbr - xtl
            h = ybr - ytl

            if label == "face":
                if w < MIN_FACE_W or h < MIN_FACE_H:
                    filtered += 1
                    continue

            if label == "person":
                if w < MIN_PERSON_W or h < MIN_PERSON_H:
                    filtered += 1
                    continue

            frame = int(box.get("frame"))

            gid = "N/A"
            score = 1.0

            for attr in box.findall("attribute"):
                name = attr.get("name")
                if name == "Global User Id":
                    gid = attr.text
                if name.lower() in ["score", "confidence"]:
                    try:
                        score = float(attr.text)
                    except:
                        pass

            if frame not in data:
                data[frame] = []

            data[frame].append({
                "label": label,
                "bbox": [xtl, ytl, xbr, ybr],
                "global_id": gid,
                "score": score
            })

    print(f"Loaded {file} | total boxes={total} filtered={filtered} skipped_labels={skipped_label}", file=sys.stderr)

    return data


# -------------------------------------------------------
# HUNGARIAN MATCHING
# -------------------------------------------------------

def match_detections(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return [], list(range(len(gt))), list(range(len(pred)))

    cost = np.zeros((len(gt), len(pred)))

    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = calculate_iou(g["bbox"], p["bbox"])
            cost[i, j] = 1 - iou if iou >= IOU_THRESHOLD else 1

    rows, cols = linear_sum_assignment(cost)

    matches = []
    unmatched_gt = set(range(len(gt)))
    unmatched_pred = set(range(len(pred)))

    for r, c in zip(rows, cols):
        if cost[r, c] < 1:
            matches.append((r, c))
            unmatched_gt.discard(r)
            unmatched_pred.discard(c)

    return matches, list(unmatched_gt), list(unmatched_pred)


# -------------------------------------------------------
# AP
# -------------------------------------------------------

def calculate_ap(gt_data, pd_data, label):
    preds = []
    total_gt = 0

    for f in gt_data:
        total_gt += sum(1 for g in gt_data[f] if g["label"] == label)

    for f in pd_data:
        for p in pd_data[f]:
            if p["label"] == label:
                preds.append((f, p))

    if total_gt == 0:
        return None

    preds.sort(key=lambda x: x[1]["score"], reverse=True)

    tp = []
    fp = []
    matched = defaultdict(set)

    for frame, p in preds:
        best_iou = 0
        best_idx = -1

        for i, g in enumerate(gt_data.get(frame, [])):
            if g["label"] != label:
                continue

            iou = calculate_iou(p["bbox"], g["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD and best_idx not in matched[frame]:
            tp.append(1)
            fp.append(0)
            matched[frame].add(best_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / total_gt
    prec = tp / (tp + fp + 1e-9)

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(prec[recall >= t]) if np.sum(recall >= t) else 0
        ap += p / 11

    return ap


# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------

def evaluate(gt_data, pd_data, label):

    frames = sorted(set(gt_data.keys()) | set(pd_data.keys()))

    rows = []
    id_history = {}
    id_switches = 0

    mapping_counts = defaultdict(lambda: defaultdict(int))

    for f in frames:
        gt = [g for g in gt_data.get(f, []) if g["label"] == label]
        pred = [p for p in pd_data.get(f, []) if p["label"] == label]

        matches, _, _ = match_detections(gt, pred)

        for g_idx, p_idx in matches:
            g_id = gt[g_idx]["global_id"]
            p_id = pred[p_idx]["global_id"]
            mapping_counts[g_id][p_id] += 1

    id_map = {}
    for g_id, counts in mapping_counts.items():
        id_map[g_id] = max(counts.items(), key=lambda x: x[1])[0]

    for f in frames:

        gt = [g for g in gt_data.get(f, []) if g["label"] == label]
        pred = [p for p in pd_data.get(f, []) if p["label"] == label]

        matches, miss, fp = match_detections(gt, pred)

        # ------------------------------------------------
        # LOG MISSES
        # ------------------------------------------------
        for m in miss:
            g = gt[m]
            log_debug(
                f"[MISS] frame={f} GT_ID={g['global_id']} bbox={g['bbox']}"
            )

        # ------------------------------------------------
        # LOG FALSE POSITIVES
        # ------------------------------------------------
        for p in fp:
            pr = pred[p]
            log_debug(
                f"[FALSE_POSITIVE] frame={f} "
                f"PRED_ID={pr['global_id']} "
                f"score={pr['score']:.3f} "
                f"bbox={pr['bbox']}"
            )

        row = {"GT": len(gt), "Miss": len(miss), "FP": len(fp), "IDTP": 0}

        for g_idx, p_idx in matches:

            g_id = gt[g_idx]["global_id"]
            p_id = pred[p_idx]["global_id"]

            iou = calculate_iou(gt[g_idx]["bbox"], pred[p_idx]["bbox"])

            log_debug(
                f"[MATCH] frame={f} "
                f"GT_ID={g_id} "
                f"PRED_ID={p_id} "
                f"IoU={iou:.3f}"
            )

            if g_id in id_map and id_map[g_id] == p_id:
                row["IDTP"] += 1

            if g_id in id_history and id_history[g_id] != p_id:
                id_switches += 1

                log_debug(
                    f"[ID_SWITCH] frame={f} "
                    f"GT_ID={g_id} "
                    f"old_pred={id_history[g_id]} "
                    f"new_pred={p_id}"
                )

            id_history[g_id] = p_id

        rows.append(row)

    df = pd.DataFrame(rows)

    gt = df["GT"].sum()
    miss = df["Miss"].sum()
    fp = df["FP"].sum()
    idtp = df["IDTP"].sum()

    matches = gt - miss

    mota = 1 - (miss + fp + id_switches) / gt if gt > 0 else 0
    denom = 2 * idtp + fp + miss
    idf1 = (2 * idtp) / denom if denom > 0 else 0
    stability = 1 - id_switches / matches if matches > 0 else 0

    return {
        "gt": gt,
        "miss": miss,
        "fp": fp,
        "id_sw": id_switches,
        "mota": mota,
        "idf1": idf1,
        "stability": stability
    }


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def compare_and_evaluate(gt_xml, pred_xml):
    log_debug(
                gt_xml
            )
    gt_data = parse_xml(gt_xml)
    pd_data = parse_xml(pred_xml)

    if gt_data is None or pd_data is None:
        return

    labels = set()

    for f in gt_data:
        for o in gt_data[f]:
            labels.add(o["label"])

    for f in pd_data:
        for o in pd_data[f]:
            labels.add(o["label"])

    labels = [l for l in labels if l in LABELS_TO_EVAL]

    results = []

    for label in labels:
        ap = calculate_ap(gt_data, pd_data, label)
        if ap is None:
            continue

        metrics = evaluate(gt_data, pd_data, label)

        results.append((label, ap, metrics))

    if not results:
        print("No results — check LABELS_TO_EVAL or GT/pred XML labels.")
        return

    print("\n" + "=" * 120)
    print(f"{'CATEGORY':<10} | {'mAP':<8} | {'MOTA':<8} | {'IDF1':<8} | {'STABILITY':<10} | {'GT':<6} | {'MISS':<6} | {'FP':<6} | {'ID_SW':<6}")
    print("-" * 120)

    maps, motas, idf1s, stabs = [], [], [], []

    for label, ap, m in results:

        print(f"{label.capitalize():<10} | {ap:<8.2%} | {m['mota']:<8.2%} | {m['idf1']:<8.2%} | {m['stability']:<10.2%} | {m['gt']:<6} | {m['miss']:<6} | {m['fp']:<6} | {m['id_sw']:<6}")

        maps.append(ap)
        motas.append(m["mota"])
        idf1s.append(m["idf1"])
        stabs.append(m["stability"])

    if len(results) > 1:
        print("-" * 120)
        print(f"{'AVERAGE':<10} | {np.mean(maps):<8.2%} | {np.mean(motas):<8.2%} | {np.mean(idf1s):<8.2%} | {np.mean(stabs):<10.2%}")

    print("=" * 120)

    # -------------------------------------------------------
    # WRITE DEBUG FILE
    # -------------------------------------------------------

    if DEBUG_LOG and debug_lines:

        with open(DEBUG_FILE, "w") as f:
            f.write("\n".join(debug_lines))

        print(f"\nDebug log written to {DEBUG_FILE}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python3 frame_compare.py <gt.xml> <pred.xml>")
        sys.exit(0)

    compare_and_evaluate(sys.argv[1], sys.argv[2])
