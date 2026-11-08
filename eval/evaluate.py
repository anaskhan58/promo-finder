"""
Score the tool against human labels.

The question this answers is not "did the model say nice things" but "if an
editor took the top of this list, how much of their time would be wasted".

Labels are 0 to 3:
    3 = post it today
    2 = usable, needs a tighter cut
    1 = weak
    0 = not a promo

A clip counts as GOOD at 2 or above, because that is the bar for an editor
opening the file and doing something with it.

    python -m promofinder eval --pred outputs/promos_all.json --labels eval/labels.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Tuple

GOOD = 2


def _load_labels(path: str) -> Dict[str, Tuple[int, str]]:
    out: Dict[str, Tuple[int, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("id") or "").strip()
            raw = (row.get("human_label_0_to_3") or "").strip()
            if not cid or raw == "":
                continue
            try:
                out[cid] = (int(float(raw)), (row.get("note") or "").strip())
            except ValueError:
                continue
    return out


def _load_predictions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("promos", [])


def _spearman(pairs: List[Tuple[float, float]]) -> float:
    """Rank correlation, ties averaged. Stdlib only, no scipy."""
    if len(pairs) < 3:
        return 0.0

    def ranks(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    xs = ranks([p[0] for p in pairs])
    ys = ranks([p[1] for p in pairs])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def run_eval(pred_path: str, labels_path: str) -> int:
    if not os.path.exists(pred_path):
        print("No predictions at %s. Run the tool first." % pred_path, file=sys.stderr)
        return 2
    if not os.path.exists(labels_path):
        print("No labels at %s. Run `label` and fill it in." % labels_path,
              file=sys.stderr)
        return 2

    labels = _load_labels(labels_path)
    promos = _load_predictions(pred_path)
    if not promos:
        print("Prediction file has no promos.", file=sys.stderr)
        return 2

    ordered = sorted(promos, key=lambda p: p.get("rank") or 10**6)
    judged = [(p, labels[p["id"]]) for p in ordered if p["id"] in labels]

    print("Evaluating %s" % os.path.abspath(pred_path))
    print("  %d clips ranked, %d of them labelled by a human" % (len(ordered), len(judged)))
    if not judged:
        print("  No overlap between predictions and labels.", file=sys.stderr)
        return 2
    print()

    # Precision at k over the labelled subset, in tool order.
    print("  Precision at k (a hit is a human label of %d or more):" % GOOD)
    for k in (5, 10, 15, 20):
        slice_ = judged[:k]
        if not slice_:
            continue
        hits = sum(1 for _p, (lab, _n) in slice_ if lab >= GOOD)
        print("    top %-2d  %d/%d  %.0f%%" % (k, hits, len(slice_), 100.0 * hits / len(slice_)))
    print()

    half = max(1, len(judged) // 2)
    top_mean = sum(lab for _p, (lab, _n) in judged[:half]) / half
    bot = judged[half:]
    bot_mean = sum(lab for _p, (lab, _n) in bot) / len(bot) if bot else 0.0
    print("  Mean human label, better half of the tool's ranking: %.2f" % top_mean)
    print("  Mean human label, worse half:                        %.2f" % bot_mean)
    print("  Separation:                                          %+.2f" % (top_mean - bot_mean))
    print()

    rho = _spearman([(float(p.get("score", 0.0)), float(lab)) for p, (lab, _n) in judged])
    print("  Spearman correlation, tool score vs human label: %+.2f" % rho)
    print()

    # The honest part: where it is wrong, named.
    misses = [(p, lab, note) for p, (lab, note) in judged if lab <= 1]
    misses.sort(key=lambda t: t[0].get("rank") or 10**6)
    if misses:
        print("  Worst false positives (tool ranked them, a human would not post them):")
        for p, lab, note in misses[:6]:
            print("    rank %-3s label %d  %s" % (p.get("rank"), lab, p["id"]))
            print("        %s" % note)
        print()

    buried = [(p, lab, note) for p, (lab, note) in judged
              if lab == 3 and (p.get("rank") or 999) > 10]
    if buried:
        print("  Good clips the tool buried below rank 10:")
        for p, lab, note in buried:
            print("    rank %-3s label %d  %s" % (p.get("rank"), lab, p["id"]))
            print("        %s" % note)
        print()

    dist: Dict[int, int] = {}
    for _p, (lab, _n) in judged:
        dist[lab] = dist.get(lab, 0) + 1
    print("  Label distribution in the sample: " +
          ", ".join("%d:%d" % (k, dist[k]) for k in sorted(dist, reverse=True)))
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/promos_all.json")
    ap.add_argument("--labels", default="eval/labels.csv")
    a = ap.parse_args()
    raise SystemExit(run_eval(a.pred, a.labels))
