"""
Command line entry point.

    python -m promofinder run     --input ../input_pack --out outputs
    python -m promofinder inspect --input ../input_pack
    python -m promofinder label   --pred outputs/promos.json --out eval/labels.csv
    python -m promofinder eval    --pred outputs/promos.json --labels eval/labels.csv
    python -m promofinder models  --provider gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

from .llm import DEFAULT_MODELS, build_scorer, list_models
from .parse import fmt_timestamp, load_episodes
from .rank import posting_order, rank
from .report import write_all
from .score import score_candidates
from .segment import build_candidates


def _show_name(input_path: str) -> str:
    base = os.path.basename(os.path.normpath(input_path))
    for root, dirs, files in os.walk(input_path):
        if any(f.lower().endswith(".txt") for f in files):
            return os.path.basename(root) or base
    return base


def cmd_run(args) -> int:
    t0 = time.time()

    print("Reading screenplays from %s ..." % args.input)
    episodes = load_episodes(args.input)
    if not episodes:
        print("No .txt screenplays found. Point --input at the unzipped pack.",
              file=sys.stderr)
        return 2
    blocks = sum(len(e.blocks) for e in episodes)
    warnings = sum(len(e.warnings) for e in episodes)
    print("  %d episodes, %d timed blocks, %d parse warnings"
          % (len(episodes), blocks, warnings))

    print("Generating candidate windows (rules only) ...")
    candidates, rejects = build_candidates(
        episodes, per_episode=args.per_episode, stitch=args.stitch)
    singles = sum(1 for c in candidates if c.kind == "single")
    print("  %d candidates (%d continuous, %d stitched)"
          % (len(candidates), singles, len(candidates) - singles))

    scorer = build_scorer(args.provider, args.model, args.effort)
    print("Scoring with %s ..." % scorer.describe())

    def progress(i, n, key):
        print("  [%2d/%2d] %s" % (i, n, key), end="\r", flush=True)

    judgements, failures = score_candidates(candidates, episodes, scorer, progress)
    print(" " * 40, end="\r")
    print("  %d clips judged, %d batches failed" % (len(judgements), len(failures)))

    rows = rank(candidates, judgements)
    # posting_order mutates rank on the rows it picks, so build the full
    # ranking list first and let the published slice be derived from it.
    all_rows = [dict(r) for r in rows]
    for i, r in enumerate(all_rows, start=1):
        r["rank"] = i
    top = posting_order(rows, top=args.top)

    usage = getattr(scorer, "usage", None)
    meta = {
        "show": _show_name(args.input),
        # Folder name only, never the absolute path: these files get shared,
        # and a home directory in them tells the reader whose machine ran it.
        "input": os.path.basename(os.path.normpath(args.input)),
        "episodes": len(episodes),
        "blocks": blocks,
        "candidates": len(candidates),
        "judged": len(judgements),
        "scorer": scorer.describe(),
        "per_episode": args.per_episode,
        "stitch": args.stitch,
        "runtime_s": round(time.time() - t0, 1),
        "token_usage": usage,
    }

    paths = write_all(args.out, top, meta, episodes, rejects, failures,
                      all_rows=all_rows)
    print()
    print("Top %d promos, best first:" % min(args.top, len(top)))
    for r in top[: args.show]:
        cand = r["candidate"]
        segs = " + ".join(
            "ep%d %s-%s" % (s.episode, fmt_timestamp(s.start), fmt_timestamp(s.end))
            for s in cand.segments)
        print("  %2d. %-34s %5.1fs  %.2f  %s"
              % (r["rank"], segs, cand.duration, r["score"],
                 (r["judgement"].get("hook", "") or "")[:60]))
    print()
    for name, path in paths.items():
        print("  %-9s %s" % (name, os.path.abspath(path)))
    print("Done in %.1fs." % (time.time() - t0))
    return 0


def cmd_inspect(args) -> int:
    """Show what the parser saw, so a format change is obvious immediately."""
    episodes = load_episodes(args.input)
    print("%d episodes parsed from %s" % (len(episodes), args.input))
    print()
    print("  ep  blocks  scenes  duration  dialogue  warnings")
    for e in episodes:
        spoken = sum(1 for b in e.blocks if b.kind == "dialogue")
        print("  %2d  %6d  %6d  %8s  %8d  %8d"
              % (e.number, len(e.blocks), len(e.scenes),
                 fmt_timestamp(e.duration), spoken, len(e.warnings)))
    allw = [(e.number, w) for e in episodes for w in e.warnings]
    if allw:
        print()
        print("Warnings:")
        for num, w in allw:
            print("  ep%d: %s" % (num, w))
    if episodes and episodes[0].blocks:
        print()
        print("First 8 blocks of episode %d:" % episodes[0].number)
        for b in episodes[0].blocks[:8]:
            print("  %s-%s %-9s %-16s %s"
                  % (fmt_timestamp(b.start), fmt_timestamp(b.end), b.kind,
                     b.speaker[:16], b.text[:60]))
    return 0


def cmd_label(args) -> int:
    """Emit a blank labelling sheet so the output can be checked by a human."""
    import csv

    with open(args.pred, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "segments", "duration_s", "model_rank",
                    "human_label_0_to_3", "note"])
        for p in data["promos"]:
            segs = " + ".join("ep%s %s-%s" % (s["episode"], s["start"], s["end"])
                              for s in p["segments"])
            w.writerow([p["id"], segs, p["total_duration_s"], p["rank"], "", ""])
    print("Wrote blank labelling sheet to %s" % os.path.abspath(args.out))
    print("Fill human_label_0_to_3: 3 = post it, 2 = usable, 1 = weak, 0 = no.")
    return 0


def cmd_eval(args) -> int:
    from eval.evaluate import run_eval  # noqa: PLC0415 - optional import

    return run_eval(args.pred, args.labels)


def cmd_models(args) -> int:
    """Which model ids this provider will serve today."""
    try:
        names = list_models(args.provider)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a message
        print("Could not list %s models: %s" % (args.provider, exc), file=sys.stderr)
        return 2
    default = DEFAULT_MODELS.get(args.provider)
    print("%d model(s) available from %s:" % (len(names), args.provider))
    for n in names:
        print("   %s%s" % (n, "   <- default" if n == default else ""))
    if default and default not in names:
        print()
        print("WARNING: the default for this provider, %s, is not in the list."
              % default, file=sys.stderr)
        print("         It has probably been retired. Pass --model explicitly.",
              file=sys.stderr)
        return 1
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="promofinder",
        description="Find postable promo clips inside microdrama screenplays.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="find promos across a whole show")
    r.add_argument("--input", required=True, help="folder of episode .txt files")
    r.add_argument("--out", default="outputs", help="output folder")
    r.add_argument("--provider", default="auto",
                   choices=["auto", "anthropic", "openai", "gemini", "ollama",
                            "heuristic"],
                   help="auto uses Claude if credentials exist, else rules only")
    r.add_argument("--model", default=None,
                   help="model id, e.g. %s" % DEFAULT_MODELS["anthropic"])
    r.add_argument("--effort", default="medium",
                   choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--per-episode", type=int, default=4,
                   help="candidate windows nominated per episode")
    r.add_argument("--stitch", type=int, default=6,
                   help="multi-segment candidates to build (0 to disable)")
    r.add_argument("--top", type=int, default=25, help="promos to publish")
    r.add_argument("--show", type=int, default=10, help="promos to print")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("inspect", help="show what the parser saw")
    i.add_argument("--input", required=True)
    i.set_defaults(func=cmd_inspect)

    l = sub.add_parser("label", help="write a blank human labelling sheet")
    l.add_argument("--pred", default="outputs/promos.json")
    l.add_argument("--out", default="eval/labels.csv")
    l.set_defaults(func=cmd_label)

    e = sub.add_parser("eval", help="score the tool against human labels")
    e.add_argument("--pred", default="outputs/promos.json")
    e.add_argument("--labels", default="eval/labels.csv")
    e.set_defaults(func=cmd_eval)

    m = sub.add_parser("models", help="list model ids a provider serves today")
    m.add_argument("--provider", default="gemini",
                   choices=["anthropic", "openai", "gemini", "ollama"])
    m.set_defaults(func=cmd_models)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
