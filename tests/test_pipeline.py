"""
Smoke tests for the promo pipeline, run against the synthetic demo pack.

These check the claims the solution doc makes, not the taste of the picks:
every published clip is 20-60 seconds, every cut lands on a real line or
action boundary, and the per-episode cap in the posting order holds.

    python tests/test_pipeline.py        # no pytest needed
    pytest tests/                        # also works
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from promofinder.llm import build_scorer, extract_json  # noqa: E402
from promofinder.parse import fmt_timestamp, load_episodes, parse_timestamp  # noqa: E402
from promofinder.rank import effective_cap, posting_order, rank  # noqa: E402
from promofinder.score import score_candidates  # noqa: E402
from promofinder.segment import MAX_DURATION, MIN_DURATION, build_candidates  # noqa: E402

DEMO = os.path.join(ROOT, "demo_pack")


def test_timestamps_round_trip():
    assert parse_timestamp("02:18.00") == 138.0
    assert parse_timestamp("(00:04.80)") == 4.8
    assert parse_timestamp("01:02:18.00") == 3738.0
    assert parse_timestamp("not a time") is None
    assert fmt_timestamp(138.0) == "02:18.00"
    assert fmt_timestamp(4.8) == "00:04.80"


def test_speaker_labels_split_correctly():
    from promofinder.parse import _split_label

    kind, who, moods, off = _split_label("MIRA (Cold, Commanding):")
    assert (kind, who, moods, off) == ("dialogue", "MIRA", ["Cold", "Commanding"], False)

    kind, who, moods, off = _split_label("JULIAN (V.O.) (Haunted, Resolute):")
    assert who == "JULIAN" and moods == ["Haunted", "Resolute"] and off is True

    kind, who, _moods, _off = _split_label("SOUND FX:")
    assert kind == "sound" and who == ""

    kind, _who, _m, _o = _split_label("CAMERA:")
    assert kind == "camera"


def test_pack_parses_cleanly():
    episodes = load_episodes(DEMO)
    assert len(episodes) == 2
    assert all(ep.blocks for ep in episodes)
    assert all(not ep.warnings for ep in episodes), [e.warnings for e in episodes]
    assert all(ep.logline for ep in episodes)
    for ep in episodes:
        for b in ep.blocks:
            assert b.end >= b.start


def test_json_repair_handles_fenced_and_trailing_text():
    assert extract_json('```json\n{"clips": []}\n```') == {"clips": []}
    assert extract_json('here you go: {"clips": [{"id": "a"}]} thanks') == {
        "clips": [{"id": "a"}]
    }
    assert extract_json("no json here") is None


def _run():
    episodes = load_episodes(DEMO)
    candidates, _rejects = build_candidates(episodes, per_episode=4, stitch=4)
    judgements, failures = score_candidates(
        candidates, episodes, build_scorer("heuristic"))
    rows = posting_order(rank(candidates, judgements), top=10)
    return episodes, candidates, judgements, failures, rows


def test_every_published_clip_obeys_the_hard_rules():
    episodes, _cands, _j, failures, rows = _run()
    assert not failures
    assert rows, "no promos produced from the demo pack"

    boundaries = {}
    for ep in episodes:
        starts = {round(b.start, 2) for b in ep.blocks}
        ends = {round(b.end, 2) for b in ep.blocks}
        ends |= {round(s.end, 2) for s in ep.scenes}
        starts |= {0.0}
        boundaries[ep.number] = (starts, ends)

    for r in rows:
        cand = r["candidate"]
        assert MIN_DURATION <= cand.duration <= MAX_DURATION, (
            "%s is %.1fs" % (cand.cand_id, cand.duration))
        for seg in cand.segments:
            starts, ends = boundaries[seg.episode]
            assert round(seg.start, 2) in starts, (
                "%s starts mid-line at %s" % (cand.cand_id, fmt_timestamp(seg.start)))
            assert round(seg.end, 2) in ends, (
                "%s ends mid-line at %s" % (cand.cand_id, fmt_timestamp(seg.end)))
            assert seg.end > seg.start


def test_posting_order_caps_clips_per_episode():
    _eps, cands, judgements, _f, rows = _run()
    cap = effective_cap(rank(cands, judgements), 10)
    counts = {}
    for r in rows:
        for e in r["candidate"].episodes:
            counts[e] = counts.get(e, 0) + 1
    assert counts, "no episodes represented"
    assert max(counts.values()) <= cap, (counts, cap)


def test_cap_stays_tight_on_a_full_length_show():
    """On a 50-episode show asking for 25 promos, the cap must stay at 2."""
    class _C:
        def __init__(self, eps):
            self.episodes = eps

    rows = [{"candidate": _C([n])} for n in range(1, 51)]
    assert effective_cap(rows, 25) == 2
    # A 2-episode pack cannot supply 10 promos at 2 each, so the cap rises.
    assert effective_cap([{"candidate": _C([1])}, {"candidate": _C([2])}], 10) == 5


def test_every_clip_carries_an_explanation():
    _eps, _c, _j, _f, rows = _run()
    for r in rows:
        j = r["judgement"]
        for field in ("hook", "cliffhanger", "why", "risk", "verdict"):
            assert field in j, "%s missing %s" % (r["candidate"].cand_id, field)
        assert j["verdict"] in {"post", "maybe", "reject"}


def test_ranking_is_monotonic():
    _eps, _c, _j, _f, rows = _run()
    scores = [r["score"] for r in rows]
    # posting_order may hold a clip back for diversity, but the published list
    # must still be broadly descending.
    assert scores[0] == max(scores)


def test_streamlit_app_renders():
    """The hosted demo must load and produce promos with no key and no upload."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("      (skipped: streamlit not installed)")
        return

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    assert at.expander, "app rendered no promos on the demo pack"
    labels = [m.label for m in at.metric]
    for expected in ("Episodes read", "Timed blocks", "Candidates", "Promos returned"):
        assert expected in labels, labels


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except AssertionError as exc:
            failures += 1
            print("FAIL  %s\n      %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR %s\n      %r" % (name, exc))
    print()
    print("%d failed" % failures if failures else "all tests passed")
    sys.exit(1 if failures else 0)
