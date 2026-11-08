"""
Render candidate clips as transcripts, send them to the scorer, merge results.

Candidates are batched by episode: one request carries every candidate from a
single episode. That cuts 200 requests down to about 50 and lets the model
compare clips against each other instead of judging each in a vacuum.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .llm import HeuristicScorer, Scorer
from .parse import CAMERA, DIALOGUE, SOUND, Block, Episode, fmt_timestamp
from .segment import Candidate, blocks_in

MAX_BLOCK_CHARS = 220


def render_block(b: Block, origin: float) -> str:
    """One transcript line, timestamped relative to the start of the clip."""
    offset = max(0.0, b.start - origin)
    text = b.text
    if len(text) > MAX_BLOCK_CHARS:
        text = text[:MAX_BLOCK_CHARS].rsplit(" ", 1)[0] + "..."

    stamp = "[+%04.1fs]" % offset
    if b.kind == DIALOGUE:
        who = b.speaker or "UNKNOWN"
        if b.offscreen:
            who += " (off-screen)"
        mood = (" (%s)" % ", ".join(b.emotions)) if b.emotions else ""
        return '%s %s%s: "%s"' % (stamp, who, mood, text)
    if b.kind == CAMERA:
        return "%s [shot] %s" % (stamp, text)
    if b.kind == SOUND:
        return "%s [sound] %s" % (stamp, text)
    return "%s [action] %s" % (stamp, text)


def render_candidate(cand: Candidate, episodes: Dict[int, Episode]) -> str:
    """A candidate clip as a readable transcript, segment by segment."""
    lines: List[str] = []
    origin = cand.segments[0].start
    for n, seg in enumerate(cand.segments, start=1):
        ep = episodes.get(seg.episode)
        if ep is None:
            continue
        if len(cand.segments) > 1:
            lines.append(
                "-- segment %d of %d: episode %d, %s to %s (%.1fs) --"
                % (n, len(cand.segments), seg.episode, fmt_timestamp(seg.start),
                   fmt_timestamp(seg.end), seg.duration)
            )
            origin = seg.start
        for b in blocks_in(ep, seg.start, seg.end):
            lines.append(render_block(b, origin))
    return "\n".join(lines)


def build_payload(episode_no: int, cands: List[Candidate],
                  episodes: Dict[int, Episode]) -> str:
    """The user message for one batch of candidates."""
    ep = episodes.get(episode_no)
    header = ["EPISODE %d" % episode_no]
    if ep is not None and ep.logline:
        header.append("Logline (for your orientation only - the clip must work "
                      "without it): %s" % ep.logline)
    header.append("")
    header.append("Judge each candidate clip below. Return one entry per clip, "
                  "using the exact id given.")
    header.append("")

    body: List[str] = []
    for cand in cands:
        spans = ", ".join(
            "ep%d %s-%s" % (s.episode, fmt_timestamp(s.start), fmt_timestamp(s.end))
            for s in cand.segments
        )
        body.append("=== CLIP id=%s | %.1fs | %s ===" % (cand.cand_id, cand.duration, spans))
        body.append(render_candidate(cand, episodes))
        body.append("")
    return "\n".join(header + body)


def _merge(key: str, cands: List[Candidate], result: dict,
           judgements: Dict[str, dict], failures: List[dict]) -> None:
    """Take the clips the model actually returned, and record the ones it skipped."""
    allowed = {c.cand_id for c in cands}
    seen = set()
    for clip in result.get("clips", []):
        cid = str(clip.get("id", "")).strip()
        if cid not in allowed:
            continue
        judgements[cid] = clip
        seen.add(cid)
    missing = [c.cand_id for c in cands if c.cand_id not in seen]
    if missing:
        failures.append({"batch": key,
                         "error": "model returned no entry for these clips",
                         "candidate_ids": missing})


def score_candidates(candidates: List[Candidate], episodes: List[Episode],
                     scorer: Scorer, progress=None, workers: int = 1
                     ) -> Tuple[Dict[str, dict], List[dict]]:
    """
    Score every candidate. Returns judgements keyed by candidate id, plus a log
    of batches that failed, so a partial run is still usable and still honest
    about what is missing.

    One batch is one episode, and batches do not depend on each other, so with
    a network scorer they are sent concurrently: a 50-episode show is ~50
    minutes serial and a few minutes at workers=6. The heuristic scorer stays
    serial regardless - it carries per-batch state on the scorer object, and it
    is instant anyway.
    """
    ep_map = {ep.number: ep for ep in episodes}

    batches: Dict[str, List[Candidate]] = {}
    for cand in candidates:
        key = "stitched" if cand.kind == "stitched" else "ep%03d" % cand.segments[0].episode
        batches.setdefault(key, []).append(cand)

    judgements: Dict[str, dict] = {}
    failures: List[dict] = []
    ordered = sorted(batches.items())
    total = len(ordered)

    serial = isinstance(scorer, HeuristicScorer) or workers <= 1

    if serial:
        for i, (key, cands) in enumerate(ordered, start=1):
            if progress:
                progress(i, total, key)
            if isinstance(scorer, HeuristicScorer):
                scorer.context = {
                    c.cand_id: {
                        "signals": c.signals,
                        "first_line": _first_line(c, ep_map),
                        "last_line": _last_line(c, ep_map),
                    }
                    for c in cands
                }
            payload = build_payload(cands[0].segments[0].episode, cands, ep_map)
            try:
                result = scorer.judge(payload)
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                failures.append({"batch": key, "error": str(exc),
                                 "candidate_ids": [c.cand_id for c in cands]})
                continue
            _merge(key, cands, result, judgements, failures)
        return judgements, failures

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    done = [0]

    def work(item):
        key, cands = item
        payload = build_payload(cands[0].segments[0].episode, cands, ep_map)
        try:
            return key, cands, scorer.judge(payload), None
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            return key, cands, None, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, cands, result, exc in pool.map(work, ordered):
            with lock:
                done[0] += 1
                if progress:
                    progress(done[0], total, key)
            if exc is not None:
                failures.append({"batch": key, "error": str(exc),
                                 "candidate_ids": [c.cand_id for c in cands]})
                continue
            _merge(key, cands, result, judgements, failures)

    failures.sort(key=lambda f: f["batch"])
    return judgements, failures


def _first_line(cand: Candidate, ep_map: Dict[int, Episode]) -> str:
    seg = cand.segments[0]
    ep = ep_map.get(seg.episode)
    if not ep:
        return ""
    for b in blocks_in(ep, seg.start, seg.start + 4.0):
        if b.kind == DIALOGUE and b.text:
            return "%s: %s" % (b.speaker, b.text)
    blocks = blocks_in(ep, seg.start, seg.start + 4.0)
    return blocks[0].text if blocks else ""


def _last_line(cand: Candidate, ep_map: Dict[int, Episode]) -> str:
    seg = cand.segments[-1]
    ep = ep_map.get(seg.episode)
    if not ep:
        return ""
    blocks = blocks_in(ep, max(seg.start, seg.end - 8.0), seg.end)
    spoken = [b for b in blocks if b.kind == DIALOGUE and b.text]
    if spoken:
        b = spoken[-1]
        return "%s: %s" % (b.speaker, b.text)
    return blocks[-1].text if blocks else ""
