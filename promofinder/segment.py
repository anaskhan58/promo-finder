"""
Turn an episode timeline into a short list of candidate promo windows.

Rules live here, not in the model. A window is only a candidate if it already
satisfies the hard constraints from the brief:

  * 20 to 60 seconds long
  * starts on a real cut point (a block boundary, never mid-line)
  * ends on a real cut point
  * contains at least some dialogue, so it reads without the video

Everything that fails those rules is dropped before the model sees it, and the
reason is recorded so the reject list can be inspected later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .features import cliff_strength, hook_strength, opens_mid_thought, punch
from .parse import DIALOGUE, Block, Episode, fmt_timestamp

MIN_DURATION = 20.0
MAX_DURATION = 60.0
HOOK_WINDOW = 3.0  # the brief's "first 3 seconds"


@dataclass
class Segment:
    """One continuous stretch of one episode."""

    episode: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "start": fmt_timestamp(self.start),
            "end": fmt_timestamp(self.end),
            "duration_s": round(self.duration, 2),
        }


@dataclass
class Candidate:
    """A proposed promo: one or more segments, in play order."""

    segments: List[Segment]
    prescore: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    kind: str = "single"  # single | stitched
    cand_id: str = ""

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def episodes(self) -> List[int]:
        seen: List[int] = []
        for s in self.segments:
            if s.episode not in seen:
                seen.append(s.episode)
        return seen

    def to_dict(self) -> dict:
        return {
            "id": self.cand_id,
            "kind": self.kind,
            "episodes": self.episodes,
            "segments": [s.to_dict() for s in self.segments],
            "total_duration_s": round(self.duration, 2),
            "prescore": round(self.prescore, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
        }


# ------------------------------------------------------------------ helpers


def blocks_in(ep: Episode, start: float, end: float) -> List[Block]:
    """Blocks overlapping [start, end), in play order."""
    return [b for b in ep.blocks if b.end > start and b.start < end]


def _scene_end_times(ep: Episode) -> set:
    return {round(s.end, 2) for s in ep.scenes}


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = _overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ scoring


def score_window(ep: Episode, start: float, end: float,
                 scene_ends: set) -> Optional[Dict[str, float]]:
    """Signals for one window, or None if it breaks a hard rule."""
    window = blocks_in(ep, start, end)
    if not window:
        return None

    spoken = [b for b in window if b.kind == DIALOGUE]
    if len(spoken) < 2:
        return None  # a silent montage does not read on mute-scroll feeds

    # Hook: the strongest thing inside the opening seconds.
    opening = [b for b in window if b.start < start + HOOK_WINDOW]
    hook = max((hook_strength(b) for b in opening), default=0.0)

    # If the very first spoken line is a continuation of a line the viewer will
    # never hear, the clip opens on nonsense no matter what follows it.
    first_spoken = next((b for b in window if b.kind == DIALOGUE), None)
    if first_spoken is not None and first_spoken.start < start + HOOK_WINDOW \
            and opens_mid_thought(first_spoken):
        hook *= 0.4

    # Cliffhanger: the strength of the block the cut lands on.
    tail = [b for b in window if b.end <= end + 0.01]
    last = max(tail, key=lambda b: b.end) if tail else window[-1]
    cliff = cliff_strength(
        last,
        is_episode_end=abs(end - ep.duration) < 0.5,
        is_scene_end=round(end, 2) in scene_ends,
    )

    # Middle: does it stay alive between hook and cut?
    body = sorted((punch(b) for b in window), reverse=True)
    sustain = sum(body[:5]) / 5.0 if body else 0.0

    # Self-contained: lots of named speakers in 30s is hard to follow cold.
    speakers = {b.speaker for b in spoken if b.speaker}
    clarity = 1.0 if len(speakers) <= 3 else max(0.2, 1.0 - 0.2 * (len(speakers) - 3))

    # Talk density: wall-to-wall camera description makes a dull clip.
    talk = sum(b.duration for b in spoken) / max(1e-6, end - start)

    duration = end - start
    sweet = 1.0 if 25.0 <= duration <= 50.0 else 0.85

    prescore = (
        2.2 * hook
        + 1.8 * cliff
        + 1.0 * sustain
        + 0.8 * clarity
        + 0.6 * min(1.0, talk * 2.0)
    ) * sweet

    return {
        "hook": hook,
        "cliff": cliff,
        "sustain": sustain,
        "clarity": clarity,
        "talk_density": talk,
        "prescore": prescore,
    }


# ------------------------------------------------------------------ generate


def candidates_for_episode(ep: Episode, top_k: int = 4,
                           rejects: Optional[List[dict]] = None) -> List[Candidate]:
    """Nominate up to top_k non-overlapping windows from one episode."""
    if not ep.blocks:
        return []

    scene_ends = _scene_end_times(ep)

    # Candidate cut points are real block boundaries, so a clip never starts or
    # ends mid-line. Starts are biased toward blocks that could open a promo.
    starts = sorted({round(b.start, 2) for b in ep.blocks
                     if hook_strength(b) > 0.25} | {0.0})
    ends = sorted({round(b.end, 2) for b in ep.blocks} | scene_ends |
                  {round(ep.duration, 2)})

    scored: List[Tuple[float, float, float, Dict[str, float]]] = []
    for s in starts:
        for e in ends:
            duration = e - s
            if duration < MIN_DURATION or duration > MAX_DURATION:
                continue
            signals = score_window(ep, s, e, scene_ends)
            if signals is None:
                continue
            scored.append((signals["prescore"], s, e, signals))

    scored.sort(key=lambda t: -t[0])

    picked: List[Candidate] = []
    for prescore, s, e, signals in scored:
        if any(_iou((s, e), (c.segments[0].start, c.segments[0].end)) > 0.5
               for c in picked):
            continue
        picked.append(
            Candidate(
                segments=[Segment(ep.number, s, e)],
                prescore=prescore,
                signals=signals,
                kind="single",
                cand_id="ep%03d-%s-%s" % (ep.number, fmt_timestamp(s).replace(":", ""),
                                          fmt_timestamp(e).replace(":", "")),
            )
        )
        if len(picked) >= top_k:
            break

    if rejects is not None and not picked and ep.blocks:
        rejects.append({
            "episode": ep.number,
            "reason": "no window satisfied the 20-60s + dialogue constraints",
        })
    return picked


def stitched_candidates(episodes: List[Episode], singles: List[Candidate],
                        limit: int = 6) -> List[Candidate]:
    """
    Build a few multi-segment promos: a hard hook from one episode cut straight
    into a cliffhanger from another.

    The brief allows a promo to join moments across episodes. These are kept
    separate and few, because a stitched clip is far easier to get wrong and
    needs the model to confirm the two halves make sense together.
    """
    by_ep = {ep.number: ep for ep in episodes}

    # Cut points for each episode, so a stitched segment lands on a real line
    # boundary exactly like a continuous one does.
    starts_by_ep = {ep.number: sorted({round(b.start, 2) for b in ep.blocks} | {0.0})
                    for ep in episodes}
    ends_by_ep = {ep.number: sorted({round(b.end, 2) for b in ep.blocks}
                                    | _scene_end_times(ep))
                  for ep in episodes}

    def snap_end(ep_no: int, target: float, floor: float) -> Optional[float]:
        """Earliest real cut point at or after target, still above floor."""
        for value in ends_by_ep.get(ep_no, []):
            if value >= target and value > floor:
                return value
        return None

    def snap_start(ep_no: int, target: float) -> Optional[float]:
        """Latest real cut point at or before target."""
        best = None
        for value in starts_by_ep.get(ep_no, []):
            if value <= target:
                best = value
            else:
                break
        return best

    # Best short hook fragments: a punchy opening block plus a little room.
    hooks: List[Tuple[float, Segment]] = []
    for ep in episodes:
        for b in ep.blocks:
            h = hook_strength(b)
            if h < 1.2 or b.kind != DIALOGUE:
                continue
            start = round(b.start, 2)
            end = snap_end(ep.number, min(start + 6.0, ep.duration), start)
            if end is None or end - start < 3.0 or end - start > 12.0:
                continue
            hooks.append((h, Segment(ep.number, start, end)))
    hooks.sort(key=lambda t: -t[0])

    # Best cliffhanger tails, taken from the strongest single candidates so the
    # second half is a stretch we already believe in.
    tails: List[Tuple[float, Segment]] = []
    for c in sorted(singles, key=lambda c: -c.signals.get("cliff", 0.0)):
        seg = c.segments[0]
        ep = by_ep.get(seg.episode)
        if ep is None:
            continue
        start = snap_start(seg.episode, max(seg.start, seg.end - 28.0))
        if start is None or seg.end - start < 8.0:
            continue
        tails.append((c.signals.get("cliff", 0.0), Segment(seg.episode, start, seg.end)))

    out: List[Candidate] = []
    used_hook_eps: set = set()
    used_tail_eps: set = set()
    for h_score, h_seg in hooks:
        if len(out) >= limit:
            break
        if h_seg.episode in used_hook_eps:
            continue
        for c_score, t_seg in tails:
            if t_seg.episode == h_seg.episode or t_seg.episode in used_tail_eps:
                continue
            total = h_seg.duration + t_seg.duration
            if total < MIN_DURATION or total > MAX_DURATION:
                continue
            used_hook_eps.add(h_seg.episode)
            used_tail_eps.add(t_seg.episode)
            out.append(
                Candidate(
                    segments=[h_seg, t_seg],
                    prescore=2.2 * h_score + 1.8 * c_score,
                    signals={"hook": h_score, "cliff": c_score,
                             "sustain": 0.0, "clarity": 0.7, "talk_density": 0.5},
                    kind="stitched",
                    cand_id="stitch-ep%03d-ep%03d" % (h_seg.episode, t_seg.episode),
                )
            )
            break
    return out


def build_candidates(episodes: List[Episode], per_episode: int = 4,
                     stitch: int = 6) -> Tuple[List[Candidate], List[dict]]:
    rejects: List[dict] = []
    singles: List[Candidate] = []
    for ep in episodes:
        singles.extend(candidates_for_episode(ep, per_episode, rejects))
    out = list(singles)
    if stitch:
        out.extend(stitched_candidates(episodes, singles, stitch))
    return out, rejects
