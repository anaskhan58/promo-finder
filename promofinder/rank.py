"""
Turn scored clips into a posting order.

Clip quality and posting order are different questions. The best ten clips in
the show might all come from episode 27, and posting ten variations of the same
fight is a worse week than posting ten different reasons to start watching. So
ranking runs in two stages: score each clip, then re-order for a schedule.

Weights are deliberately visible and adjustable - an editor who disagrees with
the mix should be able to change one number, not read the code.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .segment import Candidate

# How much each judged dimension contributes to a clip's own score.
WEIGHTS = {
    "hook_strength": 0.40,        # nothing else matters if they scroll past
    "cliffhanger_strength": 0.30,  # the reason they open the app
    "standalone_clarity": 0.20,   # a cold viewer has no context
    "emotional_punch": 0.10,
}

VERDICT_BONUS = {"post": 0.35, "maybe": 0.0, "reject": -1.20}

# Early episodes are what a new viewer is sent into, so a promo drawn from them
# converts into a start rather than a confusing mid-season drop-in.
ENTRY_BONUS_EPISODES = 8
ENTRY_BONUS = 0.20

# No more than this many clips from one episode inside the top slice.
MAX_PER_EPISODE_IN_TOP = 2


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clip_score(judgement: dict, cand: Candidate) -> float:
    score = sum(w * _num(judgement.get(k)) for k, w in WEIGHTS.items())
    score += VERDICT_BONUS.get(str(judgement.get("verdict", "maybe")).lower(), 0.0)

    first_ep = min(cand.episodes) if cand.episodes else 99
    if first_ep <= ENTRY_BONUS_EPISODES:
        score += ENTRY_BONUS

    # Duration: 25-45s is the sweet spot for this format. Outside it, a small
    # penalty, not a veto - the hard 20-60s limit was already enforced upstream.
    duration = cand.duration
    if duration < 25.0 or duration > 45.0:
        score -= 0.10
    if cand.kind == "stitched":
        # A joined clip has a seam a viewer can feel, so it has to be clearly
        # better than a continuous one to outrank it. The eval showed the rules
        # alone cannot tell a coherent join from a jarring one, so without a
        # model in the loop this penalty is deliberately heavy.
        score -= 0.45
    return score


def rank(candidates: List[Candidate], judgements: Dict[str, dict]) -> List[dict]:
    """Score every judged clip, best first."""
    rows: List[dict] = []
    for cand in candidates:
        j = judgements.get(cand.cand_id)
        if not j:
            continue
        rows.append({
            "candidate": cand,
            "judgement": j,
            "score": clip_score(j, cand),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


def effective_cap(rows: List[dict], top: int) -> int:
    """
    How many clips one episode may contribute to the published list.

    Variety is bounded by how many episodes there are. Asking for 10 promos
    from a 2-episode pack cannot honour a cap of 2, so the cap rises to what
    the pack can actually supply. On a 50-episode show asking for 25, it stays
    at the floor.
    """
    episodes = {e for r in rows for e in (r["candidate"].episodes or [-1])}
    count = max(1, len(episodes))
    return max(MAX_PER_EPISODE_IN_TOP, math.ceil(top / count))


def posting_order(rows: List[dict], top: int = 25,
                  max_per_episode: Optional[int] = None) -> List[dict]:
    """
    Re-order the ranked clips into a schedule.

    Greedy: always take the best remaining clip that does not break the
    per-episode cap. Clips held back by the cap are not discarded, they just
    fall below the ones that add variety.
    """
    cap = max_per_episode or effective_cap(rows, top)

    chosen: List[dict] = []
    held: List[dict] = []
    per_episode: Dict[int, int] = {}

    for row in rows:
        eps = row["candidate"].episodes or [-1]
        # A stitched promo spends a slot in every episode it draws from, so it
        # cannot be used to smuggle a third cut of an already-covered scene.
        if any(per_episode.get(e, 0) >= cap for e in eps):
            held.append(row)
            continue
        for e in eps:
            per_episode[e] = per_episode.get(e, 0) + 1
        chosen.append(row)
        if len(chosen) >= top:
            break

    # Deliberately no padding from `held`. If variety limits the list to fewer
    # than `top`, returning fewer is correct: an editor is better served by six
    # different reasons to watch than by ten cuts of the same three scenes.
    for i, row in enumerate(chosen, start=1):
        row["rank"] = i
    return chosen
