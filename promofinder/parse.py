"""
Parse microdrama screenplays into a timed block timeline.

The input pack format (one .txt per episode) looks like:

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...

    --- SCRIPT START ---

    SCENE 01
    Scene heading: INT. GLASS BOARDROOM - MORNING
    Scene start: 00:00.00
    Scene end: 01:12.00

    Actions and dialogue:

    **CAMERA:**
    Start: (00:00.00) | End: (00:03.20)
    Slow push-in through glass walls on a long boardroom table ...

    **MIRA (Uncertain, Breathless):**
    Start: (00:06.20) | End: (00:07.80)
    "Is something wrong?"

Field labels appear in both Title Case and ALL CAPS across the pack, so every
pattern here is case-insensitive. Blocks overlap in time (a CAMERA block and an
ACTION block routinely cover the same seconds), so the timeline is treated as
overlapping tracks rather than a strict sequence of beats.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------- timestamps

_TS = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)")


def parse_timestamp(raw: Optional[str]) -> Optional[float]:
    """'02:18.00' -> 138.0 ; '01:02:18.00' -> 3738.0. None if unparseable."""
    if raw is None:
        return None
    m = _TS.search(raw.strip())
    if not m:
        return None
    hours, minutes, seconds = m.group(1), m.group(2), m.group(3)
    total = float(seconds) + int(minutes) * 60
    if hours:
        total += int(hours) * 3600
    return total


def fmt_timestamp(seconds: Optional[float]) -> str:
    """138.0 -> '02:18.00' (the pack's own MM:SS.ss convention)."""
    if seconds is None:
        return "--:--.--"
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return "%02d:%05.2f" % (minutes, rest)


# ---------------------------------------------------------------- data model

DIALOGUE = "dialogue"
ACTION = "action"
CAMERA = "camera"
SOUND = "sound"


@dataclass
class Block:
    """One timed unit of screenplay: a line of dialogue, an action, a shot."""

    episode: int
    scene_no: int
    scene_heading: str
    kind: str  # dialogue | action | camera | sound
    speaker: str = ""  # character name, dialogue only
    emotions: List[str] = field(default_factory=list)
    offscreen: bool = False  # (O.S.) / (V.O.) - heard but not seen
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    index: int = 0  # position within the episode

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_ts"] = fmt_timestamp(self.start)
        d["end_ts"] = fmt_timestamp(self.end)
        return d


@dataclass
class Scene:
    no: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    title: str
    path: str
    logline: str = ""
    continuity: str = ""
    scenes: List[Scene] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.blocks), default=0.0)


# ---------------------------------------------------------------- patterns

_RE_EP_NUM = re.compile(r"^EPISODE\s+NUMBER\s*:\s*(?:Episode\s*)?(\d+)", re.I)
_RE_LOGLINE = re.compile(r"^EPISODE\s+LOGLINE\s*:\s*(.+)$", re.I)
_RE_CONTINUITY = re.compile(r"^CONTINUITY\s+NOTES\s*:\s*(.+)$", re.I)
_RE_SCENE = re.compile(r"^SCENE\s+0*(\d+)\s*$", re.I)
_RE_SCENE_HEADING = re.compile(r"^SCENE\s+HEADING\s*:\s*(.+)$", re.I)
_RE_SCENE_START = re.compile(r"^SCENE\s+START\s*:\s*(.+)$", re.I)
_RE_SCENE_END = re.compile(r"^SCENE\s+END\s*:\s*(.+)$", re.I)
# **CAMERA:**  /  **MIRA (Cold, Commanding):**  /  **SOUND FX:**
_RE_LABEL = re.compile(r"^\*{2}\s*([^*]+?)\s*\*{2}\s*$")
_RE_TIMES = re.compile(
    r"^Start\s*:\s*\(?([^)|]+?)\)?\s*\|\s*End\s*:\s*\(?([^)]+?)\)?\s*$", re.I
)
_RE_EP_FROM_NAME = re.compile(r"episode[\s_-]*0*(\d+)", re.I)
_RE_PARENS = re.compile(r"\(([^()]*)\)")

# Delivery markers, not emotions: the line is heard but the speaker is not seen.
_OFFSCREEN = {"o.s.", "os", "v.o.", "vo", "off", "o.c.", "offscreen", "voice over"}
_CONTINUED = {"cont'd", "contd", "cont", "continued"}

_NON_DIALOGUE = {
    "CAMERA": CAMERA,
    "VFX": CAMERA,
    "TRANSITION": CAMERA,
    "ACTION": ACTION,
    "SOUND FX": SOUND,
    "SFX": SOUND,
    "SOUND": SOUND,
    "SOUND EFFECT": SOUND,
    "SOUND EFFECTS": SOUND,
    "MUSIC": SOUND,
}


def _split_label(label: str):
    """
    'MIRA (Cold, Commanding):'          -> (dialogue, 'MIRA',      [...], False)
    'JULIAN (O.S.) (Pained, Selfless):' -> (dialogue, 'JULIAN',    [...], True)

    A label may carry several parentheticals. Delivery markers such as (O.S.)
    and (V.O.) are pulled out separately; whatever is left is the emotion tag,
    which the scoring layer leans on heavily.
    """
    label = label.strip().rstrip(":").strip()

    emotions: List[str] = []
    offscreen = False
    for group in _RE_PARENS.findall(label):
        for part in group.split(","):
            token = part.strip()
            if not token:
                continue
            low = token.lower().rstrip(".").strip()
            if low in _OFFSCREEN or token.lower() in _OFFSCREEN:
                offscreen = True
            elif low in _CONTINUED:
                continue
            else:
                emotions.append(token)

    label_name = _RE_PARENS.sub("", label).strip().rstrip(":").strip()

    key = label_name.upper()
    if key in _NON_DIALOGUE:
        return _NON_DIALOGUE[key], "", emotions, offscreen
    # Anything else carrying a label is a character speaking.
    return DIALOGUE, label_name, emotions, offscreen


def parse_episode(path: str) -> Episode:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    stem = os.path.basename(path)
    name_match = _RE_EP_FROM_NAME.search(stem)
    ep = Episode(
        number=int(name_match.group(1)) if name_match else 0,
        title=os.path.splitext(stem)[0],
        path=path,
    )

    scene_no, scene_heading = 0, ""
    scene_start, scene_end = None, None
    i, n = 0, len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        m = _RE_EP_NUM.match(line)
        if m:
            ep.number = int(m.group(1))
            i += 1
            continue
        m = _RE_LOGLINE.match(line)
        if m:
            ep.logline = m.group(1).strip()
            i += 1
            continue
        m = _RE_CONTINUITY.match(line)
        if m:
            ep.continuity = m.group(1).strip()
            i += 1
            continue

        m = _RE_SCENE.match(line)
        if m:
            if scene_no:
                ep.scenes.append(
                    Scene(scene_no, scene_heading, scene_start or 0.0, scene_end or 0.0)
                )
            scene_no = int(m.group(1))
            scene_heading, scene_start, scene_end = "", None, None
            i += 1
            continue
        m = _RE_SCENE_HEADING.match(line)
        if m:
            scene_heading = m.group(1).strip()
            i += 1
            continue
        m = _RE_SCENE_START.match(line)
        if m:
            scene_start = parse_timestamp(m.group(1))
            i += 1
            continue
        m = _RE_SCENE_END.match(line)
        if m:
            scene_end = parse_timestamp(m.group(1))
            i += 1
            continue

        m = _RE_LABEL.match(line)
        if m:
            label = m.group(1)
            kind, speaker, emotions, offscreen = _split_label(label)

            # The next non-empty line should carry the Start/End pair.
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n:
                break
            tm = _RE_TIMES.match(lines[j].strip())
            if not tm:
                ep.warnings.append(
                    "block '%s' near line %d has no Start/End line" % (label, i + 1)
                )
                i = j
                continue

            start = parse_timestamp(tm.group(1))
            end = parse_timestamp(tm.group(2))

            # Body runs until a blank line, the next label, or a section rule.
            body, k = [], j + 1
            while k < n:
                nxt = lines[k].strip()
                if not nxt or _RE_LABEL.match(nxt) or nxt.startswith("---"):
                    break
                body.append(nxt)
                k += 1
            text = " ".join(body).strip()

            if start is None or end is None:
                ep.warnings.append("unparseable timestamps near line %d" % (j + 1))
                i = k
                continue
            if end < start:
                ep.warnings.append(
                    "end before start at %s (line %d); swapped"
                    % (fmt_timestamp(start), j + 1)
                )
                start, end = end, start

            ep.blocks.append(
                Block(
                    episode=ep.number,
                    scene_no=scene_no,
                    scene_heading=scene_heading,
                    kind=kind,
                    speaker=speaker,
                    emotions=emotions,
                    offscreen=offscreen,
                    text=text.strip('"').strip(),
                    start=start,
                    end=end,
                    index=len(ep.blocks),
                )
            )
            i = k
            continue

        i += 1

    if scene_no:
        ep.scenes.append(
            Scene(scene_no, scene_heading, scene_start or 0.0, scene_end or 0.0)
        )

    ep.blocks.sort(key=lambda b: (b.start, b.end))
    for idx, b in enumerate(ep.blocks):
        b.index = idx
    if not ep.blocks:
        ep.warnings.append("no timed blocks parsed - the format may have changed")
    return ep


def load_episodes(input_path: str) -> List[Episode]:
    """Load every .txt screenplay under input_path (a file or a directory)."""
    paths: List[str] = []
    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        for root, _dirs, files in os.walk(input_path):
            for f in sorted(files):
                if f.lower().endswith(".txt"):
                    paths.append(os.path.join(root, f))
        paths.sort()

    episodes = [parse_episode(p) for p in paths]
    for pos, ep in enumerate(episodes, start=1):
        if not ep.number:
            ep.number = pos
            ep.warnings.append("episode number inferred from file order")
    episodes.sort(key=lambda e: e.number)
    return episodes
