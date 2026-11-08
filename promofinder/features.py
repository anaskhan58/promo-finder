"""
Cheap, deterministic scoring of individual screenplay blocks.

This layer exists to keep the model's bill bounded. Scoring every possible
20-60s window in 50 episodes with an LLM would be tens of thousands of calls.
Instead, plain rules nominate a short list of plausible windows per episode and
the model only judges that short list.

Nothing here is a final verdict. These scores decide what gets *looked at*,
not what gets published, so a rough signal is good enough. The weights were
tuned by hand against the first few episodes of the input pack.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .parse import ACTION, CAMERA, DIALOGUE, SOUND, Block

# ------------------------------------------------------------------ lexicons

# Physical shock: the thing that stops a thumb mid-scroll.
VIOLENCE = {
    "slap", "slaps", "slapped", "strike", "strikes", "struck", "punch", "shove",
    "shoves", "shoved", "grabs", "seizes", "throws", "hurls", "drags", "dragged",
    "blood", "bleeding", "wound", "stab", "stabs", "blade", "dagger", "sword",
    "gun", "pistol", "shot", "shoots", "collapse", "collapses", "crumples",
    "falls", "slams", "smash", "smashes", "shatters", "choke", "chokes",
    "strangle", "whip", "lash", "kick", "kicks", "blow", "bruise", "scar",
}

# Social shock: humiliation, betrayal, status collapse.
BETRAYAL = {
    "betray", "betrayed", "betrayal", "abandon", "abandoned", "altar", "jilt",
    "humiliate", "humiliated", "humiliation", "disgrace", "disowned", "shame",
    "liar", "lied", "lying", "cheat", "cheated", "affair", "mistress",
    "bastard", "whore", "worthless", "beggar", "servant", "peasant", "banish",
    "banished", "exile", "annul", "divorce", "widow", "orphan", "ruined",
    "mock", "mocking", "spit", "sneer", "scorn", "dismiss", "discard",
}

# Reveal: identity, secret, status flip - the engine of this genre.
REVEAL = {
    "duke", "duchess", "crown", "royal", "prince", "king", "queen", "title",
    "heir", "inheritance", "fortune", "secret", "truth", "reveal", "reveals",
    "revealed", "discover", "discovers", "identity", "mask", "disguise",
    "alive", "dead", "pregnant", "child", "son", "daughter", "father",
    "mother", "real", "actually", "letter", "seal", "crest", "ring", "pendant",
    "scar", "proof", "evidence", "witness", "confess", "confession",
}

# Threat and stakes.
THREAT = {
    "kill", "kills", "murder", "murdered", "die", "dies", "death", "execute",
    "execution", "hang", "poison", "poisoned", "destroy", "ruin", "regret",
    "revenge", "vengeance", "pay", "warned", "warning", "threat", "danger",
    "arrest", "prison", "trial", "judge", "verdict", "guilty", "sentence",
    "war", "battle", "enemy", "traitor", "treason",
}

# Emotion tags the pack already supplies in the speaker parenthetical.
HOT_EMOTIONS = {
    "furious": 1.0, "screaming": 1.0, "shouting": 0.9, "enraged": 1.0,
    "venomous": 0.9, "cruel": 0.8, "vicious": 0.9, "savage": 0.9,
    "cold": 0.6, "icy": 0.7, "commanding": 0.6, "authoritative": 0.5,
    "mocking": 0.7, "smug": 0.6, "sneering": 0.7, "contemptuous": 0.7,
    "desperate": 0.8, "pleading": 0.7, "begging": 0.8, "trembling": 0.6,
    "shocked": 0.9, "stunned": 0.9, "horrified": 1.0, "terrified": 1.0,
    "defiant": 0.8, "fierce": 0.8, "resolute": 0.5, "threatening": 1.0,
    "menacing": 1.0, "accusing": 0.8, "incredulous": 0.7, "disbelieving": 0.7,
    "broken": 0.7, "anguished": 0.8, "weeping": 0.6, "sobbing": 0.7,
}

# Camera grammar that signals a punch landing.
HOT_SHOTS = {
    "close-up", "extreme close-up", "push-in", "whip pan", "snap zoom",
    "insert", "slow motion", "crash zoom", "cut to black", "smash cut",
    "reaction shot", "handheld",
}

# Sound design that marks an act break.
HOT_SOUND = {
    "cliffhanger", "sting", "stinger", "ominous", "swells", "swell", "drone",
    "crash", "thud", "slap", "gasp", "scream", "shatter", "heartbeat",
    "silence", "boom", "impact",
}

# Words that signal an unresolved ending rather than a closed one.
UNRESOLVED = {
    "who", "what", "why", "how", "where", "when", "whose",
    "wait", "stop", "no", "impossible", "cannot", "never",
}

# Lines that carry no meaning without the line before them. A promo that opens
# on one of these is unintelligible in the first 3 seconds, however dramatic the
# scene around it is.
ACKNOWLEDGEMENTS = {
    "precisely", "exactly", "indeed", "correct", "naturally", "certainly",
    "quite so", "of course", "as you wish", "very well", "agreed", "obviously",
}

_WORD = re.compile(r"[a-z']+")


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _lexicon_hits(words: Iterable[str], lexicon: set) -> int:
    return sum(1 for w in words if w in lexicon)


def _emotion_heat(block: Block) -> float:
    if not block.emotions:
        return 0.0
    return max((HOT_EMOTIONS.get(e.strip().lower(), 0.0) for e in block.emotions),
               default=0.0)


def _phrase_hits(text: str, phrases: set) -> int:
    low = text.lower()
    return sum(1 for p in phrases if p in low)


# ------------------------------------------------------------------ scoring


def punch(block: Block) -> float:
    """
    How much does this block grab attention, on its own, 0..1-ish.

    Used both for hook candidates and for judging whether the middle of a
    window stays interesting.
    """
    words = _words(block.text)
    if not words:
        return 0.0

    score = 0.0
    score += 0.55 * min(2, _lexicon_hits(words, VIOLENCE))
    score += 0.40 * min(2, _lexicon_hits(words, BETRAYAL))
    score += 0.35 * min(2, _lexicon_hits(words, REVEAL))
    score += 0.45 * min(2, _lexicon_hits(words, THREAT))

    if block.kind == DIALOGUE:
        score += 0.8 * _emotion_heat(block)
        if "!" in block.text:
            score += 0.30
        if "?" in block.text:
            score += 0.20
        # Short lines land harder than speeches.
        if len(words) <= 8:
            score += 0.25
        elif len(words) > 25:
            score -= 0.20
        # Direct address is confrontational.
        if re.search(r"\byou(?:'re| are|r)?\b", block.text, re.I):
            score += 0.15
    elif block.kind == ACTION:
        score += 0.20
    elif block.kind == CAMERA:
        score += 0.20 * _phrase_hits(block.text, HOT_SHOTS)
        score -= 0.10  # shot description alone is not a hook
    elif block.kind == SOUND:
        score += 0.25 * _phrase_hits(block.text, HOT_SOUND)

    return max(0.0, score)


def opens_mid_thought(block: Block) -> bool:
    """
    True when this line only makes sense as a reply to the line before it.

    Found by eval: the rules were happily opening promos on '...no one will
    remember what was written inside.' and on 'Precisely.' Both are strong
    beats inside a scene and useless as the first second of a promo.
    """
    text = (block.text or "").strip()
    if not text:
        return False
    if text.startswith(("...", "…", "-", "–", "—")):
        return True
    stripped = text.strip(".!?… ").lower()
    if stripped in ACKNOWLEDGEMENTS:
        return True
    # A line of dialogue that starts lowercase is a continuation.
    if block.kind == DIALOGUE and text[0].islower():
        return True
    return False


def hook_strength(block: Block) -> float:
    """How well this block works as the *first thing* a scroller sees."""
    score = punch(block)
    if opens_mid_thought(block):
        score *= 0.25
    # A hook must be legible without setup. Dialogue and physical action read
    # instantly; a camera direction usually does not.
    if block.kind == DIALOGUE:
        score *= 1.25
    elif block.kind == ACTION:
        score *= 1.15
    elif block.kind == CAMERA:
        score *= 0.55
    elif block.kind == SOUND:
        score *= 0.40
    # A voice with no face on screen is a weak scroll-stopper.
    if block.offscreen:
        score *= 0.65
    # Very long blocks make a slow open.
    if block.duration > 8:
        score *= 0.8
    return score


def cliff_strength(block: Block, is_episode_end: bool = False,
                   is_scene_end: bool = False) -> float:
    """How well this block works as a cut point that leaves things hanging."""
    words = _words(block.text)
    score = 0.0

    score += 0.50 * min(2, _lexicon_hits(words, REVEAL))
    score += 0.55 * min(2, _lexicon_hits(words, THREAT))
    score += 0.35 * min(2, _lexicon_hits(words, BETRAYAL))
    score += 0.30 * min(2, _lexicon_hits(words, VIOLENCE))

    if block.kind == DIALOGUE:
        score += 0.6 * _emotion_heat(block)
        if block.text.rstrip().endswith("?"):
            score += 0.70  # an open question is the cheapest cliffhanger
        if block.text.rstrip().endswith(("...", "-", "--")):
            score += 0.45  # interrupted mid-thought
        if words and words[0] in UNRESOLVED:
            score += 0.25
        if len(words) <= 10:
            score += 0.20
    elif block.kind == SOUND:
        score += 0.45 * _phrase_hits(block.text, HOT_SOUND)
    elif block.kind == CAMERA:
        if "cut to black" in block.text.lower():
            score += 0.60
        score += 0.15 * _phrase_hits(block.text, HOT_SHOTS)
    elif block.kind == ACTION:
        score += 0.20

    if is_scene_end:
        score += 0.20
    if is_episode_end:
        score += 0.35  # the writers already engineered this beat as a hook out

    return max(0.0, score)
