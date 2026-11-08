"""
The judgement layer: one swappable scorer behind one interface.

The rules in segment.py decide *what gets looked at*. This decides *how good it
is*, which is a taste question rules are bad at. Every provider returns the same
JSON shape, so the model can be swapped without touching anything else:

    --provider anthropic --model claude-opus-5-5     (default)
    --provider anthropic --model claude-haiku-5-5    (same code, ~40x cheaper)
    --provider openai    --model gpt-4o-mini
    --provider ollama    --model llama3.1            (local, free)
    --provider heuristic                             (no API key at all)

The heuristic provider is not a toy fallback: it is what lets the repo be
cloned and run end to end with no credentials, and it is the baseline the
model has to beat in the eval.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

SYSTEM_PROMPT = """You pick promo clips for vertical microdramas (the ReelShort / DramaBox format). \
These clips get posted to Instagram Reels, TikTok and YouTube Shorts to pull in viewers who have never heard of the show.

You will be given candidate clips from one episode, as transcripts with timestamps. \
Each clip has already been checked for length and cut points. Your job is to judge how well each one would actually perform, and to be hard to impress.

Judge each clip on four things, 0 to 5:

- hook_strength: does something in the FIRST 3 SECONDS stop a scrolling thumb? A slap, a reveal, a public humiliation, an accusation, a shocking line. A clip that opens on scene-setting or a calm exchange scores 0-1, no matter how good it gets later.
- cliffhanger_strength: does it cut on an unresolved moment that forces the viewer to find out what happens next? A closed, resolved or comforting ending scores 0-1.
- standalone_clarity: could someone who has never seen the show follow it? Count how much the clip depends on names, titles or events it never explains. Needing one unexplained name is fine; needing three is not.
- emotional_punch: how strong is the feeling - rage, humiliation, vindication, dread?

Rules:
- Judge only what is in the clip. The episode logline is given for your own orientation; a clip that only makes sense if you have read it should score low on standalone_clarity.
- hook must quote or describe the actual opening moment, not a summary of the clip.
- cliffhanger must name the specific unresolved thing.
- why is ONE sentence on why it would perform.
- risk is the honest weakness of this clip, in one short phrase. Every clip has one. Do not write "none".
- verdict is "post" only if you would put it live today, "maybe" if it needs a tighter cut, "reject" if it should not go out.

Be sparing with 5s and sparing with "post". Most clips in any episode are not promos."""

JUDGEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "hook_strength": {"type": "number"},
                    "cliffhanger_strength": {"type": "number"},
                    "standalone_clarity": {"type": "number"},
                    "emotional_punch": {"type": "number"},
                    "hook": {"type": "string"},
                    "cliffhanger": {"type": "string"},
                    "why": {"type": "string"},
                    "risk": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["post", "maybe", "reject"]},
                },
                "required": [
                    "id", "hook_strength", "cliffhanger_strength",
                    "standalone_clarity", "emotional_punch", "hook",
                    "cliffhanger", "why", "risk", "verdict",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o-mini",
    # Model ids retire. gemini-2.0-flash and gemini-2.5-flash both 404 now,
    # which made --provider gemini look broken rather than out of date.
    # `python -m promofinder models --provider gemini` lists what is live.
    "gemini": "gemini-3.6-flash",
    "ollama": "llama3.1",
}


# ---------------------------------------------------------------- JSON repair


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------- providers


class Scorer:
    name = "base"

    def judge(self, payload: str) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class HeuristicScorer(Scorer):
    """
    No API key required. Converts the rule-layer signals into the same JSON
    shape, so the whole pipeline runs offline and the model has a baseline to
    beat. It cannot read a scene, so it never claims to.
    """

    name = "heuristic"

    def __init__(self):
        self.context: Dict[str, dict] = {}

    def judge(self, payload: str) -> dict:
        clips = []
        for cid, ctx in self.context.items():
            sig = ctx["signals"]
            hook = min(5.0, sig.get("hook", 0.0) * 2.0)
            cliff = min(5.0, sig.get("cliff", 0.0) * 2.2)
            clarity = min(5.0, sig.get("clarity", 0.0) * 5.0)
            punch = min(5.0, sig.get("sustain", 0.0) * 2.5)
            verdict = "post" if (hook >= 3 and cliff >= 3) else (
                "maybe" if hook >= 2 or cliff >= 2 else "reject")
            clips.append({
                "id": cid,
                "hook_strength": round(hook, 2),
                "cliffhanger_strength": round(cliff, 2),
                "standalone_clarity": round(clarity, 2),
                "emotional_punch": round(punch, 2),
                "hook": ctx.get("first_line", ""),
                "cliffhanger": ctx.get("last_line", ""),
                "why": "Keyword and emotion-tag density only; no scene was read.",
                "risk": "Scored by rules, not judgement - treat as a shortlist, not a pick.",
                "verdict": verdict,
            })
        return {"clips": clips}

    def describe(self) -> str:
        return "heuristic (lexical rules, no model)"


class AnthropicScorer(Scorer):
    name = "anthropic"

    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 16000):
        import anthropic  # imported lazily so the repo runs without the SDK

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self._use_beta = True
        self._use_schema = True
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def describe(self) -> str:
        return "anthropic:%s (effort=%s)" % (self.model, self.effort)

    def _request(self, payload: str):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
        output_config: Dict[str, object] = {"effort": self.effort}
        if self._use_schema:
            output_config["format"] = {
                "type": "json_schema",
                "schema": JUDGEMENT_SCHEMA,
            }
        kwargs["output_config"] = output_config

        if self._use_beta:
            try:
                # Server-side fallback: if a safety classifier declines the
                # request, the API re-runs it on a fallback model in the same
                # call instead of the run simply dying mid-pass.
                return self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                )
            except Exception:
                self._use_beta = False
        try:
            return self.client.messages.create(**kwargs)
        except Exception:
            if self._use_schema:
                # Older model or endpoint without structured outputs: fall back
                # to asking for JSON in the prompt and repairing it.
                self._use_schema = False
                kwargs["output_config"] = {"effort": self.effort}
                kwargs["messages"] = [{
                    "role": "user",
                    "content": payload + "\n\nReply with JSON only, matching the "
                                         "schema described above.",
                }]
                return self.client.messages.create(**kwargs)
            raise

    def judge(self, payload: str) -> dict:
        last_err = None
        for attempt in range(3):
            try:
                resp = self._request(payload)
                if getattr(resp, "stop_reason", None) == "refusal":
                    detail = getattr(resp, "stop_details", None)
                    raise RuntimeError(
                        "model declined this batch (%s)"
                        % getattr(detail, "category", "unknown"))
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
                    self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text")
                data = extract_json(text)
                if data and "clips" in data:
                    return data
                last_err = RuntimeError("no parseable JSON in response")
            except Exception as exc:  # noqa: BLE001 - retried below, raised after
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("anthropic scoring failed: %s" % last_err)


class OpenAIScorer(Scorer):
    name = "openai"

    def __init__(self, model: str, max_tokens: int = 8000):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI()

    def describe(self) -> str:
        return "openai:%s" % self.model

    def judge(self, payload: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload + "\n\nReply with JSON only."},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        data = extract_json(resp.choices[0].message.content)
        if not data or "clips" not in data:
            raise RuntimeError("no parseable JSON from openai")
        return data


class OllamaScorer(Scorer):
    """Local models via Ollama, for running with no account and no spend."""

    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def describe(self) -> str:
        return "ollama:%s" % self.model

    def judge(self, payload: str) -> dict:
        import requests

        resp = requests.post(
            self.host + "/api/chat",
            json={
                "model": self.model,
                "format": "json",
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload + "\n\nReply with JSON only."},
                ],
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = extract_json(resp.json()["message"]["content"])
        if not data or "clips" not in data:
            raise RuntimeError("no parseable JSON from ollama")
        return data


class GeminiScorer(Scorer):
    name = "gemini"

    ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
                "%s:generateContent?key=%s")

    def __init__(self, model: str):
        self.model = model
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError("set GEMINI_API_KEY to use --provider gemini")
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def describe(self) -> str:
        return "gemini:%s" % self.model

    def judge(self, payload: str) -> dict:
        import requests

        url = self.ENDPOINT % (self.model, self.key)
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": payload + "\n\nReply with JSON only."}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }

        # A whole-show run is one request per episode. Without a retry a single
        # transient 503 - and the newest flash models return them under load -
        # takes down a run that is 50 episodes deep.
        last_err = None
        for attempt in range(4):
            try:
                resp = requests.post(url, json=body, timeout=300)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(
                        "gemini %s on attempt %d" % (resp.status_code, attempt + 1))
                    time.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                payload_json = resp.json()

                meta = payload_json.get("usageMetadata") or {}
                self.usage["input_tokens"] += meta.get("promptTokenCount", 0) or 0
                self.usage["output_tokens"] += meta.get("candidatesTokenCount", 0) or 0

                text = payload_json["candidates"][0]["content"]["parts"][0]["text"]
                data = extract_json(text)
                if data and "clips" in data:
                    return data
                last_err = RuntimeError("no parseable JSON from gemini")
            except Exception as exc:  # noqa: BLE001 - retried, raised below
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("gemini scoring failed: %s" % last_err)


def list_models(provider: str) -> List[str]:
    """
    Model ids that this provider will actually serve today.

    Exists because they retire: the default here was gemini-2.0-flash, which
    now 404s, and the failure reads like a broken integration rather than a
    stale string. Swapping models is only cheap if you can see the choices.
    """
    if provider == "gemini":
        import requests

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY to list gemini models")
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key, "pageSize": 200}, timeout=60)
        resp.raise_for_status()
        return sorted(
            m["name"].replace("models/", "")
            for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", []))

    if provider == "anthropic":
        import anthropic

        return sorted(m.id for m in anthropic.Anthropic().models.list(limit=100).data)

    if provider == "openai":
        from openai import OpenAI

        return sorted(m.id for m in OpenAI().models.list().data)

    if provider == "ollama":
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=30)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))

    if provider == "heuristic":
        return []
    raise ValueError("unknown provider: %s" % provider)


def build_scorer(provider: str = "auto", model: Optional[str] = None,
                 effort: str = "medium") -> Scorer:
    """Pick a scorer. 'auto' uses Claude if a credential is present, else rules."""
    if provider == "auto":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY")
                       or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if not has_key:
            # An `ant auth login` profile also works, so try and fall back.
            try:
                return AnthropicScorer(model or DEFAULT_MODELS["anthropic"], effort)
            except Exception:
                return HeuristicScorer()
        provider = "anthropic"

    model = model or DEFAULT_MODELS.get(provider)
    if provider == "anthropic":
        return AnthropicScorer(model, effort)
    if provider == "openai":
        return OpenAIScorer(model)
    if provider == "gemini":
        return GeminiScorer(model)
    if provider == "ollama":
        return OllamaScorer(model)
    if provider == "heuristic":
        return HeuristicScorer()
    raise ValueError("unknown provider: %s" % provider)
