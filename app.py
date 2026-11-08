"""
Promo Finder - browser demo.

    streamlit run app.py

Deployed on Streamlit Community Cloud. It ships with a small synthetic demo
pack so it does something useful on first load, and it accepts uploaded
screenplays so a reviewer can run it on an episode it has never seen.

No screenplay from any real show is committed to this repo.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile

import streamlit as st

from promofinder.llm import build_scorer
from promofinder.parse import fmt_timestamp, load_episodes
from promofinder.rank import posting_order, rank
from promofinder.score import render_candidate, score_candidates
from promofinder.segment import build_candidates

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_pack")

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

st.title("Promo Finder")
st.markdown(
    "Reads microdrama screenplays and returns **postable 20-60 second promo "
    "clips**: exact in and out timestamps, the hook, the cliffhanger, why it "
    "works, and what is wrong with it."
)


# ----------------------------------------------------------------- input


def _save_uploads(files) -> str:
    """Write uploaded screenplays to a temp folder and return its path."""
    tmp = tempfile.mkdtemp(prefix="promofinder_")
    written = 0
    for f in files:
        name = os.path.basename(f.name)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                for member in zf.namelist():
                    if not member.lower().endswith(".txt") or member.endswith("/"):
                        continue
                    target = os.path.join(tmp, os.path.basename(member))
                    with open(target, "wb") as out:
                        out.write(zf.read(member))
                    written += 1
        elif name.lower().endswith(".txt"):
            with open(os.path.join(tmp, name), "wb") as out:
                out.write(f.getvalue())
            written += 1
    return tmp if written else ""


with st.sidebar:
    st.header("Input")
    source = st.radio(
        "Screenplays",
        ["Demo pack (2 episodes)", "Upload your own"],
        help="The demo pack is synthetic content written for this prototype. "
             "Upload .txt episode files, or a .zip of them, to run it on a "
             "show it has never seen.",
    )

    uploaded = None
    if source == "Upload your own":
        uploaded = st.file_uploader(
            "Episode files",
            type=["txt", "zip"],
            accept_multiple_files=True,
            help="One .txt per episode, or a single .zip containing them.",
        )
        st.caption(
            "Expected format: `EPISODE NUMBER:` header, `SCENE 01` blocks, and "
            "timed lines like `**MIRA (Furious):**` followed by "
            "`Start: (00:12.40) | End: (00:14.10)`."
        )

    st.header("Judged by")
    provider = st.selectbox(
        "Scorer",
        ["Rules only (no API key)", "Claude", "Gemini"],
        help="Rules only runs the full pipeline with no credentials and is the "
             "baseline the model has to beat.",
    )
    api_key = ""
    model_id = ""
    if provider == "Claude":
        api_key = st.text_input("Anthropic API key", type="password",
                                placeholder="sk-ant-...")
        model_id = st.selectbox(
            "Model",
            ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
            index=2,
            help="About $0.02 per 50-episode show on Haiku, $0.75 on Opus.",
        )
    elif provider == "Gemini":
        api_key = st.text_input("Gemini API key", type="password")
        model_id = st.text_input("Model", value="gemini-2.0-flash")

    st.header("Output")
    top_n = st.slider("Promos to return", 3, 40, 10)
    per_episode = st.slider("Candidates nominated per episode", 2, 8, 4)
    stitch = st.slider("Multi-episode promos to try", 0, 10, 4)

    run = st.button("Find promos", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Rules enforce the hard constraints: 20-60s, cuts on line boundaries, "
        "at least two spoken lines. The model only judges what survives."
    )


# ----------------------------------------------------------------- pipeline


@st.cache_data(show_spinner=False)
def prepare(input_dir: str, per_episode: int, stitch: int):
    episodes = load_episodes(input_dir)
    candidates, rejects = build_candidates(episodes, per_episode, stitch)
    return episodes, candidates, rejects


def render_results(rows, episodes, scorer_label, stats):
    ep_map = {e.number: e for e in episodes}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes read", stats["episodes"])
    c2.metric("Timed blocks", stats["blocks"])
    c3.metric("Candidates", stats["candidates"])
    c4.metric("Promos returned", len(rows))
    st.caption("Judged by %s" % scorer_label)

    if not rows:
        st.warning("No clip satisfied the constraints. Try raising the "
                   "candidates per episode.")
        return

    import csv as _csv

    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["rank", "episodes", "segments", "duration_s", "score", "verdict",
                "hook", "cliffhanger", "why", "risk"])
    for r in rows:
        cand, j = r["candidate"], r["judgement"]
        w.writerow([
            r.get("rank"),
            " / ".join(str(e) for e in cand.episodes),
            " + ".join("ep%d %s-%s" % (s.episode, fmt_timestamp(s.start),
                                       fmt_timestamp(s.end)) for s in cand.segments),
            round(cand.duration, 2), round(r["score"], 3), j.get("verdict", ""),
            j.get("hook", ""), j.get("cliffhanger", ""), j.get("why", ""),
            j.get("risk", ""),
        ])
    st.download_button("Download as CSV", buf.getvalue(), "promos.csv", "text/csv")

    for r in rows:
        cand, j = r["candidate"], r["judgement"]
        spans = "  +  ".join(
            "Episode %d  %s - %s" % (s.episode, fmt_timestamp(s.start),
                                     fmt_timestamp(s.end))
            for s in cand.segments)
        label = "**%d.**  %s   ·   %.1fs" % (r.get("rank", 0), spans, cand.duration)
        if cand.kind == "stitched":
            label += "   ·   joined"

        with st.expander(label, expanded=(r.get("rank") == 1)):
            left, right = st.columns([3, 1])
            with left:
                st.markdown("**Hook** · %s" % (j.get("hook") or "-"))
                st.markdown("**Cliffhanger** · %s" % (j.get("cliffhanger") or "-"))
                st.markdown("**Why it works** · %s" % (j.get("why") or "-"))
                st.markdown("**Weakness** · %s" % (j.get("risk") or "-"))
            with right:
                st.metric("Score", "%.2f" % r["score"])
                st.write(
                    {
                        "hook": j.get("hook_strength"),
                        "cliffhanger": j.get("cliffhanger_strength"),
                        "standalone": j.get("standalone_clarity"),
                        "punch": j.get("emotional_punch"),
                        "verdict": j.get("verdict"),
                    }
                )
            st.markdown("**The cut, line by line**")
            st.code(render_candidate(cand, ep_map), language=None)


# ----------------------------------------------------------------- run

if "ran_once" not in st.session_state:
    st.session_state.ran_once = False

should_run = run or not st.session_state.ran_once

if should_run:
    if source == "Upload your own":
        if not uploaded:
            st.info("Upload one or more episode .txt files, or a .zip of them, "
                    "then press **Find promos**.")
            st.stop()
        input_dir = _save_uploads(uploaded)
        if not input_dir:
            st.error("No .txt screenplays found in what you uploaded.")
            st.stop()
    else:
        input_dir = DEMO_DIR
        if not run and not st.session_state.ran_once:
            st.info("Showing the built-in demo pack. Use the sidebar to upload "
                    "your own episodes or switch on a model.")

    with st.spinner("Parsing screenplays and nominating candidates..."):
        try:
            episodes, candidates, _rejects = prepare(input_dir, per_episode, stitch)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error("Could not read those screenplays: %s" % exc)
            st.stop()

    if not episodes:
        st.error("No episodes parsed. Check the file format.")
        st.stop()

    warnings = [(e.number, w) for e in episodes for w in e.warnings]
    if warnings:
        with st.expander("Parser warnings (%d)" % len(warnings)):
            for num, w in warnings:
                st.write("Episode %d: %s" % (num, w))

    if provider == "Rules only (no API key)":
        scorer = build_scorer("heuristic")
    else:
        key_env = "ANTHROPIC_API_KEY" if provider == "Claude" else "GEMINI_API_KEY"
        if not api_key and not os.environ.get(key_env):
            st.warning(
                "No %s API key given, so this run used rules only. Paste a key "
                "in the sidebar to have a model judge the clips." % provider)
            scorer = build_scorer("heuristic")
        else:
            if api_key:
                os.environ[key_env] = api_key
            try:
                scorer = build_scorer(
                    "anthropic" if provider == "Claude" else "gemini", model_id)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                st.error("Could not reach %s (%s). Falling back to rules only."
                         % (provider, exc))
                scorer = build_scorer("heuristic")

    bar = st.progress(0.0, text="Judging clips...")

    def progress(i, n, key):
        bar.progress(i / n, text="Judging %s  (%d of %d)" % (key, i, n))

    judgements, failures = score_candidates(candidates, episodes, scorer, progress)
    bar.empty()

    if failures:
        with st.expander("%d batch(es) failed" % len(failures)):
            for f in failures:
                st.write("%s: %s" % (f["batch"], f["error"]))

    rows = posting_order(rank(candidates, judgements), top=top_n)
    render_results(
        rows, episodes, scorer.describe(),
        {
            "episodes": len(episodes),
            "blocks": sum(len(e.blocks) for e in episodes),
            "candidates": len(candidates),
        },
    )
    st.session_state.ran_once = True

else:
    st.info("Pick an input in the sidebar and press **Find promos**.")
