# Promo Finder

Reads a microdrama screenplay pack and returns a ranked list of postable promo
clips, each with an episode, exact in and out timestamps, the hook, the
cliffhanger, one line on why it works, and one line on what is wrong with it.

Built for a "find the promos" brief and run against a 50-episode
English-language microdrama. That pack is confidential and is not in this repo;
the bundled `demo_pack/` is synthetic content written for this prototype.

## Run it

No API key needed for the first command. It runs the whole pipeline on rules
alone, which is also the baseline the model has to beat.

```bash
python -m promofinder run --input demo_pack --out outputs --provider heuristic
```

With a model doing the judging (this is the intended mode):

```bash
# Claude - the default when ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY=sk-ant-...
python -m promofinder run --input <your-episode-folder> --out outputs

# or any other model, same code path
python -m promofinder run --input <your-episode-folder> --out outputs --provider anthropic --model claude-haiku-5-5
python -m promofinder run --input <your-episode-folder> --out outputs --provider gemini   --model gemini-2.0-flash
python -m promofinder run --input <your-episode-folder> --out outputs --provider ollama   --model llama3.1
```

`--input` points at a folder; every `.txt` under it is treated as one episode.
Nested folders are fine.

### Other commands

```bash
python -m promofinder inspect --input <your-episode-folder>      # what the parser saw
python -m promofinder label   --pred outputs/promos.json --out eval/labels.csv
python -m promofinder eval    --pred outputs/promos_all.json --labels eval/labels.csv
```

## Tests

```bash
python tests/test_pipeline.py     # no pytest needed
```

Ten checks on the synthetic demo pack: timestamps round-trip, speaker labels
with delivery markers split correctly, every published clip is 20-60s, every cut
lands on a real line boundary, the per-episode cap holds, and every clip carries
a hook, cliffhanger, rationale and weakness. The last check loads the
Streamlit app headlessly and asserts it renders promos with no API key; it
skips if Streamlit is not installed.

## The hosted demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app ships with `demo_pack/`, two synthetic episodes written for this
prototype, so it produces promos on first load with no uploads and no API key.
Switch the sidebar to **Upload your own** to drop in episode `.txt` files, or a
`.zip` of them, and run it on a show it has never seen.

### Deploying it free

1. Push this folder to a **public** GitHub repo. The `.gitignore` already keeps
   the show pack and everything derived from it out, so only code and the
   synthetic demo pack are published.
2. Go to `share.streamlit.io`, pick the repo, set the main file to `app.py`.
3. Deploy. `requirements.txt` is picked up automatically.

Leave the API key boxes empty in the deployed app. A key pasted into the
sidebar lives only in that session, and nothing is committed. Anyone opening
the link gets the rules-only run, which needs no credentials.

## What comes out

| File | For |
|---|---|
| `promos.csv` | the editor: rank, episode, in/out, hook, cliffhanger, why, weakness |
| `promos.md` | reading the shortlist |
| `promos.json` | the published top N, for the next tool in the chain |
| `promos_all.json` | every judged clip in score order, for the eval |
| `run_report.json` | parse warnings, rejected candidates, failed batches, token usage |

## How it works

Four stages. The split between rules and model is the point of the design.

1. **Parse** (`parse.py`) - screenplay to a timeline of timed blocks: dialogue
   with speaker and emotion tag, action, camera, sound. Blocks overlap, so the
   timeline is tracks, not a sequence. Case-insensitive throughout, because the
   pack mixes `Scene start:` and `SCENE START:`.
2. **Nominate** (`segment.py`, `features.py`) - rules propose a handful of
   windows per episode. Hard constraints are enforced here, not asked of the
   model: 20 to 60 seconds, cuts land on block boundaries so a clip never starts
   mid-line, at least two spoken lines. Lexical scoring picks which windows are
   worth a model's attention. This keeps the bill bounded: scoring every legal
   window would be tens of thousands of calls.
3. **Judge** (`llm.py`, `score.py`) - the model scores hook, cliffhanger,
   standalone clarity and emotional punch out of 5, and writes the hook,
   cliffhanger, rationale and weakness. Candidates are batched one request per
   episode, so a 50-episode show is 51 calls.
4. **Rank** (`rank.py`) - weighted score, then a second pass for posting order
   that caps how many clips come from one episode, so a week of posts is not
   ten cuts of the same fight.

### Swapping the model

Everything model-specific is behind `build_scorer()` in `llm.py`. A provider is
one class with one method that returns the shared JSON shape. The Anthropic
path uses structured outputs so the response is schema-valid, and degrades to
prompt-and-repair if the endpoint does not support it.

## Cost

Measured on this pack: 51 requests, about 146k input and 8k output tokens.

| Model | Per 50-episode show |
|---|---|
| claude-opus-5-5 | $0.75 |
| claude-sonnet-5-5 | $0.37 |
| claude-haiku-5-5 | $0.02 |

At one new show a week that is under forty dollars a year on the top model.
Parsing and candidate generation take about 16 seconds and cost nothing.

## Quality

`eval/labels.csv` holds 27 candidates labelled by hand after reading each
transcript, on a 0 to 3 scale where 2 or more means an editor could use it.
`python -m promofinder eval` scores the ranking against them. Numbers from the
rules-only baseline, and the known failures, are in the solution doc.

## Confidentiality

The show pack is the studio's own content. `.gitignore` keeps both the pack and
anything derived from it out of the repo, because `promos.md` and `promos.csv`
quote dialogue directly. Share the raw outputs through a link-only folder, not
a public repo.

## Limits

- It reads screenplays, not video. It cannot see whether a take landed.
- Timestamps are the script's, so they are only as accurate as the script.
- Stitched multi-episode promos are generated but the rules cannot tell a
  coherent join from a jarring one. They are penalised heavily and need a model
  in the loop to be trusted.
- The lexicons in `features.py` are tuned to period revenge drama. A different
  genre needs different words, though the structural signals carry over.
