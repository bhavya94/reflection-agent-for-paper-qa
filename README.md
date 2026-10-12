# reflective-qa

Does a review step make an AI agent answer questions about research papers better? This repo builds the agent, runs
the experiment, and reports the answer: **not on average**. A reviewer fixes answers where the agent said
"Unanswerable" although the paper had the answer. But it also rewrites good answers into worse ones, and the two
effects cancel.

**Results page:** https://bhavya94.github.io/reflective-qa/

**Words used here and in the code:** the code calls a *first answer* a *draft*, and the *reviewer* the *critic*.
One round of review and revision is *reflection* (`rqa reflect`). A *setup* is one version of the review step;
setups without a reviewer model use *code feedback* only.

## The task

A person asks a question about one research paper. The AI must answer from the paper's text and name the paragraphs
that support its answer. If the paper does not contain the answer, the correct reply is "Unanswerable".

The questions come from [QASPER](https://allenai.org/data/qasper) v0.3 (Dasigi et al., 2021; CC BY 4.0). It is a
public dataset of questions about natural language processing papers. Experts read each full paper and wrote the
reference answers; some questions have two or more expert answers. We use 200 questions from the development split,
picked at random with a fixed seed. Each needs only the paper's text, not its figures or tables. 14.5% of them have an
expert who answered "Unanswerable". The test split is never extracted or read; only the official evaluator comes from
its archive.

## How the agent works

For each question:

1. **Search.** BM25 keyword search picks the paragraphs that best match the question.
2. **Answer.** A writer model gives a short answer and the IDs of the paragraphs it used.
3. **Review.** Rule-based code checks (no model call) flag long answers, citation errors and yes/no format problems.
   In four of the six setups, a reviewer model also checks the answer against the same paragraphs, using a fixed
   checklist of seven problems.
4. **Revise once.** If the review found problems, the writer gets the list and answers again. If not, the first answer
   stays. Without a reviewer model, the list is the code checks' findings. With one, it is the reviewer's problems; the
   code checks then reach only the reviewer, and only in the last setup below.

There are two writers. Claude Sonnet 5.5 gets the top 20 paragraphs. Claude Haiku 5.5 is smaller and cheaper, and it
gets only 5, so its first answers have more to fix. A third run, Sonnet with no paragraphs (closed book), shows how much
the paragraphs help. The six setups:

| Setup (name in `configs/reflection.toml`) | Writer | Who reviews | Reviewer sees the code checks |
|---|---|---|---|
| `sonnet-code-feedback` | Sonnet 5.5 | No model: the code checks only | — |
| `haiku-code-feedback` | Haiku 5.5 | No model: the code checks only | — |
| `sonnet-self-critique` | Sonnet 5.5 | The writer itself | No |
| `haiku-self-critique` | Haiku 5.5 | The writer itself | No |
| `haiku-sonnet-critic` | Haiku 5.5 | Sonnet 5.5 | No |
| `haiku-sonnet-critic-checks` | Haiku 5.5 | Sonnet 5.5 | Yes |

The last two setups are almost the same, because the code checks flag almost no Haiku answers. The gap between their
results shows how much the scores move by chance.

## How answers are scored

- **Answer-F1** (0 to 1) measures how many words an answer shares with the expert answer, after the official QASPER
  evaluator's normalization. Extra words lower it, and so do missing words. When experts disagree, the best match
  counts. A correct answer in other words can score low, so use the scores to compare setups on the same questions,
  not as an absolute grade.
- **Evidence-F1** is the same idea for citations: the overlap between the cited paragraphs and the paragraphs the
  experts marked. It compares paragraph IDs instead of evidence text, and it counts only questions where an expert cited
  a paragraph. So it is not comparable to published QASPER numbers.
- **Says "Unanswerable"** is the share of questions where the agent gave that answer.
- **Change and 95% interval:** the change is the average answer-F1 after review minus before, on the same questions.
  The interval comes from a paired bootstrap with 5,000 resamples. If it includes zero, the data can't tell the change
  from chance.

## Results

All runs used Anthropic's Message Batches API, which costs half the normal price. The run files are not in the repo
(`data/` is git-ignored). The `report` commands re-score saved run files; to make new ones, use the paid `run` commands
under [Run it](#run-it).

### One answer, no review

200 questions, run 2026-10-11 for $1.14. Re-score with `uv run rqa baselines report`.

| Writer run (name in `configs/answering.toml`) | Answer-F1 | Evidence-F1 | Says "Unanswerable" | Cost |
|---|---|---|---|---|
| Claude Sonnet 5.5, top 20 BM25 paragraphs (`sonnet-k20`) | 0.512 | 0.621 | 16.0% | $0.95 |
| Claude Sonnet 5.5, no paragraphs: title and question only (`sonnet-closed-book`) | 0.274 | — | 18.0% | $0.17 |
| Claude Haiku 5.5, top 5 BM25 paragraphs (`haiku-k5`) | 0.425 | 0.415 | 29.5% | $0.02 |

### One round of review

The same 200 questions and first answers, run 2026-10-11 for $2.29. Re-score with `uv run rqa reflect report`.
"Better / worse" counts the changed answers whose answer-F1 went up or down; ties are left out.

| Setup | Answer-F1 | Change, 95% interval | Says "Unanswerable" | Changed answers: better / worse | Cost |
|---|---|---|---|---|---|
| `sonnet-code-feedback` | 0.512 → 0.520 | +0.008 [−0.001, +0.019] | 16.0% | 8 / 7 | $0.09 |
| `haiku-code-feedback` | 0.425 → 0.426 | +0.001 (one answer changed) | 29.5% | 1 / 0 | $0.00 |
| `sonnet-self-critique` | 0.512 → 0.506 | −0.006 [−0.033, +0.020] | 16.0% → 12.0% | 14 / 17 | $1.23 |
| `haiku-self-critique` | 0.425 → 0.419 | −0.007 [−0.033, +0.016] | 29.5% → 19.0% | 14 / 9 | $0.03 |
| `haiku-sonnet-critic` | 0.425 → 0.421 | −0.004 [−0.038, +0.028] | 29.5% → 24.0% | 23 / 23 | $0.47 |
| `haiku-sonnet-critic-checks` | 0.425 → 0.441 | +0.015 [−0.015, +0.045] | 29.5% → 25.5% | 26 / 15 | $0.47 |

Costs are for the review and the revision only; the first answers cost what the table above shows.

- **No setup changed answer-F1 measurably.** Every interval includes zero. The two near-identical setups ended about
  0.02 apart, so smaller differences mean nothing at this size.
- **Two effects cancel.** Reviewers catch wrong "Unanswerable" answers: those revisions gain 0.07 to 0.18 answer-F1
  on average. But when a reviewer rewords an answer that was already given, the revision usually scores lower. In
  three of the four setups with a reviewer model, those revisions lost 0.07 to 0.14 on average.
- **Shortening helps where it applies.** Code feedback shortened 15 long Sonnet answers (a median of 29 words
  shorter), and those gained 0.106 answer-F1 on average. They were too few to move the total.

The whole project cost $3.77 of a $20 API budget, small pilot runs included.

## Limits

- Answer-F1 rewards shared words, not meaning, so it can miss a correct answer in other words. This is a known,
  accepted limit of this experiment. The next step is an evaluation that judges meaning.
- We adjusted the answer and revise prompts after small pilot runs. Those pilots used the first questions of the same
  200-question sample.
- One review round, one dataset, two models.

## Run it

Every step runs through one command, `rqa`. Reports and checks read local files and cost nothing. Commands that call
the API need `ANTHROPIC_API_KEY` in `.env`, and a run refuses to start without a `--max-dollars` budget.

```bash
uv sync
uv run rqa download                       # QASPER train/dev + the official evaluator, checksum-verified
uv run rqa stats                          # a summary of the development split
uv run rqa recall                         # BM25: how much of the expert evidence the top k paragraphs contain
uv run rqa score predictions.jsonl        # answer-F1 and evidence-F1 for any predictions file
uv run --env-file .env rqa baselines estimate                        # cost of the first answers (free token counts)
uv run --env-file .env rqa baselines run sonnet-k20 --max-dollars 1.50
uv run rqa baselines report               # score every writer run on its sample
uv run rqa checks                         # how often each code check fires on the first answers
uv run --env-file .env rqa reflect run sonnet-self-critique --max-dollars 1.50   # one setup; the cap is per stage
uv run rqa reflect report                 # every setup against its first answers
uv run rqa site                           # the results page, into site/index.html
uv run pytest
```

## Code layout

One package per responsibility under `src/reflective_qa/`. Imports only point down, with no cycles, and
`tests/test_layers.py` enforces it:

1. `data` is at the bottom. `retrieval` and `eval` build on it, and `agent` uses those three.
2. `runs` knows only files and the API.
3. `experiments` wire the agent, runs and eval together.
4. `site` and `cli` sit on top.

| Package | What lives there |
|---|---|
| `data/` | QASPER records, `QasperSplit` (lookups and the seeded question sample), the download |
| `retrieval/` | BM25: `PaperRetriever`, `Retriever` (one index per paper), `RecallEvaluator` |
| `agent/` | `Writer`, `CodeChecks`, `Critic`, `ReflectionAgent`: they build requests and read replies; no API calls, no writes |
| `runs/` | `RunStore` (run files named by fingerprint), `BatchRunner` (Batch API, budget guard), `Costs`, `CostEstimator` |
| `eval/` | answer-F1 and evidence-F1 (pure functions), `Scorer`, `RevisionAnalysis` (paired bootstrap) |
| `experiments/` | `Experiment` (shared sample, run files, prices, API client), `BaselineExperiment`, `ReflectionExperiment` |
| `site/` | `ResultsCollector` and the self-contained results page |
| `cli.py` | the `rqa` command, the only module that parses arguments or prints |
