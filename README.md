# reflective-qa

A reflection agent for grounded question answering over NLP research papers, and an experiment on
when reflection pays for its extra calls.

The agent answers a question from passages retrieved from one paper. A critic reviews the draft against the
question and passages, with code checks as extra evidence, and the writer revises once if the critic finds problems.
Everything is scored against QASPER's gold answers, so each version of the agent is comparable with the M1 baselines.

Data: [QASPER](https://allenai.org/data/qasper) v0.3 (Dasigi et al., 2021; CC BY 4.0), dev split,
questions answerable from text only. The test split is never downloaded.

## Status

| Milestone | What | Status |
|---|---|---|
| M0 | Setup, verified data download, typed loader | done |
| M1 | BM25 retrieval + evidence recall@k; one-shot and closed-book baselines; answer-F1 by type | done |
| M2 | Reflection agent: code checks, a critic with a fixed checklist, one revise round | done |
| M3 | Experiment: one-shot vs. code feedback vs. self-critique vs. separate critic (± code feedback) | done |

## M1 baselines

200 text-only dev questions (seeded sample), Batch API, run 2026-10-11. Reproduce with
`uv run python -m reflective_qa.baselines report`.

| Run | Answer-F1 | Evidence-F1 | Says "Unanswerable" | Cost |
|---|---|---|---|---|
| Claude Sonnet 5.5, top 20 BM25 paragraphs | 0.512 | 0.621 | 16.0% | $0.95 |
| Claude Sonnet 5.5, closed-book (title + question) | 0.274 | — | 18.0% | $0.17 |
| Claude Haiku 5.5, top 5 BM25 paragraphs | 0.425 | 0.415 | 29.5% | $0.02 |

Answer-F1 matches the official QASPER evaluator; evidence-F1 is our paragraph-ID metric and is not comparable to
published numbers. 14.5% of these questions have an annotator who marked them unanswerable.

## M3 reflection results

One round of reflection on the M1 drafts, same 200 questions, run 2026-10-11 for $2.29. Reproduce with
`uv run python -m reflective_qa.reflection report`.

| Setup | Answer-F1 | Change, 95% CI | Says "Unanswerable" | Revised: better / worse | Cost |
|---|---|---|---|---|---|
| Sonnet 5.5, code feedback | 0.512 → 0.520 | +0.008 [−0.001, +0.019] | 16.0% | 8 / 7 | $0.09 |
| Sonnet 5.5, self-critique | 0.512 → 0.506 | −0.006 [−0.033, +0.020] | 16.0% → 12.0% | 14 / 17 | $1.23 |
| Haiku 5.5, code feedback | 0.425 → 0.426 | +0.001 (one answer changed) | 29.5% | 1 / 0 | $0.00 |
| Haiku 5.5, self-critique | 0.425 → 0.419 | −0.007 [−0.033, +0.016] | 29.5% → 19.0% | 14 / 9 | $0.03 |
| Haiku 5.5, Sonnet critic | 0.425 → 0.421 | −0.004 [−0.038, +0.028] | 29.5% → 24.0% | 23 / 23 | $0.47 |
| Haiku 5.5, Sonnet critic + checks | 0.425 → 0.441 | +0.015 [−0.015, +0.045] | 29.5% → 25.5% | 26 / 15 | $0.47 |

- **No setup changed answer-F1 measurably.** Every interval includes zero. The last two setups are near-replicates (the
  checks flag almost no Haiku drafts), and their 0.020 gap is a measure of noise.
- **Two effects cancel.** Critics catch wrong "Unanswerable" answers: those revisions gain +0.07 to +0.18 answer-F1. But
  when a critic rewords an answer that was already given, the revision usually scores lower (−0.07 to −0.14 in three
  of four critic setups).
- **Shortening helps where it applies.** Code feedback shortened 15 long Sonnet answers (median 28 words shorter),
  +0.106 answer-F1 on those, too few to move the total.

## Run

```bash
uv sync
uv run rqa download                                                    # QASPER train/dev + official evaluator, checksum-verified
uv run rqa stats                                                       # dev split summary
uv run rqa recall                                                      # BM25 evidence recall@k
uv run rqa score predictions.jsonl                                     # answer-F1 (official) + paragraph-ID evidence-F1
uv run --env-file .env python -m reflective_qa.baselines run sonnet-k20 --max-dollars 1.50   # needs ANTHROPIC_API_KEY in .env
uv run python -m reflective_qa.baselines report                                  # score every run on its sample
uv run python -m reflective_qa.checks data/raw/qasper-dev-v0.3.json    # code checks on the M1 drafts
uv run --env-file .env python -m reflective_qa.reflection run sonnet-self-critique --max-dollars 1.50   # one setup
uv run python -m reflective_qa.reflection report                                # every setup vs. its drafts
uv run pytest
```
