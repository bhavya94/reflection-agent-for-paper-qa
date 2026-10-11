# reflective-qa

A reflection agent for grounded question answering over NLP research papers, and an experiment on
when reflection pays for its extra calls.

The agent answers a question from passages retrieved from one paper. A critic checks the draft, and
the writer revises until the checks pass or a round limit is hit. The critic's checks come from error
analysis of real outputs (M2), not from criteria written up front.

Data: [QASPER](https://allenai.org/data/qasper) v0.3 (Dasigi et al., 2021; CC BY 4.0), dev split,
questions answerable from text only. The test split is never downloaded.

## Status

| Milestone | What | Status |
|---|---|---|
| M0 | Setup, verified data download, typed loader | done |
| M1 | BM25 retrieval + evidence recall@k; one-shot and closed-book baselines; answer-F1 by type | done |
| M2 | Error analysis: read ~100 outputs, name failure modes, label Pass/Fail | |
| M3 | Critic (code checks + one binary LLM check per failure mode) and revise loop | |
| M4 | Experiment: one-shot vs. self-critique vs. code feedback vs. separate critic | |

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

## Run

```bash
uv sync
uv run python -m reflective_qa.download                                # QASPER train/dev + official evaluator, checksum-verified
uv run python -m reflective_qa.stats data/raw/qasper-dev-v0.3.json     # dev split summary
uv run python -m reflective_qa.retrieval_eval data/raw/qasper-dev-v0.3.json   # BM25 evidence recall@k
uv run python -m reflective_qa.scoring data/raw/qasper-dev-v0.3.json predictions.jsonl   # answer-F1 (official) + paragraph-ID evidence-F1
uv run --env-file .env python -m reflective_qa.baselines run sonnet-k20 --max-dollars 1.50   # needs ANTHROPIC_API_KEY in .env
uv run python -m reflective_qa.baselines report                                  # score every run on its sample
uv run pytest
```
