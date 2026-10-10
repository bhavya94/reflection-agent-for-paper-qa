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
| M1 | BM25 retrieval + evidence recall@k; one-shot and closed-book baselines; answer-F1 by type | in progress |
| M2 | Error analysis: read ~100 outputs, name failure modes, label Pass/Fail | |
| M3 | Critic (code checks + one binary LLM check per failure mode) and revise loop | |
| M4 | Experiment: one-shot vs. self-critique vs. code feedback vs. separate critic | |

## Run

```bash
uv sync
uv run python -m reflective_qa.download                                # QASPER train/dev + official evaluator, checksum-verified
uv run python -m reflective_qa.stats data/raw/qasper-dev-v0.3.json     # dev split summary
uv run python -m reflective_qa.retrieval_eval data/raw/qasper-dev-v0.3.json   # BM25 evidence recall@k
uv run python -m reflective_qa.scoring data/raw/qasper-dev-v0.3.json predictions.jsonl   # answer-F1 (official) + paragraph-ID evidence-F1
uv run pytest
```
