# M0 demo: QASPER download and loader

Run on a fresh clone of commit `9dd5707`, against the real Allen AI S3 release. Full output:
[`transcript.txt`](transcript.txt).

## Reproduce

```bash
git clone <repo> && cd reflective-qa
uv sync
uv run python -m reflective_qa.download
uv run python -m reflective_qa.stats data/raw/qasper-dev-v0.3.json
uv run pytest -v
```

## Results

| # | Behavior | Expected | Observed |
|---|---|---|---|
| 1 | Download and verify QASPER v0.3 | train, dev, evaluator and READMEs in `data/raw` | All 5 files present, exit 0 |
| 2 | Test split never extracted | no `*test*.json` under `data/` | `find` returns nothing |
| 3 | Re-running is idempotent | returns without downloading when files exist (`download.py:20-21` checks before any network call) | exit 0, same files |
| 4 | **Failure case:** checksum mismatch | `ValueError`, nothing written | `Checksum mismatch … got a28fdf96…`; `tampered/raw` has no files |
| 5 | Dev split summary | matches the explainer's table | 281 papers · 1,005 questions · 841 text-only · 2,154 / 124 evidence |
| 6 | Tests, including evaluator parity on real dev data | all pass, parity test not skipped | 9 passed |

Step 4 uses a copy of `configs/data.toml` with the train/dev checksum replaced by zeros. The real
archive was downloaded and rejected, which proves the check runs on the downloaded bytes.
