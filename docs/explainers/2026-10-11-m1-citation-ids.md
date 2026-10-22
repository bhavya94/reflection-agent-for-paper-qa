# M1: cited paragraph IDs without brackets

**PR:** [reflective-qa#10](https://github.com/bhavya94/reflection-agent-for-paper-qa/pull/10) · **Code at:** [`9332d41`](https://github.com/bhavya94/reflection-agent-for-paper-qa/tree/9332d4186739a4d29690a1a50f0d9d26e3d5992b)

## Summary
The 20-question pilot found a harness bug before the full runs. The prompt labels passages as `[s3p0]`, and the
models often copied that format into their citations. Scoring compares IDs exactly, so `[s3p0]` counted as a citation
of a paragraph that doesn't exist. This PR asks for bare IDs in the prompt and strips brackets in the parser.

## In plain words
1. The model cited the right paragraph but wrote it as `[s2p1]`, not `s2p1`.
2. Scoring saw an unknown ID and counted the citation as wrong.
3. Evidence-F1 dropped by half for Sonnet and to almost zero for Haiku. That was a formatting problem, not a model
   problem, so it belongs in the harness, not in M2's list of model failures.

## How it works

```mermaid
flowchart LR
    P["passage label in the prompt: [s2p1]"] --> M[model]
    M -->|"evidence: [\"[s2p1]\"]"| PM["parse_message(): strip brackets and spaces"]
    PM -->|"s2p1"| S["evidence-F1 compares IDs"]
```

- **Prompt** ([`src/reflective_qa/answering.py:20`](https://github.com/bhavya94/reflection-agent-for-paper-qa/blob/9332d4186739a4d29690a1a50f0d9d26e3d5992b/src/reflective_qa/answering.py#L20)): IDs are to be written without brackets.
- **Parser** ([`answering.py:82`](https://github.com/bhavya94/reflection-agent-for-paper-qa/blob/9332d4186739a4d29690a1a50f0d9d26e3d5992b/src/reflective_qa/answering.py#L82)): every cited ID loses brackets and spaces around it. An ID that was never retrieved is still
  kept, so a made-up citation still counts as wrong.

## Pilot evidence (20 questions per run)

| Run | Bracketed citations | Evidence-F1 as scored | With brackets stripped |
|---|---|---|---|
| Sonnet 5.5, 20 paragraphs | 16 of 54 | 0.355 | 0.720 |
| Haiku 5.5, 5 paragraphs | 16 of 20 | 0.019 | 0.435 |

## Decision log

| Decision | Options considered | Chosen | Why | Revisit if |
|---|---|---|---|---|
| Where to fix | Prompt only; parser only; scoring tolerates brackets; prompt and parser | Prompt and parser | The prompt fixes the cause; the parser guarantees clean IDs for every consumer, including M3's code checks. Fixing it in scoring would leave the stored answers dirty | — |
| The pilot's answers | Rewrite them; keep them in their own files | Keep | The prompt change gives each run a new fingerprint, so pilot files stay as a record and the full runs start clean. Re-running 20 questions costs about $0.12 | — |

## Cost & risk
- Pilot cost so far: $0.12. The fixed runs re-answer all 200 questions: about $1.17, priced with the pilot's measured
  output tokens (Sonnet about 65 per answer, Haiku about 138).
- Other pilot findings belong to M2's error analysis, not to this PR. Sonnet's answers are often longer than the
  annotators', which lowers answer-F1. Haiku with 5 paragraphs abstains on 7 of 20 questions.

## Open questions
None for this PR.
