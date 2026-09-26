# Results

20 MultiHop-RAG queries (5 per type, sample seed 0), generator `gemma4:latest`, embeddings `nomic-embed-text`, temperature 0, top-k 8. 95% bootstrap CIs; deltas are paired against `baseline` and bold where the CI excludes zero.

## Answer quality

| technique | EM | EM Δ vs baseline | inference | comparison | temporal | null (refusal) | false refusals | LLM calls/q | median latency |
|---|---|---|---|---|---|---|---|---|---|
| hybrid | 0.650 [0.450, 0.850] | +0.100 [-0.100, +0.300] | 0.60 | 0.60 | 0.40 | 1.00 | 0.27 | 1.0 | 0.4s |
| baseline | 0.550 [0.350, 0.750] | — | 0.20 | 0.60 | 0.40 | 1.00 | 0.40 | 1.0 | 0.6s |

## Retrieval (answerable queries)

| technique | evidence recall | Δ vs baseline | all evidence found | MRR | doc recall | context chars |
|---|---|---|---|---|---|---|
| hybrid | 0.661 [0.478, 0.828] | **+0.156** [+0.033, +0.311] | 0.467 | 0.666 | 0.789 | 6918 |
| baseline | 0.506 [0.333, 0.689] | — | 0.267 | 0.453 | 0.706 | 6773 |

Refusal string: `Insufficient information.`
