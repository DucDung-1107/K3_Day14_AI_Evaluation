# Day 14 — Reflection

## What was fixed

The original golden file contained mojibake and non-verbatim evidence. It was rebuilt from the corpus with 20 records, explicit difficulty/attack strata, expert answers, and verbatim evidence. Validation now passes and covers every document.

## Retrieval findings

The BM25 baseline achieved Recall@1 0.817, Recall@3 0.917, nDCG@1 0.900, and nDCG@3 0.693. After domain query expansion and scope routing, these became 0.867, 0.983, 0.950, and 0.752. M07 and A01 were fixed at rank 1; M02 remains a rank-2 issue and A03 remains a multi-document issue.

## Answer findings — 5 Whys

1. M02 was incomplete because the extractive generator selected the scholarship overview paragraph rather than the renewal paragraph. The retriever returned the right document but ranking/chunk selection was insufficient.
2. M07 was wrong because the first retrieved paragraph described service complaints, not grade appeals. The question needs query expansion for `final grade`, `instructor`, and `clarification`.
3. A01 was unsafe/off-topic because retrieval returned medical-leave policy before the scope document. An out-of-scope classifier or scope-first gate is required.

## Improvement plan

1. Add query-type routing for adversarial/scope requests before BM25.
2. Deduplicate by evidence span and diversify by distinct policy section, not only source document.
3. Add a lightweight query expansion map for grade appeals, scholarship renewal, refunds, and late adds.
4. Use a grounded generator that synthesizes all required conditions instead of returning one paragraph.
5. Add a real LLM judge only after fixing the API credential; calibrate it against the human labels and report TP, TN, FP, FN, accuracy, precision, recall, and F1.

## Limitation

The answer artifact is an offline extractive baseline because the API returned HTTP 429 `insufficient_quota`. Consequently, no LLM-vs-human confusion matrix is claimed. The evaluator preserves this state as unavailable and is ready to rerun after credit is added.
