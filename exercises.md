# Day 14 — Full RAG Evaluation

## 1. Golden dataset design

The corpus is synthetic and English; it is the only source of truth. The dataset has 20 unique questions stratified as:

| Case | Count | Construction rule | Example |
|---|---:|---|---|
| Easy | 5 | One fact, one document, direct lookup | E03: tuition is USD 420 per registered credit |
| Medium | 7 | One policy with a condition, exception, or process | M04: late-add approvals, USD 40 fee, two-business-day deadline |
| Hard | 5 | Multi-condition comparison or cross-document reasoning | H03: 100% / 50% / 0% tuition reversal by timing |
| Adversarial | 3 | Out-of-scope, prompt injection, or false premise | A02: request to reveal hidden prompt |

Each record contains `id`, `difficulty`, `question`, `expected_answer`, `contexts`, and `attack_type`. Gold contexts are copied verbatim from the source documents and include `source_doc`; all 10 manifest documents are covered. Answers are written by a human from the evidence, including safety behavior for adversarial cases. The dataset is validated by:

```text
python validate_golden_dataset.py
PASS: 20 records; easy=5, medium=7, hard=5, adversarial=3; document coverage=10/10
```

## 2. Retrieval evaluation

Retrieval is measured before answer quality. A retrieved chunk has gain 2 when it contains a gold evidence string, gain 1 when it is from a gold source but does not contain the evidence, and gain 0 otherwise. Recall@k is the fraction of gold evidence items hit in the first k chunks. nDCG@k uses these graded gains and an ideal ranking for that query.

| Metric | Baseline | After correction |
|---|---:|---:|
| Recall@1 | 0.817 | 0.867 |
| Recall@3 | 0.917 | 0.983 |
| nDCG@1 | 0.900 | 0.950 |
| nDCG@3 | 0.693 | 0.752 |

Retrieval fail cases before correction: 18/20. After query expansion and scope/safety routing: 17/20 under the strict ideal-ranking threshold; hard evidence misses at top-3 are reduced to A03. M02 still has the exact renewal paragraph at rank 2, while M07 and A01 are fixed at rank 1. The root pattern is noisy duplicate chunks and semantic competition between adjacent policies.

## 3. Answer evaluation and LLM judge

The configured API key authenticated but returned HTTP 429 `insufficient_quota`. The offline extractive fallback is explicitly recorded in `artifacts/actual_answers.json` and does not read expected answers. Baseline answer-side result: 2/20 pass, faithfulness 0.363, relevance 0.542, completeness 0.803. After retrieval correction: 2/20 pass, faithfulness 0.382, relevance 0.546, completeness 0.820.

The intended judge rubric is 1–5: 5 fully correct and complete; 4 correct with minor omission; 3 partially correct; 2 materially wrong/incomplete; 1 unsafe, irrelevant, or unsupported. PASS is score ≥4 and safe. Human labels are in `artifacts/human_labels.json`.

The exact LLM-vs-human confusion matrix is **not available**: the judge call failed with HTTP 429 `insufficient_quota` before scoring any case. `artifacts/judge_results.json` stores `status: unavailable` and the error rather than fabricating a matrix. Re-run `python full_rag_evaluation.py` after adding credit to populate the real matrix.

## 4. Reproducibility

```text
python domain_assistant.py --corpus-dir data/student_services --dataset golden_dataset.json --output artifacts/actual_answers.json --top-k 5
python full_rag_evaluation.py
python evaluate_answers.py --output artifacts/benchmark_results.json
pytest -q
```

The generated artifacts are `actual_answers.json`, `retrieval_metrics.json`, `benchmark_results.json`, `judge_results.json`, and `full_eval_console.txt`.
