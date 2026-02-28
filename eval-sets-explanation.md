# How Evaluation Sets Are Created

All eval sets in `countercase/evaluation/data/` are created by `countercase/evaluation/create_test_set.py` (unless noted otherwise).

## 1. `citation_test_set.json` — Auto-generated from judgment citations

Built via the `citations` CLI sub-command:

```bash
python -m countercase.evaluation.create_test_set citations --case-texts <json>
```

The function `build_citation_test_set()` (lines 131–188 of `create_test_set.py`):

- Takes a `{case_id: full_judgment_text}` mapping
- Extracts citation strings from each judgment using regex patterns that match Indian law reporters (SCC, SCR, AIR SC, etc.) and case numbers ("Criminal Appeal No. 1234 of 2015")
- Resolves each citation to a corpus `case_id` via `_citation_to_case_id()`
- Cases with ≥ 2 resolved in-corpus citations become test entries, where the cited cases are treated as **relevant ground truth**
- Query text is the facts section (if provided) or first 2000 chars of the judgment

## 2. `merged.json` — Merge of multiple test sets

Built via the `merge` sub-command:

```bash
python -m countercase.evaluation.create_test_set merge file1.json file2.json --output merged.json
```

`merge_test_sets()` (lines 277–291) deduplicates by `query_case_id`, unioning `relevant_case_ids` for duplicates.

## 3. `train_set.json` + `test_set.json` — Year-based split

Built via the `split` sub-command:

```bash
python -m countercase.evaluation.create_test_set split merged.json --cutoff 2020
```

`split_by_year()` (lines 294–311) extracts the year from each `query_case_id` — cases with year ≥ 2020 go to `test_set.json`, the rest to `train_set.json`.

## 4. `cf_eval_set.json` — Counterfactual evaluation set

Created by the counterfactual evaluation module (`countercase/evaluation/counterfactual_eval.py`), not by `create_test_set.py`.

## 5. `explanations.json` — Explanation evaluation data

Created by the explanation evaluation module (`countercase/evaluation/explanation_eval.py`).

## Pipeline Summary

```
Judgment texts ──► extract_citations() ──► citation_test_set.json
                                                    │
                                         merge (with other sets)
                                                    │
                                              merged.json
                                                    │
                                          split_by_year(cutoff=2020)
                                           ┌────────┴────────┐
                                     train_set.json     test_set.json
```

The core idea: **citations within judgments serve as noisy but automatic ground truth** for retrieval relevance — if Case A cites Case B, then B is treated as relevant to A.
