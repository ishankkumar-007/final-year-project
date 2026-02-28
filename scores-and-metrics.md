# COUNTERCASE -- Scores & Metrics Reference

A complete reference of every score and metric used in the system, organized by where they appear in the pipeline.

---

## 1. Retrieval Pipeline Scores

These scores are computed during the six-stage retrieval pipeline (`HybridRetriever.retrieve()`).

### 1.1 RRF Score (Reciprocal Rank Fusion)

| Property | Value |
|----------|-------|
| **Range** | 0 to ~0.05 (theoretical max ≈ 1/k per list) |
| **Higher is** | Better |
| **Computed in** | `countercase/retrieval/rrf.py` |
| **Current k** | 20 (`RRF_K` in `settings.py`) |

**What it is:** Combines rankings from two independent retrieval systems (DPR dense vectors + ChromaDB sparse/semantic vectors) into a single fused ranking using:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

**Why it matters:** Neither DPR nor ChromaDB alone captures all relevant results. RRF merges both lists without needing to normalize their raw scores (which live on different scales). It's the primary score that determines which chunks survive to later stages.

**What the values mean:**
- `0.04–0.05` → document appeared near the top of both retrieval lists (strong match)
- `0.02–0.03` → document appeared in one list or lower in both
- `< 0.01` → document barely made the cut

---

### 1.2 MMR Score (Maximal Marginal Relevance)

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better (relevance–diversity trade-off) |
| **Computed in** | `countercase/retrieval/mmr.py` |
| **Lambda** | 0.7 (controls relevance vs diversity balance) |

**What it is:** Re-ranks the RRF results to reduce redundancy. Each document's MMR score balances two objectives:

$$\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{Sim}(d, d_j)$$

Where $\lambda = 0.7$ means 70% relevance, 30% diversity.

**Why it matters:** Without MMR, a search for "IPC 302 murder" might return 10 chunks from the same case. MMR ensures diversity — you get chunks from different cases covering different angles.

**What the values mean:**
- `0.8–1.0` → highly relevant and not redundant with already-selected results
- `0.4–0.6` → moderately relevant or somewhat similar to existing selections
- `< 0.3` → low relevance or too similar to already-selected results

---

### 1.3 Cross-Encoder Reranker Score

| Property | Value |
|----------|-------|
| **Range** | -∞ to +∞ (typically -10 to +10) |
| **Higher is** | More relevant |
| **Computed in** | `countercase/retrieval/reranker.py` |
| **Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

**What it is:** A cross-encoder model that sees both the query and document text together (unlike DPR which encodes them separately). It produces a raw logit score indicating semantic match quality.

**Why it matters:** Cross-encoders are more accurate than bi-encoders (DPR) because they can attend across query and document tokens simultaneously. They're too slow for initial retrieval (must score every document), but perfect for re-ranking a small candidate set (10–50 documents).

**What the values mean:**
- `> 0` → model considers the document relevant to the query
- `-2 to 0` → weak relevance
- `< -5` → likely not relevant

---

## 2. Sensitivity & Counterfactual Scores

These scores measure how much changing a fact affects retrieval results.

### 2.1 Mean Rank Displacement

| Property | Value |
|----------|-------|
| **Range** | 0.0 to K+1 (where K = top_k, default 10) |
| **Higher is** | More sensitive (fact change had bigger impact) |
| **Computed in** | `countercase/counterfactual/sensitivity.py` |

**What it is:** When a fact is perturbed (e.g., changing "IPC 302" to "IPC 304"), the system re-runs retrieval and compares the top-K results. For each case:

$$\text{displacement}(c) = |\text{rank}_{\text{parent}}(c) - \text{rank}_{\text{child}}(c)|$$

If a case appears in only one result set, its rank in the other is treated as K+1 (i.e., it "fell off" the list). The mean displacement averages across all affected cases.

**Why it matters:** This is the core metric of counterfactual analysis. High displacement = that fact is legally dispositive (changing it changes which precedents apply). Low displacement = that fact doesn't materially affect retrieval.

**What the values mean:**
- `> 3.0` → substantial impact — changing this fact significantly reshuffles precedent results
- `1.5–3.0` → moderate impact — some results shift
- `< 1.0` → minimal impact — retrieval is robust to changes in this fact
- `0.0` → no impact at all

---

### 2.2 Aggregate Sensitivity Score (per fact type)

| Property | Value |
|----------|-------|
| **Range** | 0.0 to K+1 |
| **Higher is** | More sensitive |
| **Computed in** | `countercase/counterfactual/sensitivity.py` (`compute_sensitivity_scores()`) |

**What it is:** Averages mean rank displacement across all perturbation edges of the same fact type (Section, PartyType, Evidence, Numerical) in the perturbation tree.

**Why it matters:** Answers the question *"which category of facts matters most for this case?"* For a murder case, you'd expect Section (IPC 302) and Evidence to have high sensitivity, while Numerical (exact compensation amount) might have low sensitivity.

**Canonical fact types:**
- **Section** — statutes/provisions cited (e.g., IPC 302, Article 21)
- **PartyType** — petitioner/respondent type (Individual, State, Corporation)
- **Evidence** — types of evidence (FIR, medical report, eyewitness)
- **Numerical** — amounts, ages, durations

---

### 2.3 Displacement Threshold

| Property | Value |
|----------|-------|
| **Default** | 1.5 (for evaluation), 1.0 (for tree expansion pruning) |
| **Purpose** | Separates "high sensitivity" from "low sensitivity" |

**What it is:** A cutoff used in two contexts:
1. **Tree expansion** (`min_displacement_threshold`): nodes with mean displacement below threshold are not expanded further (pruning uninteresting branches)
2. **Counterfactual evaluation**: facts with sensitivity > threshold are classified as "dispositive" (legally decisive)

---

## 3. Retrieval Evaluation Metrics

These are standard IR metrics computed by the evaluation harness against ground-truth test sets.

### 3.1 MRR@K (Mean Reciprocal Rank)

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Computed in** | `countercase/evaluation/metrics.py` |
| **Typical K** | 10 |

**What it is:** The reciprocal of the rank position of the first relevant result:

$$\text{MRR@K} = \frac{1}{\text{rank of first relevant result}}$$

If no relevant result appears in the top K, MRR = 0.

**Why it matters:** Measures how quickly a user finds what they need. In legal research, finding the right precedent in position 1 vs position 5 matters enormously.

**What the values mean:**
- `1.0` → first result is relevant (perfect)
- `0.5` → second result is the first relevant one
- `0.1` → first relevant result is at position 10
- `0.0` → no relevant result in top K

---

### 3.2 NDCG@K (Normalized Discounted Cumulative Gain)

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Computed in** | `countercase/evaluation/metrics.py` |
| **Relevance model** | Binary (relevant = 1, not = 0) |

**What it is:** Measures ranking quality considering all relevant results and their positions. Results at higher ranks contribute more:

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i + 1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where IDCG is the ideal DCG (all relevant docs at the top).

**Why it matters:** Unlike MRR (which only cares about the first relevant result), NDCG rewards putting *all* relevant results near the top. Crucial when a lawyer needs to see multiple precedents, not just one.

**What the values mean:**
- `1.0` → all relevant results are ranked at the very top (ideal ordering)
- `0.7–0.9` → good ranking with most relevant results near the top
- `0.3–0.6` → relevant results are scattered throughout the ranking
- `< 0.2` → poor ranking

---

### 3.3 Recall@K

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Computed in** | `countercase/evaluation/metrics.py` |

**What it is:** Fraction of all relevant documents that appear anywhere in the top K:

$$\text{Recall@K} = \frac{|\text{relevant} \cap \text{top-K}|}{|\text{relevant}|}$$

**Why it matters:** Answers *"did we find everything?"* High recall means the system doesn't miss important precedents. A system with Recall@10 = 0.3 is missing 70% of relevant cases.

**What the values mean:**
- `1.0` → all relevant cases found in top K
- `0.5` → half the relevant cases found
- `0.0` → none of the relevant cases found

---

## 4. Counterfactual Evaluation Metrics

These evaluate whether the perturbation system correctly identifies which facts matter.

### 4.1 Dispositive Accuracy

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Computed in** | `countercase/evaluation/counterfactual_eval.py` |

**What it is:** Fraction of expert-annotated dispositive facts (facts that should matter) that the system correctly assigns high sensitivity (above threshold).

**Why it matters:** If experts say "IPC section matters for this case" and the system agrees, that's a correct identification. Low accuracy means the system isn't detecting legally important facts.

---

### 4.2 Non-Dispositive Accuracy

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Computed in** | `countercase/evaluation/counterfactual_eval.py` |

**What it is:** Fraction of non-dispositive facts (facts that shouldn't matter) that the system correctly assigns low sensitivity (below threshold).

**Why it matters:** If the system says "accused's exact age is critical" for a murder case, that's likely a false alarm. High non-dispositive accuracy means the system isn't crying wolf.

---

### 4.3 Spearman's Rank Correlation (ρ)

| Property | Value |
|----------|-------|
| **Range** | -1.0 to +1.0 |
| **+1.0 means** | Perfect agreement |
| **Computed in** | `countercase/evaluation/counterfactual_eval.py` |

**What it is:** Measures how well the system's sensitivity ranking agrees with expert-provided importance rankings across fact dimensions:

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

Where $d_i$ is the difference between system rank and expert rank for fact type $i$.

**Why it matters:** This captures whether the *ordering* is right, not just individual classifications. Even if absolute scores differ, if the system ranks Section > Evidence > PartyType > Numerical in the same order as experts, ρ will be high.

**What the values mean:**
- `+0.8 to +1.0` → strong agreement with expert judgment
- `+0.4 to +0.7` → moderate agreement
- `0.0` → no correlation (random ordering)
- `< 0` → system gets the ordering backwards

---

### 4.4 LLM Accept Rate

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Computed in** | `countercase/evaluation/counterfactual_eval.py` |

**What it is:** Fraction of proposed perturbations that the LLM validator accepted as legally plausible (not self-contradictory or impossible).

**Why it matters:** Low accept rate means the perturbation generator is producing nonsensical "what-if" scenarios. High accept rate means perturbations are realistic.

---

## 5. Explanation Evaluation Metrics

### 5.1 Faithfulness Score

| Property | Value |
|----------|-------|
| **Range** | 0.0 to 1.0 |
| **Higher is** | Better |
| **Target** | > 0.90 |
| **Computed in** | `countercase/evaluation/explanation_eval.py` |

**What it is:** Fraction of sentences in an explanation that are "grounded" in the source text. A sentence is grounded if ≥ 40% of its content words appear in at least one source chunk.

**Why it matters:** Explanations must be traceable to actual case text — they shouldn't hallucinate facts. A faithfulness of 0.95 means 95% of explanation sentences have evidence in the retrieved chunks.

**What the values mean:**
- `> 0.90` → highly faithful, almost every claim is grounded
- `0.70–0.90` → mostly faithful but some unsupported statements
- `< 0.70` → concerning — many claims lack source evidence

---

### 5.2 Human Evaluation Likert Scores

These are collected from legal expert annotators via CSV templates.

#### Clarity (1–5)

| Score | Meaning |
|-------|---------|
| 5 | Perfectly clear, easy to understand |
| 4 | Clear with minor ambiguity |
| 3 | Understandable but could be clearer |
| 2 | Confusing in parts |
| 1 | Incomprehensible |

#### Legal Accuracy (1–5)

| Score | Meaning |
|-------|---------|
| 5 | Legally precise and correct |
| 4 | Mostly accurate, minor issues |
| 3 | Partially accurate |
| 2 | Contains significant legal errors |
| 1 | Legally incorrect |

#### Usefulness (1–5)

| Score | Meaning |
|-------|---------|
| 5 | Highly useful for legal research |
| 4 | Useful with minor improvements needed |
| 3 | Somewhat useful |
| 2 | Marginally useful |
| 1 | Not useful at all |

---

## Summary Table

| Score | Range | Used In | Measures |
|-------|-------|---------|----------|
| RRF Score | 0 – ~0.05 | Retrieval Stage 3 | Fused relevance from DPR + ChromaDB |
| MMR Score | 0 – 1.0 | Retrieval Stage 4 | Relevance balanced with diversity |
| Reranker Score | -∞ to +∞ | Retrieval Stage 5 | Cross-encoder semantic match |
| Mean Displacement | 0 – K+1 | Sensitivity analysis | Impact of changing one fact |
| Sensitivity Score | 0 – K+1 | Perturbation tree | Aggregate impact per fact type |
| MRR@K | 0 – 1.0 | Eval harness | Rank of first relevant result |
| NDCG@K | 0 – 1.0 | Eval harness | Quality of full ranking |
| Recall@K | 0 – 1.0 | Eval harness | Coverage of relevant results |
| Dispositive Accuracy | 0 – 1.0 | CF evaluation | Correctly identified important facts |
| Non-Dispositive Accuracy | 0 – 1.0 | CF evaluation | Correctly identified unimportant facts |
| Spearman ρ | -1 to +1 | CF evaluation | Ranking agreement with experts |
| LLM Accept Rate | 0 – 1.0 | CF evaluation | Plausibility of generated perturbations |
| Faithfulness | 0 – 1.0 | Explanation eval | Grounding in source text |
| Clarity | 1 – 5 | Human eval | Readability of explanations |
| Legal Accuracy | 1 – 5 | Human eval | Correctness of legal claims |
| Usefulness | 1 – 5 | Human eval | Practical value for lawyers |
