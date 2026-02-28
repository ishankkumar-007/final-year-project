# Query Construction & Question Answering in COUNTERCASE

## Why Queries Are Template-Based (Not LLM-Generated)

Currently, retrieval queries are built by **template concatenation** in `countercase/retrieval/query_builder.py` — stitching fact sheet fields into fixed sentence patterns. There is no LLM generation involved in query construction.

### Example Output

```
"This case involves a Individual against the State. The relevant legal provisions
are IPC-302, IPC-34. Evidence includes eyewitness testimony, forensic report.
The outcome was: Allowed."
```

### Why Not Use LLM-Generated Queries?

| Concern | Template | LLM-Generated |
|---------|----------|----------------|
| **Latency** | ~0ms | ~2-5s per query (Mistral API call) |
| **Cost** | Free | API tokens per query × every perturbation node |
| **Determinism** | Same fact sheet → same query every time | Non-deterministic, hard to reproduce |
| **Perturbation tree** | A depth-3 tree with 30 nodes = 30 queries instantaneous | 30 × 3s = **90 seconds** of API calls just for query construction |
| **Controllability** | You know exactly what changed between parent/child queries | LLM might rephrase in ways that confuse the sensitivity analysis |

The perturbation tree's sensitivity scoring **depends on** the query changing in exactly one controlled dimension per node. If an LLM rephrases the query differently each time, you can't attribute retrieval differences to the fact you perturbed — it could be the LLM's wording that caused the shift.

### Where LLM Generation IS Used

The project uses LLM generation at the **explanation** stage (Phase 6), not the query stage:

```
Query (template) → Retrieval → Results → LLM explains WHY these results appeared
```

This is deliberate — the retrieval query needs to be **mechanistic and controllable**, while the explanation needs to be **fluent and interpretive**. Different requirements, different tools.

---

## The System Does Not Do Question Answering

COUNTERCASE is a **retrieval and counterfactual analysis** system, not a generative QA system.

```
User input:  a legal case (facts, sections, parties)
     ↓
System output: similar precedent cases + "what if" analysis
```

There is no component that takes a question like *"Can a minor be tried under IPC 302?"* and produces a natural language answer.

### What Happens If a User Pastes a Question

If someone types a question into the Streamlit text area:

```
"What is the punishment for murder under Section 302 IPC?"
```

The system would:

1. Try to extract a fact sheet from it → mostly empty (no parties, no evidence, maybe `IPC-302` detected)
2. Build a near-empty query → `"This case involves a petitioner against the respondent. The relevant legal provisions are IPC-302."`
3. Retrieve chunks that mention IPC 302 → returns judgment paragraphs, not answers
4. Build a perturbation tree on the sparse fact sheet → not meaningful

The user gets back **raw judgment chunks**, not an answer to their question.

### Design Intent

From `plan.md`:

> *"The system retrieves precedent cases and evaluates counterfactual sensitivity of legal facts."*

It is designed for **lawyers who have a case** and want to understand which facts drive which precedents — not for general legal Q&A.

### Future Work: Adding QA via RAG

To add question answering, a **RAG (Retrieval-Augmented Generation)** layer would be needed after retrieval:

```
Question → Retrieve relevant chunks → LLM generates answer grounded in chunks
```

```python
# Hypothetical — NOT part of current system
def answer_question(question: str, retriever: HybridRetriever) -> str:
    chunks = retriever.retrieve(question, top_k=5)
    context = "\n".join(c["text"] for c in chunks)
    prompt = f"Based on the following legal texts, answer: {question}\n\n{context}"
    return call_mistral_api(prompt)
```

This is a different system with different evaluation criteria (answer correctness vs retrieval relevance). Worth noting as future work, not adding to the current scope.
