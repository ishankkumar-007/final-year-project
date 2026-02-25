Project Description
Overview
The Indian judicial system produces an enormous volume of unstructured case data, making it challenging for legal professionals to retrieve relevant precedents efficiently. Existing legal retrieval systems primarily focus on factual similarity, overlooking the deeper legal issues and lacking mechanisms to explain or test the relevance of retrieved precedents.
COUNTERCASE aims to bridge this gap by integrating legal case retrieval with counterfactual reasoning—enabling “what-if” analysis of case facts to assess how legal outcomes might change. The project will utilize Indian legal datasets and employ transformer-based models (LegalBERT, InLegalBERT) for semantic retrieval, combined with explainable AI techniques for interpretability.
________________________________________
Proposed System
The proposed system combines three core modules:
1.	Legal Case Retrieval Module:
Uses hybrid search (BM25 + dense embeddings) to retrieve relevant precedents based on both factual and legal issue similarity.
2.	Counterfactual Reasoning Module:
Simulates hypothetical scenarios by altering key facts in a case to observe how retrieved precedents or outcomes change—capturing the sensitivity of legal reasoning to factual variations.
3.	Explanation Engine:
Generates human-understandable explanations linking query facts to retrieved precedents, and clearly outlines how counterfactual changes affect relevance or predicted decisions.
This unified architecture provides transparent, context-aware retrieval and a means to test legal reasoning robustness under alternative scenarios.
________________________________________
Workflow
1.	Input: A legal case (query) or its segmented facts and issues.
2.	Retrieval: Retrieve candidate precedents using hybrid lexical and semantic search.
3.	Re-ranking: Rank retrieved cases using a fine-tuned transformer cross-encoder.
4.	Counterfactual Generation: Identify and modify key factual variables to generate “what-if” variants.
5.	Analysis: Compare retrieved results for each variant to determine which facts influence precedent applicability.
6.	Explanation: Present a detailed, text-based justification of relevance and counterfactual impact for user interpretation.
________________________________________
Objectives
•	To develop a retrieval system capable of identifying legally relevant precedents beyond mere factual overlap.
•	To introduce a counterfactual reasoning framework that tests the causal influence of key facts on case retrieval outcomes.
•	To improve explainability in legal AI systems through transparent, text-grounded justifications of retrieved cases.




ideas:
1. Maximal Marginal Relevance balances relevance with diversity. Each retrieved chunk has to earn its place by adding new information, not just scoring high.
2. also provide the source/pdf for the block/chunk that is retrieved
3. a. chunking strategy: 
    - for legal dataset keep long chunks, as there’s more context
    -  size, overlap and stride
    -  all-MiniLM-L6-v2
    -  storage - ChromaDB
    b. embedding strategy
    c. retrieval strategy
4. rag paired with fine tuning?
4. The no.1 mistake people make while building RAGS is Optimizing embeddings while ignoring metadata filtering.

    Vector search answers *how similar* documents are.

    Metadata decides which documents are even allowed to compete.

    If you don’t pre-filter, your system:

    - Searches 2019 docs for 2025 questions ❌
    - Wastes 75% of the context window ❌
    - Adds +30ms latency for worse recall ❌

    🔍 Production smell test

    Query: *“What were our Q4 2025 cloud infrastructure costs?”*

    If results include 2023 docs → your RAG is broken.

    ✅ Production-grade fix: Staged Hybrid Filtering

    Stage 1 — Pre-filter (indexed, selective)

    - date_range, department, access level
    - 1M docs → 100K
    - +6ms latency, 95% recall

    Stage 2 — ANN vector search

    - Semantic ranking on the filtered subset (HNSW / IVF)

    Stage 3 — Post-filter (non-indexed)

    - author_verified, tags, word_count
    - Lightweight refinement only

    📊 Real benchmarks (1M docs)

    - No filter: 12ms, wrong category
    - Post-filter: 45ms, 87% recall
    - Pre-filter: 18ms, 95% recall ✅

    💼 Interview-level insight

    “Pre-filter when selectivity <10%.

    Post-filter only when >50%.

    10–50% → staged hybrid.”

    📌 Takeaway

    Vector search finds meaning.

    cMetadata enforces reality.

    Production RAG needs both.


to use/notes:
1. LangChain RecursiveCharacterTextSplitter for chunking, all-MiniLM-L6-v2 for embeddings, ChromaDB as vector store, 
2. since the files are pdf, so need to get text from them (note: the text in selectable if i open it in a pdf viewer)
3. do not use emojis in the codebase

task: 
You are professional industry level architect for RAG pipelines. Plan the project. highlight novelty and what has not been done in the domain.
write the plan to plan.md file
