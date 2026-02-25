# COUNTERCASE: Legal Case Retrieval with Counterfactual Reasoning
## Technical System Plan

---

## 1. Executive Summary

COUNTERCASE is a hybrid retrieval-augmented generation system for Indian Supreme Court case law that addresses a critical gap in legal AI: existing systems retrieve precedents based on semantic or lexical similarity but provide no mechanism to understand which facts in a query case causally determine precedent applicability. COUNTERCASE introduces a perturbation tree framework that systematically alters key facts in a case and re-runs retrieval at each altered state, exposing which precedents are sensitive to specific factual changes and which remain stable. This enables lawyers and researchers to answer questions that standard RAG cannot address, such as "would this precedent still apply if the petitioner were a minor instead of an adult?" or "how does shifting the compensation amount above the statutory threshold change which cases are relevant?" The system operates over the AWS Data Exchange Indian Supreme Court Judgments corpus and combines structured fact sheet extraction, hybrid BM25 and dense retrieval with metadata pre-filtering, NER-rule-LLM based fact perturbation, and rank displacement sensitivity analysis to deliver a tool that is both practically useful for legal research and a novel contribution to Indian legal NLP.

---

## 2. Novelty and Prior Work Gap

The Indian legal NLP landscape has made significant progress in judgment outcome prediction and named entity recognition but has not addressed the problem of retrieval sensitivity analysis. OpenNyAI and the ILDC corpus have enabled research on predicting whether a case will be upheld or overturned based on text features, and the FIRE and SemEval legal information retrieval tracks have established benchmarks for finding relevant statutes or prior cases given a query. However, these systems treat retrieval as a one-shot operation: given a query case, return the top K most similar precedents. They cannot answer counterfactual questions about how the retrieval results would change if a specific fact were different.

Standard RAG pipelines over legal corpora suffer from the same limitation. A dense retrieval model trained on semantic similarity will rank cases that share factual patterns highly, but it provides no insight into which of those shared facts are legally operative. If two cases both involve Section 302 IPC and both involve minors, a similarity-based retriever will rank them as relevant to each other, but it cannot tell the user whether the relevance is driven by the section citation, the age of the party, or some other latent factor in the embedding space. This is a serious deficiency because legal reasoning is explicitly causal: a precedent applies because specific facts match specific legal tests, and changing one of those facts can make the precedent inapplicable.

COUNTERCASE fills this gap with the perturbation tree framework. The tree is a directed graph where each node represents a fact sheet state and each edge represents a single fact alteration. At each node, the full hybrid retrieval pipeline is executed and the result set is stored. The user can traverse the tree to see which precedents drop out when a fact changes and which new precedents appear, with explicit rank displacement scores quantifying sensitivity. The tree is constructed through a three-stage perturbation pipeline that combines NER-based span tagging, rule-based perturbation with legal domain knowledge, and LLM-based plausibility validation. This architecture has not been implemented in the Indian legal NLP domain. The closest related work is counterfactual outcome prediction in sentiment analysis or causal inference over observational legal data, but those systems do not operate on the retrieval layer and do not produce structured perturbation trees that the user can navigate interactively.

The architectural contribution is the combination of auto-parsed structured fact sheets as the unit of counterfactual manipulation, hybrid retrieval with metadata-aware pre-filtering as the evaluation function at each tree node, and rank-diff sensitivity scoring as the method for quantifying which facts drive precedent applicability. Each of these components exists in some form in prior work, but their integration into a perturbation tree over legal retrieval results is novel. Additionally, the design decision to operate entirely on the retrieval layer without requiring a separate annotated counterfactual dataset is a practical advantage that enables deployment on the full Supreme Court corpus without manual fact annotation.

---

## 3. Dataset

The corpus is the AWS Data Exchange Indian Supreme Court Judgments dataset, a collection of judgment PDFs spanning 1950 to the present. The dataset is accessible via the AWS Marketplace after subscription, with programmatic access through the ADX API or direct S3 bucket sync. The judgments vary widely in length: short bail orders may be two pages, while constitutional bench decisions on fundamental rights can exceed one hundred pages. Formatting is inconsistent across decades. Judgments from the 1950s through the 1980s often contain scanning artifacts despite being digitized from printed volumes, including inconsistent line spacing, broken paragraph boundaries, and mangled citation strings. Post-2000 judgments are generally cleaner because they were born digital, but even modern judgments lack a standardized section heading schema. Some use "Facts and Background" while others use "Statement of the Case" or no explicit heading at all.

Metadata natively provided with the dataset is minimal and varies by year. Pre-1990 entries may have only a case number and year. Post-2000 entries are more likely to include bench composition and outcome labels, but these are not guaranteed. The metadata fields that must be extracted from the judgment text itself include statutes and act sections cited, party names and roles, judge names and bench size, outcome label, and the nature of the petition. Citation strings within judgments are inconsistent: some use a standardized law reporter format while others use informal references or omit the year entirely.

Known quality issues include OCR remnants in scanned judgments manifesting as character substitution errors, broken table formatting where numerical data was presented in columnar layouts, and inconsistent page numbering. Some older judgments restart page numbering mid-document or omit page numbers altogether. Headers and footers from the original printed volumes are sometimes retained in the PDF, causing repeated boilerplate noise in the extracted text. Cause-list metadata occasionally appears at the start of a judgment, listing unrelated cases heard on the same day, which must be filtered out during preprocessing.

This is the single dataset used for both RAG indexing and counterfactual reasoning. No separate annotated counterfactual dataset is required because the counterfactual module operates on the retrieval layer: it re-queries the same index with perturbed fact representations and compares result sets. This architectural decision is justified by practicality and novelty. Creating a counterfactual-annotated legal corpus would require human annotators to write alternate fact versions of cases and label which precedents would apply under each version. This is prohibitively expensive and does not scale to the full Supreme Court corpus. By operating on retrieval directly, COUNTERCASE requires zero counterfactual-specific annotation. The system learns nothing about counterfactuals during indexing; it evaluates them at query time by treating each perturbed fact sheet as a new retrieval query. This means the system generalizes to any case in the corpus and to user-provided cases that are not in the corpus, which is a significant practical advantage and a point of novelty in the design.

---

## 4. Data and Preprocessing Pipeline

PDF ingestion is handled with pdfplumber because it preserves text layout information including bounding boxes and table structures, which PyMuPDF does not expose as cleanly. The ingestion pipeline must handle multi-column layouts common in older Supreme Court judgments, where the left and right columns must be merged in reading order rather than extracted sequentially. Headers and footers are detected by position heuristics: any text block that appears in the top 5 percent or bottom 5 percent of the page across multiple consecutive pages is classified as boilerplate and removed. Cause-list noise is detected by matching against a regular expression for "Diary No." or "Before:" followed by a list of judge names at the document start, and all text before the first case title is discarded.

Page numbering inconsistencies are resolved by using the PDF's internal page index rather than any printed page number in the text. Each extracted text chunk is tagged with the zero-indexed page number from the PDF object, which is then incremented by one when presented to the user. This ensures that source attribution is stable even when the judgment itself has non-sequential page numbers.

The metadata extraction schema is defined as follows: case_id is a string combining the case number and year, such as "Criminal Appeal 1234/2015". Court is always "Supreme Court of India" for this corpus. Bench_type is extracted by counting judge names in the judgment header and classifying as "Single", "Division", "Three-Judge", "Constitution Bench (5+)", or "Unknown" if parsing fails. Year and month are extracted from the judgment date, which is typically found in the first few lines. Act_sections is a list of strings extracted by a regular expression matching patterns like "Section 302 IPC", "Section 498A", "Article 21 of the Constitution", with normalization to handle variations like "Sec.", "S.", "Art.". Citation_string is the formal reporter citation if present, otherwise null. Judge_names is a list extracted from the bench composition in the header. Outcome_label is extracted by searching for keywords like "appeal allowed", "appeal dismissed", "petition disposed of", in the final paragraphs and classifying as "Allowed", "Dismissed", "Disposed", or "Unknown". Pdf_path is the S3 key or local file path. Page_range is the start and end page indices for the judgment in the PDF, which is trivial for single-judgment PDFs but necessary when a volume PDF contains multiple judgments.

Structured fact sheet extraction operates on the "Facts" or "Background" section of each judgment. The section is located using a two-stage heuristic. First, the text is segmented by heading-like lines: lines that are all-caps, bold, or followed by a paragraph break. Common heading patterns like "FACTS", "FACTUAL BACKGROUND", "STATEMENT OF THE CASE", "SUBMISSIONS", "HELD" are matched. The section between the start of the document or the first heading and the heading that matches "ISSUE" or "SUBMISSIONS" is taken as the facts section. If no heading is found, the first 20 percent of the judgment text is taken as a fallback because in older judgments the facts are stated without a formal heading.

The fact sheet schema is structured JSON with the following fields: parties contains petitioner_type and respondent_type, where the type is a closed-vocabulary label like "Individual", "Corporation", "State", "UnionOfIndia", "Minor", extracted by matching party names against known entity patterns and role keywords in the facts section. Evidence_items is a list of dictionaries, each containing an evidence_type (such as "DyingDeclaration", "Confession", "MedicalReport", "Document") and a brief descriptive string extracted from the facts section. Sections_cited is the same list as in the metadata schema but filtered to only those sections explicitly mentioned in the facts section rather than the entire judgment. Numerical_facts is a dictionary with keys for amounts (monetary values with units), ages (integers and descriptors like "minor"), and durations (time periods in years, months, days). Outcome is copied from the outcome_label in the metadata.

This fact sheet is populated by passing the extracted facts section text to an LLM with a structured prompt. The prompt is: "You are a legal analyst. Extract a structured fact sheet from the following case facts. Return valid JSON with the following schema: {schema}. Only include information explicitly stated in the text. If a field is not present, use null or an empty list." The LLM used is a quantized Mistral-7B model run locally if a GPU is available, or an API call to a hosted model if not. The output is parsed as JSON and validated against the schema. If parsing fails, the pipeline retries once with an example-augmented prompt, and if it fails again, the case is flagged for manual review and the fact sheet is left null. This approach is deterministic up to the LLM call and is preferred over pure rule-based extraction because the variety of fact presentation formats across seven decades of judgments makes exhaustive rule writing infeasible.

Chunking is performed with LangChain RecursiveCharacterTextSplitter configured with a chunk size of 1024 tokens and an overlap of 128 tokens. The separators list is customized for legal text: split first on double newlines (paragraph boundaries), then on section-number patterns like "1.", "2.", "(a)", "(i)", and finally fall back to character boundaries if a single paragraph exceeds the chunk size. Token counting is performed with the tiktoken library using the cl100k_base encoding, which is close enough to the embedding model's tokenizer for planning purposes. Long chunks are justified because legal text is dense with context: pronouns, defined terms, and compound sentence structures span many clauses, and a chunk of fewer than 400 tokens often cuts off mid-argument. The 128-token overlap ensures that a sentence split across two chunks is fully present in at least one chunk.

Section type tagging is applied to each chunk based on which structural section of the judgment it came from. The judgment is pre-segmented into sections by matching headings like "FACTS", "ISSUES", "SUBMISSIONS", "ANALYSIS", "HELD", "RATIO DECIDENDI", "OBITER DICTA". Each chunk inherits the section type of the segment it belongs to. Chunks that span a section boundary are tagged with the section type of the majority of their tokens. Section-type metadata is stored per chunk in ChromaDB and can be used as a filter at retrieval time if the user wants to restrict retrieval to only the "Held" or "Ratio" sections of precedents.

---

## 5. Indexing Architecture

The dual-index design consists of a BM25 sparse lexical index and a ChromaDB dense vector index. These are separate data stores queried in parallel at retrieval time. BM25 is implemented using the rank_bm25 Python library over the chunk text. The BM25 index is a in-memory inverted index mapping terms to chunk IDs with term frequencies and document frequencies pre-computed. For production deployment at scale, this can be swapped for Elasticsearch with BM25 scoring, but rank_bm25 is sufficient for a corpus of under 100,000 chunks, which is the expected size for Supreme Court judgments over seven decades given the chunk size. ChromaDB is configured with the all-MiniLM-L6-v2 embedding model from sentence-transformers. This model produces 384-dimensional embeddings and is CPU-compatible, making it suitable for the compute-flexible requirement. Embeddings are computed in batches of 32 during indexing and stored in ChromaDB with HNSW indexing for approximate nearest neighbor search.

The metadata schema in ChromaDB includes the following filterable fields: year is an integer extracted from the case date. Court is always "Supreme Court of India" for this corpus but is included for future extensibility. Bench_type is a string enum with values "Single", "Division", "Three-Judge", "Constitution", "Unknown". Act_sections is a list of strings, where each string is a normalized section reference like "IPC-302" or "Constitution-Article-21". ChromaDB supports filtering on list fields with a contains operator. Section_type is a string enum tagging the structural origin of the chunk: "Facts", "Issues", "Submissions", "Held", "Ratio", "Obiter", "Unknown". Outcome_label is a string enum: "Allowed", "Dismissed", "Disposed", "Unknown". Chunk_id is a unique identifier composed of the case_id and a zero-padded chunk index. Source_pdf is the S3 key or file path. Page_number is the integer page index from the PDF.

Staged hybrid filtering is critical to retrieval performance. The architecture distinguishes three stages. Stage 1 is pre-filtering at index time on high-selectivity metadata fields. When a user query specifies a year range, such as 2010 to 2015, this eliminates approximately 90 percent of the corpus before any vector search is performed. The selectivity rule is: apply pre-filtering when the filter would exclude more than 50 percent of the corpus. Year_range is always applied as a pre-filter because a five-year window over a seventy-year corpus has 93 percent selectivity. Act_sections is applied as a pre-filter when the user specifies a particular act section, because queries for "IPC 302" cases will typically match fewer than 5 percent of the corpus. Bench_type is applied as a pre-filter when the user restricts the query to Constitution Bench decisions, which are rare, but not when querying for Division Bench decisions, which are the majority. The pre-filter is implemented in ChromaDB by constructing a where clause with the filter conditions before executing the ANN search.

Stage 2 is ANN vector search over the filtered subset. ChromaDB performs HNSW traversal on the subset of documents that passed the pre-filter, returning the top K candidates by cosine similarity to the query embedding. The parameter K is set to 50 in the hybrid retrieval configuration, meaning the dense retrieval stage returns 50 candidates.

Stage 3 is optional post-filtering on non-indexed fields or low-selectivity fields. For example, if the user wants to restrict results to judgments authored by a specific judge but that judge has written 30 percent of the corpus, pre-filtering would be inefficient. Instead, the top K results from Stage 2 are post-filtered by iterating over the retrieved chunks and discarding those that do not match the judge name in their metadata. Post-filtering is also used for derived fields like word_count if the user requests only short judgments, because word_count is not indexed and must be computed or retrieved from metadata on the fly.

The selectivity justification is as follows. Pre-filtering on year_range from 2010 to 2015 over a corpus spanning 1950 to 2025 reduces the candidate set from 75 years to 5 years, or approximately 93 percent reduction. This reduction occurs before any embedding computation or vector search, which is computationally expensive. The cost of traversing the HNSW graph is proportional to the number of indexed vectors, so reducing the candidate set by 93 percent before HNSW traversal is a near-order-of-magnitude speedup. Conversely, post-filtering on year_range would require retrieving the top 50 chunks from the entire corpus and then discarding 93 percent of them, which wastes compute and degrades precision because only 3 to 4 of the top 50 would pass the filter, forcing the system to either return too few results or re-query with a higher K.

---

## 6. Retrieval Module

Hybrid fusion combines the BM25 and dense retrieval result lists using Reciprocal Rank Fusion. RRF is the correct choice because BM25 scores and cosine similarity scores are incommensurable: BM25 scores are unbounded and depend on term frequency and document length, while cosine similarity is bounded in [0,1] and depends on embedding geometry. Weighted score combination requires normalizing both score ranges and choosing a weight hyperparameter, which is data-dependent and unstable. RRF instead operates on rank positions, which are directly comparable. The RRF formula is: score(chunk) = sum over all retrieval lists of (1 / (k + rank(chunk))), where k is a small constant (typically 60) that reduces the impact of high-rank outliers. Each chunk's RRF score is computed by summing its rank-based contribution from the BM25 list and the dense list. Chunks that appear in both lists receive contributions from both and are ranked higher. Chunks that appear in only one list still receive a contribution from that list and may rank in the top K if their single-list rank is high enough.

MMR is applied post-fusion to ensure diversity in the final result set. The MMR objective is: select the chunk that maximizes (lambda * relevance - (1 - lambda) * max_similarity_to_selected_chunks), where relevance is the RRF score and max_similarity_to_selected_chunks is the maximum cosine similarity between the candidate chunk's embedding and the embeddings of all already-selected chunks. The parameter lambda_mult controls the tradeoff between relevance and diversity. For legal retrieval, lambda_mult is set to 0.6 by default. This is slightly relevance-heavy because legal precedents from the same IPC section or the same constitutional article are often legitimately similar in their reasoning, and penalizing thematic overlap too aggressively would exclude relevant precedents. However, some diversity is necessary because returning ten precedents that all state the same ratio decidendi in slightly different words is not useful. MMR ensures that each retrieved chunk adds new information relative to the already-retrieved set.

Re-ranking is performed with a cross-encoder model over the top K candidates from the MMR stage. The cross-encoder is a fine-tuned transformer that takes the concatenation of the query text and a candidate chunk as input and outputs a relevance score directly, without relying on separate embeddings. The cross-encoder used in the zero-shot configuration is cross-encoder/ms-marco-MiniLM-L-6-v2, a lightweight model trained on MS MARCO passage ranking. For domain-specific re-ranking, the cross-encoder can be fine-tuned on a dataset of Indian SC relevance judgments, such as query-precedent pairs labeled as relevant or not relevant by legal experts. The tradeoff is clear: zero-shot cross-encoder re-ranking is fast to deploy and requires no labeled data, but its precision on legal text is lower because it was trained on web search queries. Fine-tuning on Indian SC data improves precision significantly but requires annotation effort to create training pairs and compute to fine-tune the model. The system is designed so the re-ranker is a swappable component: the hybrid fusion and MMR stages output a ranked list of candidates with metadata, and the re-ranker is a function that takes this list and returns a re-ordered list. This modularity allows deployment with the zero-shot re-ranker initially and upgrading to a fine-tuned model later without changing the upstream pipeline.

Source attribution is enforced as a first-class requirement. Each chunk returned to the user carries the following metadata: source_pdf is the file path or S3 key of the judgment PDF. Page_number is the integer page index where the chunk appears, incremented by one for human readability. Chunk_id is a unique identifier for reproducibility. Section_type indicates whether the chunk came from the Facts, Held, Ratio, or another section of the judgment. Metadata includes the case_id, year, bench_type, act_sections, and outcome_label from the indexing schema. This metadata is surfaced in the user interface alongside the retrieved chunk text, allowing the user to navigate directly to the source PDF and verify the context of the retrieved passage.

---

## 7. Counterfactual Reasoning Module

The counterfactual reasoning module is the novel core of COUNTERCASE and must be designed with full architectural specificity. The input to this module is a structured fact sheet auto-parsed from a query case using the pipeline described in Section 4. The fact sheet schema is: parties contains petitioner_type and respondent_type. Evidence_items is a list of evidence objects, each with an evidence_type and a description. Sections_cited is a list of normalized section strings. Numerical_facts is a dictionary with amounts, ages, and durations. Outcome is the outcome label. This structured representation is what gets perturbed, not the raw text of the case. The query case can be one of the indexed Supreme Court judgments or a new case provided by the user in which case the user must either provide the fact sheet directly or submit case text from which the fact sheet is auto-parsed.

Fact types in scope for perturbation are: numerical and quantitative facts, such as monetary amounts (compensation awarded, loan amount in dispute), ages (of parties or victims), and durations (length of employment, time elapsed between events). Legal sections cited, such as IPC 302 to IPC 304, or Article 14 of the Constitution to Article 21. Party type or relationship, such as employee versus contractor, tenant versus licensee, minor versus adult, petitioner as an individual versus a corporation. Presence or absence of a piece of evidence, such as toggling whether a dying declaration was available or whether a medical report was submitted. All four fact types must be handled by the perturbation pipeline.

Fact perturbation is a hybrid three-stage pipeline. Stage 1 is NER-based identification. A legal NER model is fine-tuned on Indian legal text using labeled corpora from OpenNyAI or annotated subsets of the Supreme Court corpus. The NER model tags legally operative spans in the fact sheet text as one of the following entity types: MONETARY_AMOUNT, AGE, DURATION, LEGAL_SECTION, PARTY_ROLE, EVIDENCE_TYPE. These tagged spans become perturbation candidates. The NER model must be domain-specific because general-purpose NER models do not reliably recognize act section patterns like "IPC 302" or party role terms like "licensee". Fine-tuning is performed on a small labeled dataset of 500 to 1000 fact sheet sentences annotated by legal experts, using a BERT-based sequence tagger.

Stage 2 is rule-based perturbation. For each tagged span type, a deterministic perturbation rule is applied. Numerical facts are scaled by a legally meaningful factor. Monetary amounts are shifted above or below statutory thresholds, such as the threshold for summary proceedings under the CPC or the compensation cap under the Motor Vehicles Act. Ages are moved across legally significant boundaries, such as 18 years for majority, 7 years for the age of criminal responsibility under IPC, or 60 years for retirement age in service matters. Durations are scaled by a factor of two or halved, or shifted to cross a statutory limitation period. Legal sections are substituted using a hand-curated section adjacency map. The adjacency map is a graph where nodes are legal sections and edges connect sections in the same chapter or sections that are commonly cited together in case law. For example, IPC 302 (murder) is adjacent to IPC 304 (culpable homicide), IPC 304A (death by negligence), and IPC 300 (definition of murder). The map is curated by legal experts and encodes domain knowledge about which section substitutions are plausible. Party types are swapped along predefined axes from a closed vocabulary. The party type ontology includes: Individual, Minor, Corporation, Partnership, State, UnionOfIndia, ForeignNational, PublicSector, PrivateSector, Employee, Contractor, Tenant, Licensee. Swaps are applied along legally meaningful axes: Individual to Minor, Employee to Contractor, Tenant to Licensee. Evidence presence is toggled as a boolean: if an evidence item is present in the original fact sheet, a perturbed version is created with that item removed; if an evidence item is absent, a perturbed version is created with a plausible item added, selected from a predefined list of common evidence types for the case category.

Stage 3 is LLM validation. Each perturbed fact sheet generated by the rule-based stage is passed to an LLM with a validation prompt. The LLM is a lightweight quantized Mistral-7B model run locally, or a hosted API call if compute is CPU-only. The validation prompt is: "You are a legal expert. A case has the following original facts: {original_fact_sheet}. A counterfactual version has been generated with the following altered facts: {perturbed_fact_sheet}. Answer two questions: (1) Is this counterfactual legally plausible—could a case with these altered facts plausibly occur in Indian jurisprudence? (2) Does the alteration change a legally operative fact, or is it a minor stylistic change that would not affect legal reasoning? Respond with JSON: {plausible: true/false, operative: true/false, reasoning: brief explanation}." The LLM output is parsed and perturbations that score false on either plausible or operative are discarded. This validation filter prevents nonsensical perturbations like changing IPC 302 to an unrelated section from another act, or changing a party's age from 25 to 26 when no age-based threshold is crossed.

The perturbation tree architecture is the core data structure. The tree is a directed acyclic graph where each node is a fact sheet state and each directed edge represents a single fact change. The root node is the original fact sheet. Level 1 children are single-fact perturbations of the root: each perturbation candidate that passes LLM validation becomes a child node connected by an edge labeled with the fact type and the change description. Level 2 children are single-fact perturbations applied to Level 1 nodes, meaning two facts have been changed from the original. A Level 2 node has two parent paths: one direct path from the root through a Level 1 node, and the set of all Level 1 ancestors. The tree depth is capped at a configurable maximum, with a recommended default of depth 3. Branching factor per node is also configurable: each node generates all valid perturbations from Stage 2, but only the top N by LLM plausibility score are retained as children to prevent combinatorial explosion.

Tree construction proceeds breadth-first. The root node is expanded first: all perturbation candidates are identified by NER, perturbed by rule, validated by LLM, and added as Level 1 children. Then each Level 1 node is expanded in the same way, producing Level 2 children. This continues until the depth cap is reached or no new children can be generated. The construction is lazy: nodes are expanded on-demand when the user navigates to them, rather than pre-computing the entire tree.

For each node in the tree, the full hybrid retrieval pipeline is executed. The retrieval query is constructed from the fact sheet at that node by concatenating the party types, the sections cited, a summary of the numerical facts, and the evidence items into a structured query string. This query string is passed through the pre-filter, BM25 and ANN search, RRF fusion, MMR re-ranking, and cross-encoder re-ranking as described in Section 6. The top K result set is stored at the node along with the fact sheet state and the edge description from the parent.

User interaction is bidirectional. The system auto-generates perturbations and constructs the tree, but the user can also manually edit a fact sheet at any node in the UI. The UI presents the fact sheet as a form with editable fields. The user can change a party type, add or remove a section citation, modify a numerical value, or toggle an evidence item. When the user commits the edit, the system creates a new child node with the manually edited fact sheet and runs retrieval from that node. The new node is attached to the tree as a child of the node where the edit was made. Both auto-perturbation and manual editing produce nodes in the same tree structure, giving the user full control to explore directions the auto-perturbation logic did not consider.

Comparison and sensitivity scoring is performed for every parent-child pair of nodes in the tree. The system computes the following: dropped cases are precedents that appeared in the parent's top K result set but do not appear in the child's top K. These are precedents that were sensitive to the fact change. Newly appeared cases are precedents that appear in the child's top K but were not in the parent's top K. These are precedents that became applicable only under the perturbed fact state. Rank displacement per case is defined as the absolute difference in rank position between the parent and child result sets. A case present in both sets with ranks r_parent and r_child has displacement |r_parent - r_child|. A case present in only one set is assigned a rank of K+1 in the set where it is absent, so its displacement is |r_present - (K+1)|. Aggregate fact sensitivity score per fact dimension is computed as the mean rank displacement across all tree edges where that fact type was perturbed. For example, if three edges in the tree perturb the age fact and the mean rank displacement across those three edges is 8.5, then age has a sensitivity score of 8.5 for this case. This allows the user to identify which fact types are most legally operative for the query case.

The side-by-side diff view presents two result sets from a parent and child node. Cases are color-coded: stable cases present in both sets at similar rank (displacement less than 3) are shown in neutral color. Dropped cases are shown in red with their rank in the parent set and the label "not retrieved after fact change". Newly appeared cases are shown in green with their rank in the child set and the label "newly applicable". Each case is displayed with its rank in both sets, its source citation, and its sensitivity score. The user can click on a case to view the full retrieved chunk and its explanation. The diff view also displays the edge description, such as "Age changed from 25 to 17 (now a minor)", and the aggregate sensitivity score if multiple edges are being compared.

---

## 8. Explanation Engine

The explanation engine generates two types of natural language justifications: per-result explanations and counterfactual explanations. Both are grounded strictly in retrieved text and metadata to prevent hallucination.

Per-result explanation is generated for each retrieved case in a result set. The explanation is a one or two sentence justification of why this case is relevant to the query case. The justification is constructed by identifying shared features between the query fact sheet and the retrieved chunk metadata. Shared act sections are the strongest signal: if the query case involves IPC 302 and the retrieved case also involves IPC 302, the explanation states "This case involves Section 302 IPC, which is also cited in your case." Matching factual patterns are identified by token-level similarity between the query fact sheet text and the retrieved chunk text. BM25 term overlap scores are computed per term: terms that appear in both the query fact sheet and the retrieved chunk with high IDF weights are highlighted. The explanation states "This case involves a similar factual pattern regarding {term1}, {term2}, as evidenced by the retrieved text." Similar party roles are identified by comparing the petitioner_type and respondent_type fields: if the query case has petitioner_type "Minor" and the retrieved case metadata indicates a minor petitioner, the explanation states "This case also involves a minor as the petitioner, which may affect the legal analysis." Cross-encoder attention weights can be used if the cross-encoder model exposes attention scores: the tokens in the retrieved chunk with the highest attention weights when paired with the query are the tokens that drove the match, and these are included in the explanation. The critical constraint is that the explanation must not assert anything not present in the retrieved chunk text or metadata. If the explanation generator cannot identify a grounded reason for the match, it states "This case was retrieved based on overall semantic similarity" rather than fabricating a specific shared feature.

Counterfactual explanation is generated for each edge in the perturbation tree. The explanation is a one-paragraph summary answering three questions: what fact changed, which cases dropped out and why, and which cases appeared and what in them matches the new fact state. The explanation template is: "When {fact_type} was changed from {original_value} to {perturbed_value}, {N_dropped} precedents dropped out of the top K results and {N_new} new precedents became applicable. The dropped cases include {case_1}, {case_2}, ..., which were relevant under the original facts because {explanation_of_dependence_on_original_fact}. The newly appeared cases include {case_A}, {case_B}, ..., which are applicable under the perturbed facts because {explanation_of_match_to_perturbed_fact}." The explanation of dependence is constructed by examining the dropped cases: if a dropped case cited the same legal section as the original fact sheet but that section was changed in the perturbation, the explanation states "these cases cited Section {X}, which is no longer applicable after the section citation was changed to Section {Y}." If a dropped case involved a party type that was changed, the explanation states "these cases involved a {original_party_type}, but the perturbed case involves a {perturbed_party_type}, which falls under different legal principles." The explanation of match to the perturbed fact is constructed symmetrically for newly appeared cases. This analysis is deterministic and rule-based: the explanation generator does not call an LLM for free-form generation because that introduces hallucination risk. Instead, it uses template filling with metadata fields and BM25 term overlap scores.

Output format is structured JSON with a derived text summary. The JSON schema is: node_id is a unique identifier for the tree node. Fact_sheet_state is the full fact sheet object at this node. Retrieval_results is a list of retrieved chunks, each with its chunk_id, source_pdf, page_number, section_type, rank, RRF score, cross-encoder score, and metadata. Diff_vs_parent is an object containing dropped_cases, newly_appeared_cases, stable_cases, and per-case rank displacement scores, or null if this is the root node. Sensitivity_scores is a dictionary mapping fact dimension names to their aggregate sensitivity score. Explanations is a dictionary mapping result chunk_ids to their per-result explanation strings, plus a counterfactual_summary field if this node is not the root. The JSON is the ground truth output. The human-readable text summary is generated deterministically from the JSON by concatenating the per-result explanations and the counterfactual summary with section headers and formatting. This prevents hallucination by construction: the text summary is a projection of the JSON, not a separate generative process.

---

## 9. System Architecture — End-to-End Data Flow

The full pipeline operates as follows:

```
PDF Corpus (AWS Data Exchange — Indian Supreme Court Judgments)
        |
        | [S3 Sync or ADX API Download]
        v
[Ingestion + Text Extraction]
    Tool: pdfplumber
    Handles: multi-column layouts, headers/footers, cause-list noise, page numbering
    Output: raw text per page with page indices and PDF path
        |
        v
[Preprocessing]
    Section detection: identify Facts, Issues, Held, Ratio, Obiter via heading patterns
    Noise removal: discard headers/footers, cause-list boilerplate
    Metadata extraction: regex + heuristics for case_id, year, bench_type, act_sections, judge_names, outcome_label
    Output: clean text segmented by section, plus metadata dict
        |
        +---> [Fact Sheet Extraction]
                Rule-based: locate Facts section by heading match or fallback to first 20%
                LLM-based: pass Facts section to Mistral-7B with schema prompt
                Output: structured fact sheet JSON {parties, evidence_items, sections_cited, numerical_facts, outcome}
                Stored: per case in a fact sheet database keyed by case_id
        |
        v
[Chunking]
    Tool: LangChain RecursiveCharacterTextSplitter
    Config: chunk_size=1024 tokens, overlap=128 tokens, separators=['\n\n', section-number patterns, char boundary]
    Tagging: each chunk tagged with section_type inherited from preprocessing
    Output: list of chunks, each with text, section_type, source_pdf, page_number, chunk_id
        |
        v
[Dual Indexing]
    BM25 Index:
        Tool: rank_bm25 library (or Elasticsearch for production)
        Input: chunk text
        Output: inverted index mapping terms -> (chunk_id, term_freq)
    ChromaDB Index:
        Embedding model: all-MiniLM-L6-v2
        Batch size: 32 chunks per embedding call
        Metadata fields: year, court, bench_type, act_sections (list), section_type, outcome_label, chunk_id, source_pdf, page_number
        ANN: HNSW index over 384-dim embeddings
        Output: vector store with metadata-filterable ANN search
        |
        v
[Query Input]
    User provides: raw case text OR structured fact sheet OR case_id of indexed judgment
        |
        v
[Auto Fact Sheet Parsing]
    If raw text: run Fact Sheet Extraction pipeline (rule + LLM)
    If case_id: lookup fact sheet from fact sheet database
    If structured fact sheet: use directly
    Output: root fact sheet for perturbation tree
        |
        v
[Root Node of Perturbation Tree]
    Node: {node_id: 0, fact_sheet: root_fact_sheet, parent: null, edge_description: null, retrieval_results: null}
        |
        v
[Hybrid Retrieval at Root Node]
    Stage 1: Metadata Pre-filter
        ChromaDB where clause: filter by year_range, act_sections, bench_type if specified
        Reduces candidate set before ANN search
    Stage 2: BM25 + ANN Vector Search
        BM25: query = fact sheet converted to text query string
        BM25 returns top 50 chunks by BM25 score
        ANN: query embedding = all-MiniLM-L6-v2 embedding of query string
        ChromaDB HNSW search over filtered subset returns top 50 chunks by cosine similarity
    Stage 3: RRF Fusion
        Combine BM25 ranks and ANN ranks via Reciprocal Rank Fusion
        RRF score = sum over lists of (1 / (60 + rank))
        Output: merged list of chunks sorted by RRF score
    Stage 4: MMR Re-ranking
        Select top K chunks by MMR with lambda_mult=0.6
        Ensures diversity: each selected chunk must be sufficiently dissimilar to already-selected chunks
        Output: top K diverse and relevant chunks
    Stage 5: Cross-encoder Re-ranking
        Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (zero-shot) or fine-tuned InLegalBERT cross-encoder
        Input: (query string, chunk text) pairs
        Output: relevance scores per pair
        Re-sort top K by cross-encoder score
    Stage 6: Source Attribution
        Attach metadata {source_pdf, page_number, chunk_id, section_type, case metadata} to each result chunk
    Store: retrieval_results at root node
        |
        v
[Perturbation Tree Expansion]
    For each node at current depth (starting with root):
        NER Tagging:
            Fine-tuned legal NER model tags fact sheet spans as MONETARY_AMOUNT, AGE, DURATION, LEGAL_SECTION, PARTY_ROLE, EVIDENCE_TYPE
        Rule-based Perturbation:
            For each tagged span, apply fact-type-specific perturbation rule:
                Numerical: scale across legal thresholds
                Sections: substitute via adjacency map
                Party type: swap along predefined axis
                Evidence: boolean toggle
        LLM Plausibility Validation:
            Pass each perturbed fact sheet + original to Mistral-7B with validation prompt
            Prompt returns {plausible: bool, operative: bool, reasoning: str}
            Discard if plausible=false or operative=false
        Child Node Creation:
            For each valid perturbation, create child node with perturbed fact sheet and edge description
            Attach child to parent in tree
        On-demand Expansion:
            Nodes expanded lazily when user navigates to them (optional optimization)
    Repeat until depth cap reached
        |
        v
[Per-node Retrieval]
    For each node in tree:
        Run full hybrid retrieval pipeline (Stage 1 through Stage 6) with that node's fact sheet as query
        Store retrieval_results at that node
        |
        v
[Diff + Sensitivity Scoring]
    For each parent-child edge:
        Compare retrieval_results of parent and child:
            Dropped cases: in parent top K but not in child top K
            Newly appeared cases: in child top K but not in parent top K
            Stable cases: in both, with rank displacement < 3
        Rank displacement per case: |rank_parent - rank_child|, where absent = K+1
        Store diff at child node as diff_vs_parent
    Aggregate sensitivity per fact dimension:
        For each fact type (age, section, party_type, evidence, amount):
            Collect all edges where that fact type was perturbed
            Mean rank displacement across those edges -> sensitivity score for that fact type
        Store sensitivity_scores at root node (or per node if multiple paths through tree)
        |
        v
[Explanation Engine]
    Per-result Explanation:
        For each retrieved chunk:
            Identify shared features: act_sections, BM25 term overlap, party_type match
            Template: "This case involves {feature X}, which is also present in your case."
            Ground in metadata and chunk text only
        Store in explanations dict keyed by chunk_id
    Counterfactual Explanation:
        For each edge:
            Template: "When {fact} changed from {A} to {B}, {N} cases dropped and {M} appeared because {reason}."
            Reason: rule-based analysis of metadata overlap with dropped/appeared cases
        Store as counterfactual_summary in child node
        |
        v
[JSON Output]
    Schema: {node_id, fact_sheet_state, retrieval_results, diff_vs_parent, sensitivity_scores, explanations}
    Export entire tree as JSON or per-node JSON on-demand
        |
        v
[Text Summary Generation]
    Deterministic projection of JSON to human-readable text:
        Section 1: Query case fact sheet
        Section 2: Retrieved precedents with per-result explanations
        Section 3: Counterfactual analysis with diff tables and sensitivity scores
        Section 4: Perturbation tree structure with edge descriptions
    Output: Markdown or plain text report
        |
        v
[Output to User]
    UI Components:
        Side-by-side diff view: color-coded dropped/stable/new cases with ranks
        Perturbation tree visualisation: interactive tree graph, click to navigate
        Sensitivity scores: bar chart or table per fact dimension
        Source-attributed retrieved chunks: links to PDF and page number
        Explanations: per-result and counterfactual summaries in expandable panels
        Manual edit interface: form to edit fact sheet fields and trigger re-retrieval
    Export: JSON and text summary downloadable
```

---

## 10. Evaluation Plan

Retrieval quality is evaluated on a held-out test set of Supreme Court judgments with manually annotated relevant precedents. The test set is constructed by selecting 100 cases spanning multiple decades and legal domains. For each test case, two legal experts independently identify up to 20 relevant precedents from the corpus. Relevance is defined as: a precedent is relevant if it addresses the same legal issue or applies the same legal principle, regardless of whether the factual pattern is identical. Disagreements between annotators are resolved by a third expert. The test set is split by time: cases from 2020 to 2024 are held out, and the system is indexed on cases from 1950 to 2019. This prevents data leakage where the test query case cites a precedent that was decided after the query case.

Metrics are MRR@10 (Mean Reciprocal Rank of the first relevant precedent in the top 10), NDCG@10 (Normalized Discounted Cumulative Gain at rank 10, which accounts for multiple relevant precedents at different ranks), and Recall@K for K in [5, 10, 20] (fraction of relevant precedents retrieved in top K). These metrics are standard in information retrieval and allow comparison to prior work on legal retrieval.

Baselines to beat are: BM25-only retrieval without dense embedding search. Dense-only retrieval with all-MiniLM-L6-v2 without BM25. Hybrid retrieval with RRF but without MMR. Hybrid retrieval with MMR but without cross-encoder re-ranking. Each baseline is evaluated on the same test set with the same metrics. The expectation is that BM25-only performs well on queries with specific legal terms but poorly on paraphrased queries. Dense-only performs well on paraphrased queries but poorly on queries requiring exact section matches. Hybrid without MMR suffers from redundancy in the top K. Hybrid without cross-encoder re-ranking has lower precision because it relies on embedding similarity rather than direct relevance scoring. The full system with all components should achieve the highest scores across all metrics.

Counterfactual module evaluation is qualitative and quantitative. Quantitative: select a set of cases where the legal outcome is known to hinge on a specific fact. Examples include cases decided under IPC 304 versus IPC 302 based on whether the death was intentional or accidental, cases involving the age of majority where the outcome differs if the party is a minor, and cases involving the admissibility of dying declarations where the presence of the declaration is dispositive. For each such case, perturb the dispositive fact and verify that the rank displacement score is high. Perturb a non-dispositive fact (such as the month in which the case was filed, if that is not relevant to the legal issue) and verify that the rank displacement score is low. A successful counterfactual module produces high sensitivity scores for legally operative facts and low scores for non-operative facts. This can be quantified as a correlation metric: compute the Spearman rank correlation between human-annotated fact importance (experts rank facts by legal operability) and system-computed sensitivity scores. A correlation above 0.6 indicates the system is detecting legally meaningful facts.

Qualitative: the LLM validation filter's accept/reject rate is logged. A healthy rate is 60 to 80 percent acceptance: if acceptance is too high, the validation is too permissive and nonsensical perturbations are passing; if acceptance is too low, the rule-based perturbations are generating too many implausible candidates and the LLM is doing all the work. Manually audit a sample of 50 accepted perturbations: two legal experts review each perturbed fact sheet and label it as plausible or implausible. Inter-annotator agreement and agreement with the LLM filter is measured. If the LLM agrees with human experts on 80+ percent of cases, the validation filter is reliable.

Perturbation tree coverage is measured by comparing the auto-generated tree to a gold-standard tree created by legal experts. For a subset of 20 test cases, experts manually enumerate all legally meaningful fact perturbations and arrange them in a tree structure. The system-generated tree is compared to the gold tree: what fraction of expert-identified perturbations appear in the system tree at depth 2? At depth 3? What fraction of system-generated perturbations are not in the expert tree and are thus spurious? High recall (system tree contains most expert perturbations) and high precision (few spurious perturbations) indicate good coverage.

Explanation quality is evaluated by faithfulness and human evaluation. Faithfulness: for a sample of 100 per-result explanations, annotators check whether each sentence in the explanation can be traced to a span in the retrieved chunk text or metadata. A sentence is faithful if it asserts only information present in the source. The faithfulness score is the fraction of explanation sentences that pass this check. A score above 90 percent is the target. Human evaluation: 50 test cases with their retrieval results and counterfactual explanations are reviewed by five legal domain experts. Each expert rates each explanation on a 5-point Likert scale for: clarity (is the explanation understandable to a lawyer?), legal accuracy (does the explanation correctly describe the legal relationship?), and usefulness of the counterfactual summary (does it help the user understand how the fact change affected relevance?). Mean scores and inter-rater reliability (Krippendorff's alpha) are reported. Scores above 3.5/5 indicate the explanations are useful.

---

## 11. Implementation Roadmap

Phase 1 focuses on foundational data infrastructure. Deliverables: ADX dataset ingestion script with S3 sync or API download, automated retry on network failure, and logging of ingested PDF paths. PDF text extraction pipeline using pdfplumber with multi-column layout handling and header/footer removal. Metadata extraction for case_id, year, bench_type, act_sections, judge_names, outcome_label, validated on a sample of 500 judgments by manual review. BM25 index built with rank_bm25 library and ChromaDB index built with all-MiniLM-L6-v2 embeddings and HNSW. Baseline hybrid retrieval with RRF fusion tested on a small manually annotated set of 20 query-precedent pairs. Acceptance criteria: retrieval returns results in under 2 seconds per query for a corpus of 10,000 judgments, and manual review confirms correct metadata extraction on 95+ percent of sampled cases.

Phase 2 integrates retrieval enhancements. Deliverables: MMR re-ranking with configurable lambda_mult parameter and validation that top K results are diverse. Cross-encoder re-ranking with cross-encoder/ms-marco-MiniLM-L-6-v2 in zero-shot mode, with latency profiling to ensure re-ranking of top 50 candidates completes in under 500ms. Source attribution metadata attached to all retrieved chunks with validation that source_pdf and page_number are correct by spot-checking 50 results. Evaluation harness implemented with MRR@10, NDCG@10, Recall@K metrics computed over the held-out test set. Acceptance criteria: hybrid retrieval with MMR and cross-encoder achieves MRR@10 above 0.5 and NDCG@10 above 0.6 on the test set, and source attribution links are correct on 100 percent of spot-checked results.

Phase 3 builds the structured fact sheet extraction pipeline. Deliverables: rule-based section detection for locating the Facts section in judgment text, validated by manual review on 100 judgments. LLM-based fact sheet population using Mistral-7B quantized model run locally or via API, with schema validation and retry logic. Fact sheet database keyed by case_id with JSON storage. NER-based perturbation candidate tagging using a fine-tuned legal NER model on a labeled dataset of 1000 annotated fact sheet sentences. Acceptance criteria: fact sheet extraction pipeline produces valid JSON for 90+ percent of test cases, and NER model achieves F1 above 0.8 on held-out NER test set for LEGAL_SECTION, AGE, MONETARY_AMOUNT entity types.

Phase 4 implements the perturbation logic. Deliverables: rule-based perturbation functions for all four fact types—numerical, section, party type, evidence—with unit tests covering edge cases. Section adjacency map curated by legal experts covering IPC, CPC, Constitution, and major central acts, with at least 200 section-to-section edges. LLM validation filter using Mistral-7B with plausibility and operability checks, with logging of accept/reject decisions. Single-level perturbation tree construction: given a root fact sheet, expand to Level 1 children and store. Basic diff view showing dropped and newly appeared cases for one parent-child pair. Acceptance criteria: rule-based perturbations generate legally plausible candidates on manual review of 50 samples, LLM validation filter achieves 60-80 percent acceptance rate, and diff view correctly identifies dropped/new cases by comparing rank lists.

Phase 5 extends the tree to multiple levels and adds user interaction. Deliverables: perturbation tree at depth 2 and 3 with breadth-first expansion and configurable depth cap. User manual editing interface allowing field-level edits to fact sheets, with JSON schema validation before accepting edits. Aggregate sensitivity scoring per fact dimension computed as mean rank displacement across all edges where that fact type was perturbed. Side-by-side diff UI with color-coded stable/dropped/new cases, rank positions, and edge descriptions. Acceptance criteria: tree construction completes in under 5 minutes for depth 3 with branching factor 5, manual edits correctly trigger re-retrieval and node creation, and sensitivity scores correlate with expert-annotated fact importance at Spearman rho above 0.6 on a test set of 20 cases.

Phase 6 implements the explanation engine. Deliverables: per-result explanation generator using BM25 term overlap and metadata matching, with template-based text generation. Counterfactual explanation generator for edges, using rule-based analysis of dropped and appeared cases. JSON output format with schema validation, serialization to file. Text summary generator that deterministically projects JSON to Markdown with section headers, tables, and inline links. Acceptance criteria: faithfulness evaluation shows 90+ percent of explanation sentences are grounded in source text, and human evaluation by legal experts rates explanations above 3.5/5 on clarity and usefulness on a sample of 50 cases.

Phase 7 conducts ablation studies and prepares the research writeup. Deliverables: full evaluation on held-out test set comparing all baselines—BM25-only, dense-only, hybrid without MMR, hybrid without cross-encoder—against the full system. Statistical significance testing of metric differences using paired t-tests. Counterfactual module evaluation on cases with known dispositive facts, with quantitative sensitivity scores and qualitative audit of perturbation plausibility. Explanation quality evaluation with faithfulness and human ratings. Research paper draft covering novelty, architecture, evaluation, and open questions. Acceptance criteria: full system achieves statistically significant improvement over all baselines on MRR@10 and NDCG@10 with p < 0.05, counterfactual module achieves Spearman correlation above 0.6, and explanation faithfulness is above 90 percent.

---

## 12. Key Risks and Mitigations

Fact sheet extraction quality is a critical dependency because all downstream counterfactuals depend on a correctly parsed fact sheet. If the NLP pipeline mis-parses the facts section—for example, misidentifying the Submissions section as Facts, or failing to extract a key section citation—the perturbations will be applied to incomplete or incorrect data, rendering the counterfactual analysis wrong. Mitigation: expose the parsed fact sheet to the user in the UI before the perturbation tree is built. The user reviews the fact sheet and can manually correct any errors in a structured form. Include a confidence score from the LLM parser and flag low-confidence cases for mandatory user review. Log all parsing failures with the case_id and PDF path for offline debugging and retraining of the LLM parser on failure cases.

LLM validation cost and latency is a risk because calling an LLM for every perturbation candidate multiplies latency at tree depth. At depth 3 with branching factor 5, there are 5 Level 1 nodes, 25 Level 2 nodes, and 125 Level 3 nodes, each requiring validation—totaling 155 LLM calls per tree. At 200ms per call, this is 31 seconds of LLM latency alone. Mitigation: cache validation results keyed by (original_fact_sheet, perturbed_fact_sheet, perturbation_type). Many perturbations are structurally similar, such as changing age from 25 to 17, vs. age from 30 to 17—both cross the majority boundary. Cache the validation result and reuse it for similar perturbations. Batch API calls: send multiple validation prompts in a single API request if the LLM provider supports batching. Use a small quantized local model like Mistral-7B-Instruct-GGUF on CPU or GPU rather than a hosted API where compute allows, reducing per-call latency to under 50ms.

Perturbation tree explosion is a risk at depth 3 with multiple fact types. If each fact type has 3 perturbation values and there are 4 fact types, each node can have 12 children, leading to 12^3 = 1728 nodes at depth 3. Mitigation: cap the branching factor per node. Rank perturbation candidates by LLM plausibility score and retain only the top 3 to 5 per node. Prune nodes dynamically: after a child node is created and retrieval is run, compare its retrieval results to its parent. If the rank displacement is below a threshold—say, mean displacement less than 1.0, indicating the perturbation had negligible effect—mark that branch as uninteresting and do not expand it further. This prunes branches where the counterfactual did not meaningfully change retrieval, focusing compute on high-sensitivity paths through the tree.

Hallucination in explanations is a risk if the explanation generator uses free-form LLM generation without grounding constraints. An LLM might assert "This case applies because the defendant was a minor" when the retrieved chunk does not mention age. Mitigation: constrain the explanation generator to extractive or template-based methods only. Per-result explanations are generated by identifying overlap between the query fact sheet and the retrieved chunk metadata, then filling a template with those overlapping features. If no overlap is found, the explanation states "retrieved due to overall similarity" rather than fabricating a reason. Counterfactual explanations are rule-based: they analyze the metadata of dropped and appeared cases and fill a template with deterministic logic. Do not call an LLM to generate counterfactual summaries in free-form unless the LLM output is post-processed by an entailment checker that verifies every sentence is grounded in the input data.

Dataset coverage gaps are a limitation because the ADX corpus covers only the Supreme Court, excluding High Courts and subordinate courts. A user querying with a High Court case may find that the retrieved precedents are all Supreme Court judgments, which are binding but may not address the specific procedural issue the user cares about. Mitigation: scope the system explicitly to Supreme Court cases in the documentation and UI. Display a disclaimer: "This system retrieves precedents from the Indian Supreme Court corpus. High Court and subordinate court judgments are not included." In the research writeup, document the boundary clearly and frame the system as a Supreme Court precedent search tool, not a general Indian case law search tool. This sets correct expectations and prevents reviewers from penalizing the system for lacking coverage that was never in scope.

Compute uncertainty is a risk because the project does not yet have a fixed compute allocation and may need to run on CPU-only infrastructure during development. Mitigation: design all model components as swappable with lightweight defaults. The embedding model default is all-MiniLM-L6-v2, which is CPU-compatible and produces embeddings in under 50ms per chunk on a modern CPU. The cross-encoder default is cross-encoder/ms-marco-MiniLM-L-6-v2 in zero-shot mode, which is also CPU-compatible. The LLM validator uses Mistral-7B in quantized GGUF format with llama.cpp, which runs on CPU at acceptable latency. Document upgrade paths explicitly: if a GPU becomes available, upgrade the cross-encoder to a fine-tuned InLegalBERT cross-encoder, upgrade the embedding model to a larger legal-domain model, and run the LLM validator on GPU for 5x speedup. Write the retrieval pipeline so the embedding model and cross-encoder are passed as constructor parameters, not hard-coded, allowing a one-line configuration change to swap models.

---

## 13. Open Research Questions

Several genuine open questions emerge from this architecture that could become research contributions in the writeup. These are empirical questions that the system is designed to answer through evaluation, not questions with pre-determined answers.

What is the right granularity for legal fact perturbation—entity span level versus structured schema field level—and does granularity affect the legal plausibility of generated counterfactuals? The current design perturbs at the schema field level: the fact sheet has discrete fields like age, party_type, sections_cited, and each field is perturbed independently. An alternative design would perturb at the entity span level: tag every span in the raw case text with an entity type and perturb in-place, producing a modified text rather than a modified fact sheet. Field-level perturbation has the advantage of producing structured data that is easy to query and compare, but it may lose nuance because multiple related facts are reduced to a single field value. Span-level perturbation preserves nuance but produces unstructured output that is harder to compare. Does one approach generate more legally plausible counterfactuals in human evaluation? This can be tested by implementing both and comparing expert ratings of plausibility.

How should the perturbation tree be pruned without losing legally meaningful branches? The current design prunes branches where the rank displacement is low, under the assumption that a perturbation with no effect on retrieval is not interesting. But this assumes that retrieval rank displacement is a valid proxy for legal meaningfulness. A small textual change—such as toggling the presence of a dying declaration—might produce only a rank displacement of 2, meaning most precedents stay in the top K but shift slightly in rank. However, this could be legally significant because the admissibility of dying declarations is a dispositive issue in certain cases. A retrieval-similarity-based pruning criterion might discard this branch as "low impact" when it is actually high legal impact. How should the pruning threshold be set to balance computational efficiency with coverage of legally meaningful variations? This is an empirical question that can be answered by comparing expert-generated gold-standard trees to system-generated trees with different pruning thresholds and measuring recall of expert-identified important branches.

Is rank displacement a valid proxy for legal sensitivity, or do legal experts find that rank changes are sometimes caused by lexical artefacts rather than genuine legal distinction? Rank displacement quantifies how much the retrieval results change when a fact is perturbed, but retrieval is not a perfect oracle of legal relevance. A case might drop in rank because the perturbed fact sheet uses different terminology that the BM25 index does not match, even though the legal principle is the same. Conversely, a case might remain in the top K despite a legally dispositive fact change because it shares many other lexical features with the query. The human evaluation in Section 10 is designed to measure this: experts will review cases flagged as "high sensitivity" by the system and judge whether the sensitivity is legally justified or an artefact of the retrieval model. If experts disagree with the system's sensitivity scores on more than 30 percent of cases, then rank displacement is not a valid proxy and alternative sensitivity metrics should be explored, such as outcome-change prediction or explicit legal principle matching.

Can the topology of the perturbation tree itself—its depth, branching density, and the distribution of sensitivity scores across edges—be used to characterise how legally robust a case is as a precedent? A case whose retrieval results are insensitive to all perturbations may be a more stable and universally applicable precedent than one that is highly sensitive. For example, a constitutional principle case like Kesavananda Bharati (basic structure doctrine) should be retrieved regardless of the specific facts of the query case, and its sensitivity scores should be low across all fact dimensions. In contrast, a case decided on a narrow factual distinction—such as whether a party was a licensee or a tenant—should have high sensitivity to the party_type fact and low sensitivity to other facts. If this hypothesis holds, the tree topology could be a new metric for precedent robustness: stable precedents have flat sensitivity distributions and shallow trees because few perturbations affect their applicability, while narrow precedents have spiked sensitivity distributions and deep trees because their applicability hinges on specific facts. This can be tested by computing tree metrics for a set of known broad versus narrow precedents and measuring whether the tree topology correlates with expert annotations of precedent scope.

---

## End of Plan

This plan provides a comprehensive, opinionated technical specification for COUNTERCASE. The architecture is designed to be deployable on flexible compute, evaluated with rigour against baselines, and capable of producing novel research contributions in Indian legal NLP. The perturbation tree framework, hybrid metadata-aware retrieval with MMR and re-ranking, and grounded explanation engine together fill a critical gap in legal AI: the ability to reason about which facts causally determine precedent applicability. Implementation proceeds through seven phased milestones with clear acceptance criteria, and the system is designed to handle the known quality issues in the AWS Data Exchange Indian Supreme Court Judgments corpus while requiring zero counterfactual-specific annotation. All risks have documented mitigations, and the open research questions provide avenues for publishable contributions beyond the working prototype.
