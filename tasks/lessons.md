# COUNTERCASE - Lessons Learned

## Phase 1

- **Conda env required**: The `torch2` conda environment must be activated before running any pipeline scripts. The base Python env is missing `tiktoken`, `langchain-text-splitters`, and other dependencies.
- **Nested f-string quoting**: When running Python one-liners via PowerShell, nested quotes in f-strings (e.g., `meta.get("key")`) cause `SyntaxError`. Use a script file instead of inline `-c` for complex tests.
- **DPR pooler weights warning**: The DPR models from HuggingFace emit an `UNEXPECTED` key warning for `pooler.dense.weight/bias`. This is benign and can be ignored -- the pooler is not used by DPR encoders.
- **ChromaDB cosine distance**: ChromaDB returns distances (0=identical, 2=opposite) not similarities. Must convert: `score = 1.0 - distance / 2.0`.
- **Metadata schema richness**: The ADX parquet metadata for 2024 includes: `title`, `petitioner`, `respondent`, `judge`, `author_judge`, `citation`, `case_id`, `cnr`, `decision_date`, `disposal_nature`, `court`, `path`, `year`, etc. Much richer than expected -- `author_judge` has 782 nulls, all other key fields are fully populated.
