# Typology retrieval

Implement the retrieval half of a RAG system: use TF-IDF + cosine similarity to rank a small
document corpus (six money-laundering typologies) by relevance to a query. This is the keyword
half — lesson 009 adds the embedding half to create a hybrid retriever.

## Key concept

**RAG's retrieval half starts here.** A large language model alone knows facts only from its
training data and reasoning, not from a specialized domain corpus you've collected. To make an
LLM's output faithful to your own documents — whether a technical handbook, a set of research
papers, or a set of known fraud typologies — you retrieve the most relevant documents first, then
pass them to the LLM as context. That retrieval step is this lesson.

**TF-IDF + cosine similarity** is the right tool for a small, fixed corpus like six typologies:

- **No model download:** TF-IDF is pure linear algebra (not a downloaded neural network), so there
  is nothing to download, no quota limits on calls, and no cold start latency.
- **Exact reproducible ranking:** every query produces the same ranking every time — useful for
  debugging and compliance audits.
- **Zero new dependencies:** scikit-learn is already in your dependency stack (lesson 006 added it
  for the baseline classifier).

For larger corpora — thousands or millions of documents — this approach breaks: TF-IDF's
computational cost climbs with corpus size, and keyword matching alone misses semantic
relationships (a query about "illegal transfer" should match "money laundering" even if those
exact words never appear together in the corpus). At that scale, a hybrid retriever combining
keyword search with a learned embedding model becomes necessary.

**At a regulated shop:** this is the keyword/BM25 half of a hybrid retrieval pipeline typically
run on OpenSearch/Elasticsearch or Postgres+pgvector — audited infrastructure that a compliance
team already knows how to monitor and retain logs from. Reusing already-certified infrastructure
rather than onboarding a new vector-DB vendor sidesteps licensing, audit, and data residency
friction. Lesson 009 adds the embedding half this lesson deliberately leaves out.

## Implementation order

1. **Place the typology corpus.** Copy `aml-tutor/tests/fixtures/typologies.json` into
   `aml-triage/data/typologies.json` — original content, a plain file copy, no download or license
   friction (unlike the PaySim CSV). This becomes the document corpus for all retrieval calls.

2. **Implement `top_k_typologies(query, k=3, corpus_path=None) -> list[dict]`** in
   `src/aml_triage/retrieval.py`. The function ranks the corpus by TF-IDF similarity to the query
   and returns the top *k* results as a list of dicts, each shaped `{"id", "title", "text",
   "score"}`, sorted by score in descending order:

   - `corpus_path` defaults to `None`: when `None`, load from `"data/typologies.json"` relative to
     the process's cwd (which is `aml-triage`'s own root under every real validation command).
     When passed, use the provided path — the baked-in test passes an explicit path to the
     aml-tutor fixture.
   - All results must have score ≥ 0 (cosine similarity is non-negative), and they must be sorted
     descending.
   - If `k` is larger than the corpus size, return all documents in the corpus.

3. **Factor out three private helper functions for lesson 009 to reuse:**
   - `_load_corpus(corpus_path)`: load and parse the JSON corpus.
   - `_documents(corpus)`: convert each corpus entry to a single concatenated string:
     `"{title}. {text}"` for each entry, to be passed to TF-IDF vectorization.
   - `_tfidf_scores(query, documents)`: return the cosine similarity score of the query against
     each document, as a 1D array in the same order as `documents`. Use
     `sklearn.feature_extraction.text.TfidfVectorizer` and
     `sklearn.metrics.pairwise.cosine_similarity` — the exact sklearn functions used in the
     reference implementation.

   Lesson 009's embedding retriever will import `_load_corpus`, `_documents`, and `_tfidf_scores`
   by name and call them in a different order to avoid duplicating this TF-IDF logic.

4. **Do not commit scratch code.** The reference implementation lives in
   `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py` (or wherever your plan's scratch
   root points). Only the spec and the baked-in test in this tutorial's git history. The learner
   writes their own `retrieval.py` inside their `aml-triage` repo, which is never committed to
   this one.

### If you ask the tutor to do this step for you

No shell command is needed — scikit-learn is already a dependency, and there is nothing to
download (unlike lesson 002's PaySim CSV). Unlike the CSV, `typologies.json` has no licensing or
download friction, so a doer can place both `aml-triage/data/typologies.json` and write
`src/aml_triage/retrieval.py` by hand, without any `uv add` or external tool invocation.

## Checks

Ask the learner to answer these in their own words before moving to lesson 009:

- Why would this exact approach (TF-IDF + cosine similarity) stop working well once the typology
  library grows to thousands of documents, and what's the first thing you'd swap in?
- The query `"many small transfers just under a reporting threshold"` should rank
  `TY-001` (Structuring / smurfing) first. Explain why — what words in the query match words in
  the typology's title or text?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/008_test_retrieval.py
```

All four tests must pass:

- `test_returns_k_results_sorted_by_descending_score`: confirms `k` results are returned and
  sorted by score descending.
- `test_result_shape_matches_the_corpus_entries`: confirms each result has the right keys and
  format.
- `test_structuring_query_ranks_the_structuring_typology_first`: confirms semantic matching works
  — a query about structuring ranks the structuring typology first.
- `test_k_larger_than_corpus_returns_the_whole_corpus`: confirms edge case handling.

```json validation
[
  {
    "id": "008-typology-retrieval",
    "label": "Typology retrieval",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/008_test_retrieval.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Lesson 010 will validate any triage decisions the LLM makes by checking that every cited
typology ID was actually one of the top *k* results shown to the LLM. If `k` is too small to
include the typology that actually matches a transaction, a well-formed decision citing that
typology will be rejected — not because the decision was wrong, but because the model could only
cite what it was shown. That dependency enforces that the `k` choice here has consequences.
