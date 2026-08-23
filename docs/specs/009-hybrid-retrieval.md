# Hybrid retrieval

Combine the keyword (TF-IDF) retriever from lesson 008 with learned semantic embeddings to rank a
corpus by relevance. A hybrid approach beats either method alone for compliance-flavored text:
exact terms and thresholds favor keyword matching, while paraphrase and conceptual similarity
favor embeddings.

## Key concept

**Keyword matching alone misses paraphrases; embeddings alone miss exact compliance terms.**
Lesson 008 ranked documents using TF-IDF + cosine similarity, which excels at exact-match
retrieval — the query "structuring" matches documents containing "structuring," and the math
never confuses the two. But a query phrased as "many small transfers to avoid detection" might
miss a document whose exact title is "Structuring / smurfing" if the document's text doesn't
repeat those exact keywords enough times. An embedding model — learned from millions of sentences
to recognize semantic similarity — would catch that paraphrase, but it can also hallucinate
matches that aren't there: if a query is about "avoiding currency controls" and the corpus
includes a document about "exchanging currency," an embedding model might rank them as similar
even though they are not the same compliance risk at all.

**Hybrid retrieval blends both.** A single score combines TF-IDF (exact keyword weight) and
embedding similarity (semantic weight) using a tunable blend parameter `alpha`:
```
blended_score = alpha * tfidf_score + (1 - alpha) * embedding_score
```

When `alpha=1.0`, the embedding term vanishes entirely and blending produces pure TF-IDF —
useful for testing determinism independent of model version. When `alpha=0.0`, the TF-IDF term
vanishes and blending produces pure embeddings. When `alpha=0.5`, both contribute equally. A
real deployment would tune `alpha` on labeled retrieval-quality data; this six-document teaching
corpus is too small to support meaningful tuning, so the tests use `alpha=1.0` (deterministic)
or assert only soft structural properties rather than exact rankings for other values.

**At a regulated shop:** the embedding half would come from a model already inside an approved
enterprise contract (Azure OpenAI / Bedrock / Vertex embeddings) or self-hosted inside the
bank's own VPC when the text is AML / SAR-adjacent and cannot leave the compliance boundary to
a new subprocessor at all — this lesson's local, open-weights model (`sentence-transformers`) is
the version of that same idea that requires no vendor relationship at all. The keyword half of
a hybrid pipeline typically runs on OpenSearch / Elasticsearch or Postgres+pgvector — audited
infrastructure a compliance team already knows how to monitor and retain logs from.

## Implementation order

1. **Add the `sentence-transformers` dependency:** Run `uv add sentence-transformers` in the
   `aml-triage` directory. This brings in a pre-trained embedding model, `all-MiniLM-L6-v2`,
   which is downloaded (one time, ~90MB from HuggingFace) the first time you call
   `SentenceTransformer("all-MiniLM-L6-v2")`.

2. **Implement `top_k_typologies_hybrid(query, k=3, corpus_path=None, alpha=0.5) -> list[dict]`**
   in `src/aml_triage/retrieval.py`, reusing `_load_corpus`, `_documents`, and `_tfidf_scores`
   from lesson 008 rather than duplicating them:
   - Compute TF-IDF scores for the query using the existing `_tfidf_scores` helper.
   - Compute embedding scores for the query and all documents using a new `_embedding_scores`
     helper (see below).
   - Blend the two score arrays: `blended = alpha * tfidf + (1 - alpha) * embedding`.
   - Rank by blended score descending and return the top *k* results in the same shape as
     `top_k_typologies`: `{"id", "title", "text", "score"}`.

3. **Implement three helper functions:**
   - `_get_embedding_model()`: Lazy-load the `SentenceTransformer` once into a module-level
     global `_EMBEDDING_MODEL`. The first call initializes and downloads the model; subsequent
     calls return the cached model. This is the pattern lesson 006's `get_model` already
     established for sklearn's `XGBClassifier`.
   - `_embedding_scores(query, documents)`: Encode the query and all documents using the model,
     then compute cosine similarity between the query embedding and all document embeddings,
     returning a 1D array of similarity scores in the same order as `documents`. Use
     `sklearn.metrics.pairwise.cosine_similarity` (already imported for TF-IDF).
   - The existing `_load_corpus`, `_documents`, and `_tfidf_scores` helpers carry over unchanged.

4. **Do not commit scratch code.** The reference implementation lives in
   `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py` (or wherever your plan's scratch
   root points). Only the spec and the baked-in test in this tutorial's git history. The learner
   writes their own `retrieval.py` inside their `aml-triage` repo, which is never committed to
   this one.

### If you ask the tutor to do this step for you

The doer cannot run `uv add sentence-transformers` itself (shell access needed) or trigger the
first-run model download (network access needed). Once `sentence-transformers` is installed by
the instructor or by a learner-run setup step, the doer can write the code in
`src/aml_triage/retrieval.py` by hand — `_get_embedding_model`, `_embedding_scores`, and the
blending logic in `top_k_typologies_hybrid` are all deterministic functions with no runtime
dependencies outside what Python + sklearn + sentence-transformers provide.

## Checks

Ask the learner to answer these in their own words before moving to lesson 010:

- When would you raise `alpha` toward 1.0 (emphasizing exact-match TF-IDF) versus lower it toward
  0.0 (emphasizing semantic embeddings) for this specific corpus? Describe a scenario where each
  choice makes sense.
- `alpha=1.0` produces exactly the same ranking as lesson 008's pure TF-IDF retriever. Prove this
  is true by running the test yourself, then explain the arithmetic: how does the blending formula
  ensure this property?
- What happens to the embedding model's initialization the first time `_embedding_scores` is
  called with real data — and why does lazy initialization matter for a system that might
  initialize many times during testing?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/009_test_hybrid_retrieval.py
```

All four tests must pass:

- `test_alpha_1_matches_pure_tfidf_ranking_exactly`: confirms `alpha=1.0` produces exactly the
  same ranking as the pure TF-IDF function — this case is deterministic, independent of
  embedding model version or random seeds.
- `test_returns_k_results_with_expected_shape`: confirms the result shape matches lesson 008's
  shape, and `k` results are returned.
- `test_structuring_query_ranks_the_structuring_typology_in_the_top_two`: confirms that a query
  about structuring ranks the structuring typology in the top 2. This is a soft assertion
  (top 2, not exactly first) because embedding scores can shift slightly across model versions,
  but the semantic signal should rank this typology highly nonetheless.
- `test_alpha_0_still_returns_a_valid_shape`: confirms `alpha=0.0` (pure embeddings) still
  returns `k` valid results with the expected shape.

```json validation
[
  {
    "id": "009-hybrid-retrieval",
    "label": "Hybrid retrieval",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/009_test_hybrid_retrieval.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Today's test proves the blending arithmetic is correct (via the `alpha=1.0` exact-match case), not
that 0.5 is the right weight for this corpus — a real deployment would tune `alpha` against
labeled retrieval-quality data, which this six-document teaching corpus is too small to support
meaningfully. The `alpha=0.0` and `alpha=0.5` cases assert only structural properties (result
shape and count) rather than exact rankings, because embedding scores vary slightly across
`sentence-transformers` versions and model weights. The semantic signal — that a query about
structuring should rank a structuring-related typology highly — is strong enough to survive
model version shifts, but the exact top-3 order might change.
