"""Baked-in check for lesson 009 — hybrid retrieval.

alpha=1.0 zeroes out the embedding term entirely, so that one case stays exactly deterministic
regardless of which sentence-transformers version or model weights are installed — the other
assertions are intentionally soft (structure and "in the top two"), since embedding scores
themselves can shift slightly across model versions.
"""


def test_alpha_1_matches_pure_tfidf_ranking_exactly(typologies_path):
    from aml_triage.retrieval import top_k_typologies, top_k_typologies_hybrid

    query = "many small transfers just under a reporting threshold"
    tfidf_only = top_k_typologies(query, k=3, corpus_path=str(typologies_path))
    hybrid_alpha_1 = top_k_typologies_hybrid(query, k=3, corpus_path=str(typologies_path), alpha=1.0)
    assert [r["id"] for r in tfidf_only] == [r["id"] for r in hybrid_alpha_1]


def test_returns_k_results_with_expected_shape(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid("cash withdrawal pattern", k=3, corpus_path=str(typologies_path))
    assert len(results) == 3
    assert set(results[0].keys()) == {"id", "title", "text", "score"}


def test_structuring_query_ranks_the_structuring_typology_in_the_top_two(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid(
        "many small transfers just under a reporting threshold", k=2, corpus_path=str(typologies_path)
    )
    assert "TY-001" in [r["id"] for r in results]


def test_alpha_0_still_returns_a_valid_shape(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid(
        "cash withdrawal pattern", k=6, corpus_path=str(typologies_path), alpha=0.0
    )
    assert len(results) == 6
