"""Baked-in check for lesson 008 — typology retrieval (the keyword/TF-IDF half of hybrid retrieval).

TF-IDF + cosine similarity over a fixed, six-document corpus is exact linear algebra, not a
downloaded model — so this test can assert exact structural and ordering properties.
"""


def test_returns_k_results_sorted_by_descending_score(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("large cash withdrawal pattern", k=3, corpus_path=str(typologies_path))
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_result_shape_matches_the_corpus_entries(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("structuring below a threshold", k=1, corpus_path=str(typologies_path))
    assert set(results[0].keys()) == {"id", "title", "text", "score"}
    assert results[0]["id"].startswith("TY-")


def test_structuring_query_ranks_the_structuring_typology_first(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies(
        "many small transfers just under a reporting threshold", k=1, corpus_path=str(typologies_path)
    )
    assert results[0]["id"] == "TY-001"


def test_k_larger_than_corpus_returns_the_whole_corpus(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("anything", k=100, corpus_path=str(typologies_path))
    assert len(results) == 6
