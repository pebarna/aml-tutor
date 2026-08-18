"""Baked-in check for lesson 006 — train the baseline."""


def _prepare_splits(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    train_featured = add_features(train_df)
    test_featured = add_features(test_df)
    y_train = train_featured.pop("isFraud")
    y_test = test_featured.pop("isFraud")
    return train_featured, y_train, test_featured, y_test


def test_model_fits_and_predicts_valid_probabilities(fixture_path, split_step):
    from aml_triage.imbalance import compute_scale_pos_weight
    from aml_triage.model import train_baseline

    X_train, y_train, X_test, y_test = _prepare_splits(fixture_path, split_step)
    weight = compute_scale_pos_weight(y_train)
    model = train_baseline(X_train, y_train, weight)
    scores = model.predict_proba(X_test)[:, 1]
    assert len(scores) == len(X_test)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_model_is_at_least_somewhat_better_than_chance(fixture_path, split_step):
    from sklearn.metrics import roc_auc_score

    from aml_triage.imbalance import compute_scale_pos_weight
    from aml_triage.model import train_baseline

    X_train, y_train, X_test, y_test = _prepare_splits(fixture_path, split_step)
    weight = compute_scale_pos_weight(y_train)
    model = train_baseline(X_train, y_train, weight)
    scores = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, scores) > 0.5
