"""Baked-in check for lesson 003 — time-based split."""


def test_split_has_no_overlapping_steps(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    assert set(train_df["step"]).isdisjoint(set(test_df["step"]))


def test_split_is_chronologically_ordered_and_complete(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    assert train_df["step"].max() <= split_step
    assert test_df["step"].min() > split_step
    assert len(train_df) + len(test_df) == len(df)


def test_boundary_step_lands_entirely_in_train(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    boundary_rows_in_raw = (df["step"] == split_step).sum()
    boundary_rows_in_train = (train_df["step"] == split_step).sum()
    assert boundary_rows_in_raw > 0, "split_step does not match any row in the fixture"
    assert boundary_rows_in_train == boundary_rows_in_raw
    assert (test_df["step"] == split_step).sum() == 0
