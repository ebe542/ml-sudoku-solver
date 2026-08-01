import pytest

from sudoku_ml.analysis.analyze_feature_importance import (
    FEATURE_GROUPS,
    get_feature_name
)


def test_grid_feature_name() -> None:
    assert get_feature_name(0) == "grid_r1_c1"
    assert get_feature_name(80) == "grid_r9_c9"


def test_target_position_feature_name() -> None:
    assert get_feature_name(81) == "target_cell_index"


def test_candidate_feature_names() -> None:
    assert get_feature_name(82) == "candidate_1"
    assert get_feature_name(90) == "candidate_9"


def test_row_interaction_feature_names() -> None:
    assert get_feature_name(91) == "row_candidate_frequency_1"
    assert get_feature_name(99) == "row_candidate_frequency_9"


def test_column_interaction_feature_names() -> None:
    assert get_feature_name(100) == "column_candidate_frequency_1"
    assert get_feature_name(108) == "column_candidate_frequency_9"


def test_block_interaction_feature_names() -> None:
    assert get_feature_name(109) == "block_candidate_frequency_1"
    assert get_feature_name(117) == "block_candidate_frequency_9"


def test_unknown_feature_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_feature_name(118)
