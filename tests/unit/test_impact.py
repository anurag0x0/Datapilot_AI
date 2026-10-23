
import pandas as pd
import pytest

from src.datapilot.impact import compare_dataframes


def test_detects_changed_cells():
    before = pd.DataFrame({"name": ["A", "B"], "score": [10, 20]})
    after = pd.DataFrame({"name": ["A", "B"], "score": [10, 25]})

    result = compare_dataframes(before, after)

    assert result["summary"]["cells_changed"] == 1
    assert result["cell_changes"].iloc[0]["Column"] == "score"
    assert result["cell_changes"].iloc[0]["Before"] == 20
    assert result["cell_changes"].iloc[0]["After"] == 25


def test_detects_missing_values_reduced():
    before = pd.DataFrame({"name": ["A", None, "C"]})
    after = pd.DataFrame({"name": ["A", "B", "C"]})

    result = compare_dataframes(before, after)

    assert result["summary"]["missing_before"] == 1
    assert result["summary"]["missing_after"] == 0
    assert result["summary"]["missing_values_reduced"] == 1


def test_detects_removed_rows_by_index():
    before = pd.DataFrame({"value": [10, 20, 30]}, index=[0, 1, 2])
    after = before.drop(index=1)

    result = compare_dataframes(before, after)

    assert result["summary"]["rows_before"] == 3
    assert result["summary"]["rows_after"] == 2
    assert result["summary"]["rows_removed"] == 1


def test_detects_added_and_removed_columns():
    before = pd.DataFrame({"name": ["A"], "old": [10]})
    after = pd.DataFrame({"name": ["A"], "new": [20]})

    result = compare_dataframes(before, after)

    assert result["summary"]["columns_added"] == 1
    assert result["summary"]["columns_removed"] == 1
    assert result["summary"]["added_column_names"] == ["new"]
    assert result["summary"]["removed_column_names"] == ["old"]


def test_two_missing_values_are_not_counted_as_a_change():
    before = pd.DataFrame({"value": [None, 10]})
    after = pd.DataFrame({"value": [None, 10]})

    result = compare_dataframes(before, after)

    assert result["summary"]["cells_changed"] == 0


def test_original_dataframes_are_not_modified():
    before = pd.DataFrame({"value": [None, 10]})
    after = pd.DataFrame({"value": [0, 10]})
    before_original = before.copy(deep=True)
    after_original = after.copy(deep=True)

    compare_dataframes(before, after)

    pd.testing.assert_frame_equal(before, before_original)
    pd.testing.assert_frame_equal(after, after_original)


def test_rejects_duplicate_column_names():
    before = pd.DataFrame([[1, 2]], columns=["value", "value"])
    after = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError, match="unique column names"):
        compare_dataframes(before, after)


def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        compare_dataframes([1, 2, 3], pd.DataFrame({"value": [1]}))


def test_reports_per_column_changes():
    before = pd.DataFrame({"name": ["A"], "score": [10]})
    after = pd.DataFrame({"name": ["B"], "score": [10]})

    result = compare_dataframes(before, after)
    summary = result["column_summary"].set_index("Column")

    assert summary.loc["name", "Changed Cells"] == 1
    assert summary.loc["score", "Changed Cells"] == 0
