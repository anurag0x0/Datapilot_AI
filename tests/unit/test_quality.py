
import pandas as pd
import pytest

from src.datapilot.quality import audit_dataframe


def test_detects_missing_values_and_required_fields():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Aman", " ", "Riya"],
    })

    result = audit_dataframe(
        df,
        required_fields=["name"],
    )

    findings = result["findings"]

    assert result["missing_cells"] == 1

    required = findings[findings["Rule"] == "Required field"]
    assert len(required) == 1
    assert required.iloc[0]["Affected Rows"] == 1
    assert required.iloc[0]["Severity"] == "High"


def test_detects_duplicate_rows():
    df = pd.DataFrame({
        "id": [1, 2, 2],
        "city": ["Delhi", "Agra", "Agra"],
    })

    result = audit_dataframe(df)

    assert result["duplicate_rows"] == 2

    duplicate_findings = result["findings"][
        result["findings"]["Rule"] == "Duplicate rows"
    ]

    assert duplicate_findings.iloc[0]["Affected Rows"] == 2


def test_detects_duplicate_business_keys():
    df = pd.DataFrame({
        "employee_id": [101, 102, 101, None],
    })

    result = audit_dataframe(
        df,
        unique_fields=["employee_id"],
    )

    finding = result["findings"][
        result["findings"]["Rule"] == "Unique field"
    ].iloc[0]

    assert finding["Affected Rows"] == 2


def test_detects_invalid_numeric_values():
    df = pd.DataFrame({
        "revenue": ["100", "unknown", "250", None],
    })

    result = audit_dataframe(
        df,
        numeric_fields=["revenue"],
    )

    finding = result["findings"][
        result["findings"]["Rule"] == "Numeric validity"
    ].iloc[0]

    assert finding["Affected Rows"] == 1


def test_detects_allowed_value_violations():
    df = pd.DataFrame({
        "status": ["Active", "Inactive", "Unknown"],
    })

    result = audit_dataframe(
        df,
        allowed_values={"status": ["Active", "Inactive"]},
    )

    finding = result["findings"][
        result["findings"]["Rule"] == "Allowed values"
    ].iloc[0]

    assert finding["Affected Rows"] == 1


def test_detects_numeric_range_violations():
    df = pd.DataFrame({
        "age": [20, 35, 150, -2],
    })

    result = audit_dataframe(
        df,
        numeric_ranges={"age": (0, 120)},
    )

    finding = result["findings"][
        result["findings"]["Rule"] == "Numeric range"
    ].iloc[0]

    assert finding["Affected Rows"] == 2


def test_score_is_bounded_and_grade_is_returned():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    })

    result = audit_dataframe(df)

    assert 0 <= result["quality_score"] <= 100
    assert result["quality_grade"] in {
        "Excellent", "Good", "Needs attention", "Poor"
    }


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame({
        "id": [1, 2],
        "name": [" Aman ", None],
    })
    original = df.copy(deep=True)

    audit_dataframe(
        df,
        required_fields=["name"],
        numeric_fields=["id"],
    )

    pd.testing.assert_frame_equal(df, original)


def test_unknown_configured_column_raises_error():
    df = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError, match="unknown column"):
        audit_dataframe(df, required_fields=["missing_column"])


def test_duplicate_column_names_are_rejected():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["id", "id"],
    )

    with pytest.raises(ValueError, match="Duplicate column names"):
        audit_dataframe(df)


def test_empty_dataframe_with_columns_is_supported():
    df = pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "name": pd.Series(dtype="object"),
    })

    result = audit_dataframe(df)

    assert result["total_rows"] == 0
    assert result["total_columns"] == 2
    assert result["quality_score"] == 100.0
    assert result["findings"].empty
