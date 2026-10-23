
from __future__ import annotations

import re
from typing import Any

import pandas as pd


SUPPORTED_TYPES = {"string", "integer", "numeric", "date", "boolean"}


def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expected_types: dict[str, str] | None = None,
    rules: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Validate a DataFrame without modifying the original data.

    Supported expected types:
        string, integer, numeric, date, boolean

    Supported rules per column:
        not_null=True
        min=<number>
        max=<number>
        regex=<pattern>
        allowed_values=[...]
    """
    required_columns = required_columns or []
    expected_types = expected_types or {}
    rules = rules or {}

    issues: list[dict[str, Any]] = []

    def add_issue(
        column: str,
        row_index: Any,
        value: Any,
        rule: str,
        message: str,
        severity: str = "ERROR",
    ) -> None:
        issues.append(
            {
                "row_index": row_index,
                "column": column,
                "value": repr(value) if value is not None else None,
                "rule": rule,
                "severity": severity,
                "message": message,
            }
        )

    # 1. Check required columns.
    for column in required_columns:
        if column not in df.columns:
            add_issue(
                column=column,
                row_index=None,
                value=None,
                rule="required_column",
                message=f"Required column '{column}' is missing.",
            )

    # 2. Check expected data types.
    for column, expected_type in expected_types.items():
        expected_type = expected_type.lower().strip()

        if expected_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported type '{expected_type}' for '{column}'. "
                f"Choose from: {sorted(SUPPORTED_TYPES)}"
            )

        if column not in df.columns:
            continue

        series = df[column]

        if expected_type == "string":
            invalid = series.notna() & ~series.map(
                lambda value: isinstance(value, str)
            )

        elif expected_type == "numeric":
            converted = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & converted.isna()

        elif expected_type == "integer":
            converted = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & (
                converted.isna()
                | (converted.notna() & (converted % 1 != 0))
            )

        elif expected_type == "date":
            converted = pd.to_datetime(series, errors="coerce")
            invalid = series.notna() & converted.isna()

        else:  # boolean
            valid_values = {True, False, 0, 1, "true", "false", "0", "1"}
            invalid = series.notna() & ~series.map(
                lambda value: (
                    value in valid_values
                    if isinstance(value, (str, int, bool))
                    else False
                )
            )

        for row_index in df.index[invalid]:
            add_issue(
                column,
                row_index,
                series.loc[row_index],
                "data_type",
                f"Expected {expected_type} value.",
            )

    # 3. Apply optional column rules.
    for column, column_rules in rules.items():
        if column not in df.columns:
            add_issue(
                column,
                None,
                None,
                "rule_column",
                f"Rules refer to missing column '{column}'.",
                severity="WARNING",
            )
            continue

        series = df[column]

        if column_rules.get("not_null", False):
            for row_index in df.index[series.isna()]:
                add_issue(
                    column,
                    row_index,
                    None,
                    "not_null",
                    "Value is required but missing.",
                )

        for bound in ("min", "max"):
            if bound not in column_rules:
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            invalid_numeric = series.notna() & numeric.isna()

            for row_index in df.index[invalid_numeric]:
                add_issue(
                    column,
                    row_index,
                    series.loc[row_index],
                    bound,
                    f"Cannot check {bound}: value is not numeric.",
                )

            valid_numeric = numeric.notna()

            if bound == "min":
                mask = valid_numeric & (numeric < column_rules[bound])
                message = f"Value is below minimum {column_rules[bound]}."
            else:
                mask = valid_numeric & (numeric > column_rules[bound])
                message = f"Value exceeds maximum {column_rules[bound]}."

            for row_index in df.index[mask]:
                add_issue(
                    column,
                    row_index,
                    series.loc[row_index],
                    bound,
                    message,
                )

        if "regex" in column_rules:
            pattern = column_rules["regex"]
            for row_index, value in series.items():
                if pd.isna(value):
                    continue
                if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
                    add_issue(
                        column,
                        row_index,
                        value,
                        "regex",
                        "Value does not match the required format.",
                    )

        if "allowed_values" in column_rules:
            allowed = column_rules["allowed_values"]
            invalid = series.notna() & ~series.isin(allowed)

            for row_index in df.index[invalid]:
                add_issue(
                    column,
                    row_index,
                    series.loc[row_index],
                    "allowed_values",
                    f"Value must be one of: {allowed}.",
                )

    issue_df = pd.DataFrame(
        issues,
        columns=[
            "row_index",
            "column",
            "value",
            "rule",
            "severity",
            "message",
        ],
    )

    error_count = int((issue_df["severity"] == "ERROR").sum())
    warning_count = int((issue_df["severity"] == "WARNING").sum())

    return {
        "valid": error_count == 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issue_df,
    }
