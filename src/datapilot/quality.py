
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


SEVERITY_WEIGHTS = {
    "High": 30.0,
    "Medium": 15.0,
    "Low": 5.0,
}


def _blank_mask(series: pd.Series) -> pd.Series:
    """Treat null values and whitespace-only strings as missing."""
    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        mask = mask | series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )

    return mask.fillna(True).astype(bool)


def _validate_columns(
    df: pd.DataFrame,
    configured_fields: Sequence[Any],
    setting_name: str,
) -> None:
    unknown = [field for field in configured_fields if field not in df.columns]

    if unknown:
        raise ValueError(
            f"{setting_name} contains unknown column(s): {unknown}"
        )


def audit_dataframe(
    df: pd.DataFrame,
    *,
    required_fields: Sequence[Any] = (),
    unique_fields: Sequence[Any] = (),
    numeric_fields: Sequence[Any] = (),
    allowed_values: Mapping[Any, Sequence[Any]] | None = None,
    numeric_ranges: Mapping[Any, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    """
    Profile and validate a DataFrame without modifying the input.

    Quality score:
        Starts at 100.
        Each rule deducts severity_weight * affected_rows / total_rows.
        The final score is clamped between 0 and 100.

    This is a transparent rule-based indicator, not a statistical
    guarantee that a dataset is correct.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if len(df.columns) == 0:
        raise ValueError("The dataset must contain at least one column.")

    if df.columns.has_duplicates:
        raise ValueError("Duplicate column names are not supported.")

    required_fields = tuple(required_fields)
    unique_fields = tuple(unique_fields)
    numeric_fields = tuple(numeric_fields)
    allowed_values = allowed_values or {}
    numeric_ranges = numeric_ranges or {}

    _validate_columns(df, required_fields, "required_fields")
    _validate_columns(df, unique_fields, "unique_fields")
    _validate_columns(df, numeric_fields, "numeric_fields")
    _validate_columns(df, tuple(allowed_values), "allowed_values")
    _validate_columns(df, tuple(numeric_ranges), "numeric_ranges")

    for col, bounds in numeric_ranges.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f"numeric_ranges[{col!r}] must be a (minimum, maximum) tuple."
            )

        minimum, maximum = bounds

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"numeric_ranges[{col!r}] minimum exceeds maximum."
            )

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    findings: list[dict[str, Any]] = []

    def add_finding(
        *,
        rule: str,
        severity: str,
        column: Any,
        mask: pd.Series,
        description: str,
    ) -> None:
        affected_rows = int(mask.fillna(False).astype(bool).sum())

        if affected_rows == 0:
            return

        weight = SEVERITY_WEIGHTS[severity]
        rate = affected_rows / total_rows if total_rows else 0.0

        findings.append({
            "Rule": rule,
            "Severity": severity,
            "Column": str(column),
            "Affected Rows": affected_rows,
            "Affected (%)": round(rate * 100, 2),
            "Penalty": round(weight * rate, 2),
            "Description": description,
        })

    # 1. Missing values, including blank strings.
    blank_masks = {
        col: _blank_mask(df[col])
        for col in df.columns
    }

    for col, mask in blank_masks.items():
        add_finding(
            rule="Missing values",
            severity="High" if col in required_fields else "Medium",
            column=col,
            mask=mask,
            description="Values are null or contain only whitespace.",
        )

    # 2. Required field checks are explicitly reported as their own rule.
    for col in required_fields:
        add_finding(
            rule="Required field",
            severity="High",
            column=col,
            mask=blank_masks[col],
            description="This field is required but has a missing value.",
        )

    # 3. Exact duplicate records.
    if total_rows:
        duplicate_mask = df.duplicated(keep=False)

        add_finding(
            rule="Duplicate rows",
            severity="Medium",
            column="(entire row)",
            mask=duplicate_mask,
            description=(
                "Rows repeat across all columns; each member of a "
                "duplicate group is counted."
            ),
        )

    # 4. Uniqueness of configured business keys.
    for col in unique_fields:
        valid = ~blank_masks[col]
        duplicate_mask = pd.Series(False, index=df.index)

        duplicate_mask.loc[valid] = df.loc[valid, col].duplicated(
            keep=False
        )

        add_finding(
            rule="Unique field",
            severity="High",
            column=col,
            mask=duplicate_mask,
            description=(
                "Non-missing values repeat in a field configured as unique."
            ),
        )

    # 5. Numeric parsing checks.
    for col in numeric_fields:
        values = df[col]
        parsed = pd.to_numeric(values, errors="coerce")

        invalid_mask = values.notna() & parsed.isna()

        add_finding(
            rule="Numeric validity",
            severity="High",
            column=col,
            mask=invalid_mask,
            description=(
                "Non-missing values cannot be interpreted as numbers."
            ),
        )

    # 6. Allowed-value rules.
    for col, allowed in allowed_values.items():
        allowed_set = set(allowed)
        missing_mask = blank_masks[col]

        invalid_mask = (
            ~missing_mask
            & ~df[col].isin(allowed_set)
        )

        add_finding(
            rule="Allowed values",
            severity="Medium",
            column=col,
            mask=invalid_mask,
            description=f"Value is not in the configured allowed set: {list(allowed)}.",
        )

    # 7. Numeric range rules.
    for col, (minimum, maximum) in numeric_ranges.items():
        values = df[col]
        parsed = pd.to_numeric(values, errors="coerce")

        # Non-numeric values are already reported by numeric validation
        # if the column is also configured in numeric_fields.
        valid_numeric = values.notna() & parsed.notna()
        out_of_range = pd.Series(False, index=df.index)

        if minimum is not None:
            out_of_range |= valid_numeric & (parsed < minimum)

        if maximum is not None:
            out_of_range |= valid_numeric & (parsed > maximum)

        add_finding(
            rule="Numeric range",
            severity="Medium",
            column=col,
            mask=out_of_range,
            description=f"Numeric value is outside the configured range [{minimum}, {maximum}].",
        )

    findings_df = pd.DataFrame(
        findings,
        columns=[
            "Rule",
            "Severity",
            "Column",
            "Affected Rows",
            "Affected (%)",
            "Penalty",
            "Description",
        ],
    )

    # Transparent, rule-based score. Findings can overlap, so penalties
    # are intentionally accumulated per configured rule.
    total_penalty = float(findings_df["Penalty"].sum()) if not findings_df.empty else 0.0
    score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 2)

    if score >= 95:
        grade = "Excellent"
    elif score >= 85:
        grade = "Good"
    elif score >= 70:
        grade = "Needs attention"
    else:
        grade = "Poor"

    severity_counts = {
        severity: (
            int((findings_df["Severity"] == severity).sum())
            if not findings_df.empty else 0
        )
        for severity in ("High", "Medium", "Low")
    }

    column_profile = []

    for col in df.columns:
        missing_count = int(blank_masks[col].sum())

        column_profile.append({
            "Column": str(col),
            "Data Type": str(df[col].dtype),
            "Rows": total_rows,
            "Missing": missing_count,
            "Missing (%)": round(
                missing_count * 100 / total_rows, 2
            ) if total_rows else 0.0,
            "Unique Values": int(df[col].nunique(dropna=True)),
        })

    return {
        "quality_score": score,
        "quality_grade": grade,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_cells": total_cells,
        "missing_cells": int(sum(mask.sum() for mask in blank_masks.values())),
        "duplicate_rows": int(df.duplicated(keep=False).sum()),
        "findings_count": len(findings_df),
        "severity_counts": severity_counts,
        "findings": findings_df,
        "column_profile": pd.DataFrame(column_profile),
    }
