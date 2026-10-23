
from __future__ import annotations

from typing import Any

import pandas as pd


def _is_missing(value: Any) -> bool:
    """Safely check whether a scalar value is missing."""
    result = pd.isna(value)
    return bool(result) if not hasattr(result, "__len__") else False


def _same_value(left: Any, right: Any) -> bool:
    """Treat two missing values as equal."""
    if _is_missing(left) and _is_missing(right):
        return True

    if _is_missing(left) or _is_missing(right):
        return False

    try:
        result = left == right
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def compare_dataframes(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compare original and cleaned DataFrames.

    Returns:
        summary: overall before/after impact metrics
        column_summary: per-column change and missing-value metrics
        cell_changes: cell-level old/new values for common rows/columns

    Assumption:
        Rows are matched by their DataFrame index, and columns by name.
        If cleaning resets or changes indexes, row-level matching may
        be inaccurate. Preserve the original index for reliable comparison.
    """
    if not isinstance(before, pd.DataFrame):
        raise TypeError("'before' must be a pandas DataFrame.")

    if not isinstance(after, pd.DataFrame):
        raise TypeError("'after' must be a pandas DataFrame.")

    if before.columns.has_duplicates or after.columns.has_duplicates:
        raise ValueError("DataFrames must have unique column names.")

    before_copy = before.copy(deep=True)
    after_copy = after.copy(deep=True)

    before_columns = list(before_copy.columns)
    after_columns = list(after_copy.columns)

    common_columns = [
        column for column in before_columns if column in after_columns
    ]
    added_columns = [column for column in after_columns if column not in before_columns]
    removed_columns = [column for column in before_columns if column not in after_columns]

    before_index = list(before_copy.index)
    after_index = list(after_copy.index)

    common_index = [
        index for index in before_index
        if index in after_index
    ]
    added_indexes = [index for index in after_index if index not in before_index]
    removed_indexes = [index for index in before_index if index not in after_index]

    # Cell-by-cell comparison for rows and columns present in both versions.
    cell_changes: list[dict[str, Any]] = []

    for index in common_index:
        for column in common_columns:
            old_value = before_copy.at[index, column]
            new_value = after_copy.at[index, column]

            if not _same_value(old_value, new_value):
                cell_changes.append(
                    {
                        "Row Index": index,
                        "Column": column,
                        "Before": old_value,
                        "After": new_value,
                    }
                )

    cell_changes_df = pd.DataFrame(
        cell_changes,
        columns=["Row Index", "Column", "Before", "After"],
    )

    # Missing-value counts across each complete dataset.
    missing_before = int(before_copy.isna().sum().sum())
    missing_after = int(after_copy.isna().sum().sum())

    # Per-column summary. Columns removed by cleaning are also included.
    column_summary: list[dict[str, Any]] = []

    for column in before_columns + [
        c for c in after_columns if c not in before_columns
    ]:
        in_before = column in before_columns
        in_after = column in after_columns

        old_series = before_copy[column] if in_before else pd.Series(dtype=object)
        new_series = after_copy[column] if in_after else pd.Series(dtype=object)

        changed_count = (
            int((cell_changes_df["Column"] == column).sum())
            if not cell_changes_df.empty
            else 0
        )

        column_summary.append(
            {
                "Column": column,
                "Present Before": in_before,
                "Present After": in_after,
                "Changed Cells": changed_count,
                "Missing Before": int(old_series.isna().sum()),
                "Missing After": int(new_series.isna().sum()),
            }
        )

    column_summary_df = pd.DataFrame(column_summary)

    summary = {
        "rows_before": int(len(before_copy)),
        "rows_after": int(len(after_copy)),
        "rows_removed": len(removed_indexes),
        "rows_added": len(added_indexes),
        "columns_before": len(before_columns),
        "columns_after": len(after_columns),
        "columns_removed": len(removed_columns),
        "columns_added": len(added_columns),
        "cells_changed": len(cell_changes),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_values_reduced": missing_before - missing_after,
        "common_rows_compared": len(common_index),
        "common_columns_compared": len(common_columns),
        "removed_column_names": removed_columns,
        "added_column_names": added_columns,
    }

    return {
        "summary": summary,
        "column_summary": column_summary_df,
        "cell_changes": cell_changes_df,
    }
