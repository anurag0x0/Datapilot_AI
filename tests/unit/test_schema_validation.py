
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.datapilot.schema_validation import validate_schema


SUPPORTED_TYPES = [
    "Do not check",
    "string",
    "integer",
    "numeric",
    "date",
    "boolean",
]


def render_schema_validation(df: pd.DataFrame) -> None:
    st.markdown("### Schema & Type Validation")
    st.caption(
        "Validate required columns, expected types and business rules "
        "without modifying your working dataset."
    )

    if df is None or df.empty:
        st.info("Upload a dataset containing at least one row.")
        return

    columns = list(df.columns)

    with st.form("schema_validation_form"):
        required_text = st.text_input(
            "Required columns (comma-separated)",
            placeholder="customer_id, age, email",
        )

        required_columns = [
            name.strip()
            for name in required_text.split(",")
            if name.strip()
        ]

        st.markdown("#### Expected Data Types")
        expected_types = {}
        type_columns = st.columns(2)

        for i, column in enumerate(columns):
            with type_columns[i % 2]:
                selected_type = st.selectbox(
                    f"Expected type: {column}",
                    SUPPORTED_TYPES,
                    key=f"schema_type_{i}",
                )

            if selected_type != "Do not check":
                expected_types[column] = selected_type

        st.markdown("#### Additional Rules")
        rule_columns = st.multiselect(
            "Select columns for extra rules",
            columns,
        )

        rules = {}

        for i, column in enumerate(rule_columns):
            with st.expander(f"Rules for {column}"):
                rule = {}

                if st.checkbox(
                    "Values must not be missing",
                    key=f"schema_not_null_{i}",
                ):
                    rule["not_null"] = True

                use_min = st.checkbox(
                    "Set minimum value",
                    key=f"schema_use_min_{i}",
                )
                if use_min:
                    rule["min"] = st.number_input(
                        f"Minimum for {column}",
                        value=0.0,
                        key=f"schema_min_{i}",
                    )

                use_max = st.checkbox(
                    "Set maximum value",
                    key=f"schema_use_max_{i}",
                )
                if use_max:
                    rule["max"] = st.number_input(
                        f"Maximum for {column}",
                        value=100.0,
                        key=f"schema_max_{i}",
                    )

                use_allowed = st.checkbox(
                    "Restrict allowed values",
                    key=f"schema_use_allowed_{i}",
                )
                if use_allowed:
                    allowed_text = st.text_input(
                        f"Allowed values for {column}",
                        placeholder="Active, Inactive, Pending",
                        key=f"schema_allowed_{i}",
                    )
                    rule["allowed_values"] = [
                        value.strip()
                        for value in allowed_text.split(",")
                        if value.strip()
                    ]

                if rule:
                    rules[column] = rule

        submitted = st.form_submit_button(
            "Run Schema Validation",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            started = time.perf_counter()

            result = validate_schema(
                df=df,
                required_columns=required_columns,
                expected_types=expected_types,
                rules=rules,
            )

            st.session_state["schema_validation_result"] = result
            st.session_state["schema_validation_duration"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop("schema_validation_result", None)
            st.error(f"Validation configuration error: {exc}")
            return

        except Exception:
            st.session_state.pop("schema_validation_result", None)
            st.error("Schema validation failed. Check the application logs.")
            return

    result = st.session_state.get("schema_validation_result")

    if result is None:
        st.info("Configure rules and click Run Schema Validation.")
        return

    st.markdown("---")
    st.markdown("#### Validation Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Checked", result["total_rows"])
    c2.metric("Columns Checked", result["total_columns"])
    c3.metric("Errors", result["error_count"])
    c4.metric("Warnings", result["warning_count"])

    st.caption(
        f"Execution time: "
        f"{st.session_state.get('schema_validation_duration', 0)} ms"
    )

    if result["valid"]:
        st.success("No validation errors found for the selected rules.")
    else:
        st.error(f"Found {result['error_count']} validation error(s).")

    issues = result["issues"]

    if issues.empty:
        st.success("No issues to report.")
        return

    issue_filter = st.selectbox(
        "Filter issues",
        ["All issues", "Errors only", "Warnings only"],
        key="schema_issue_filter",
    )

    visible_issues = issues.copy()

    if issue_filter == "Errors only":
        visible_issues = visible_issues[
            visible_issues["severity"] == "ERROR"
        ]
    elif issue_filter == "Warnings only":
        visible_issues = visible_issues[
            visible_issues["severity"] == "WARNING"
        ]

    st.dataframe(
        visible_issues,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Validation Report (CSV)",
        data=visible_issues.to_csv(index=False).encode("utf-8-sig"),
        file_name="datapilot_schema_validation_report.csv",
        mime="text/csv",
        key="download_schema_validation_report",
    )

    st.info(
        "Validation only reports issues; it does not automatically "
        "modify your dataset."
    )
