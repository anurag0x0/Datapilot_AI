from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st

from src.datapilot.impact import compare_dataframes
from src.datapilot.schema_validation import validate_schema


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "DataPilot AI"
APP_VERSION = "3.0.0"
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 500
PREVIEW_ROWS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("datapilot")

st.set_page_config(
    page_title="DataPilot AI | Data Operations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #0A1020;
        --sidebar: #0D1729;
        --panel: #121F34;
        --panel-hover: #192A43;
        --border: #344862;
        --text: #F4F7FC;
        --muted: #C0CDE0;
        --teal: #48E0C2;
        --blue: #8AB8FF;
        --green: #76E2A8;
        --amber: #FFD27A;
        --red: #FF8799;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"] {
        background: var(--bg) !important;
    }

    .block-container {
        max-width: 1600px;
        padding: 1.6rem 2rem 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    /* Main typography */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5 {
        color: #FFFFFF !important;
        font-weight: 750 !important;
    }

    .stApp p,
    .stApp li,
    .stApp label {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    .dp-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 18px;
    }

    .dp-logo {
        display: flex;
        width: 48px;
        height: 48px;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border-radius: 14px;
        background: linear-gradient(135deg, #48E0C2, #8AB8FF);
        color: #071522 !important;
        font-size: 25px;
        font-weight: 900;
    }

    .dp-brand-name {
        font-size: 21px;
        font-weight: 850;
        color: #FFFFFF !important;
    }

    .dp-brand-sub {
        font-size: 10px;
        letter-spacing: 1.7px;
        color: var(--muted) !important;
        margin-top: 3px;
    }

    .dp-eyebrow {
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 2px;
        color: var(--teal) !important;
        margin-bottom: 10px;
    }

    .dp-title {
        font-size: clamp(29px, 3vw, 40px);
        font-weight: 850;
        line-height: 1.2;
        letter-spacing: -1.3px;
        color: #FFFFFF !important;
    }

    .dp-subtitle {
        color: var(--muted) !important;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 9px;
        margin-bottom: 22px;
    }

    .dp-divider {
        height: 1px;
        background: var(--border);
        margin: 22px 0;
    }

    .dp-section-label {
        color: var(--muted) !important;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1.3px;
        margin: 12px 0 8px;
    }

    .dp-pill {
        display: inline-block;
        background: #10352F;
        border: 1px solid #286D60;
        color: var(--teal) !important;
        padding: 6px 10px;
        border-radius: 30px;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: .5px;
    }

    /* KPI cards */
    .dp-card {
        background: linear-gradient(145deg, #192A43, #121F34);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 17px;
        min-height: 115px;
        margin-bottom: 10px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, .10);
    }

    .dp-card-label {
        color: var(--muted) !important;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1px;
    }

    .dp-card-value {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 850;
        line-height: 1.3;
        margin-top: 9px;
    }

    .dp-card-note {
        color: var(--muted) !important;
        font-size: 11px;
        margin-top: 6px;
    }

    [data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 13px;
        padding: 15px !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--teal) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        padding: 12px 13px;
        border-radius: 7px 7px 0 0;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #D9E5F5 !important;
        font-weight: 650 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--teal) !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {
        color: var(--teal) !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        min-height: 42px;
        background: #223750 !important;
        color: #FFFFFF !important;
        border: 1px solid #526987 !important;
        border-radius: 9px !important;
        font-weight: 750 !important;
        transition: border-color .15s ease;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    .stFormSubmitButton > button * {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--teal) !important;
        color: #071522 !important;
        border-color: var(--teal) !important;
    }

    .stButton > button[kind="primary"] *,
    .stDownloadButton > button[kind="primary"] *,
    .stFormSubmitButton > button[kind="primary"] * {
        color: #071522 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--teal) !important;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
        opacity: .65;
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background: #111E32 !important;
        color: #FFFFFF !important;
        border: 1px solid #526987 !important;
        border-radius: 8px !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #C0CDE0 !important;
    }

    [data-baseweb="select"] > div {
        background: #142239 !important;
        border-color: #526987 !important;
    }

    [data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    [data-baseweb="popover"],
    [role="listbox"] {
        background: #17263D !important;
    }

    [data-baseweb="popover"] *,
    [role="listbox"] * {
        color: #FFFFFF !important;
    }

    /* Prominent sidebar upload box */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #17263D !important;
        border: 2px dashed var(--teal) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #17263D !important;
        border: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: var(--teal) !important;
        color: #071522 !important;
        border: 1px solid var(--teal) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        color: #071522 !important;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px;
        overflow: hidden;
        background: #F7F9FC !important;
    }

    [data-testid="stDataFrame"] iframe {
        color-scheme: light;
    }

    [data-testid="stTable"] {
        background: #F7F9FC !important;
    }

    [data-testid="stTable"] table {
        background: #F7F9FC !important;
        color: #17243A !important;
    }

    [data-testid="stTable"] th {
        background: #203550 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stTable"] td {
        background: #F7F9FC !important;
        color: #17243A !important;
    }

    /* Alerts and footer */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    .dp-footer {
        color: var(--muted) !important;
        font-size: 11px;
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .dp-card {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. REUSABLE DATA FUNCTIONS
# ============================================================

def normalize_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Convert whitespace-only strings to missing values."""
    result = df.copy(deep=True)

    for col in result.select_dtypes(
        include=["object", "string"]
    ).columns:
        result[col] = result[col].map(
            lambda value: (
                pd.NA
                if isinstance(value, str) and not value.strip()
                else value
            )
        )

    return result


def get_metrics(df: pd.DataFrame) -> dict:
    audit = normalize_blanks(df)
    rows, columns = audit.shape
    cells = rows * columns
    missing = int(audit.isna().sum().sum())

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "missing_pct": round(missing * 100 / cells, 2) if cells else 0,
        "duplicates": int(audit.duplicated().sum()),
    }


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    audit = normalize_blanks(df)
    output = []
    total_rows = len(audit)

    for col in audit.columns:
        series = audit[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        output.append({
            "Column": str(col),
            "Data Type": str(series.dtype),
            "Non-null": int(series.notna().sum()),
            "Missing": missing,
            "Missing (%)": round(missing * 100 / total_rows, 2)
                if total_rows else 0,
            "Unique Values": unique,
            "Constant Column": unique <= 1,
        })

    return pd.DataFrame(output)


def run_quality_checks(
    df: pd.DataFrame,
    required: list,
    unique_fields: list,
    numeric_fields: list,
) -> pd.DataFrame:
    audit = normalize_blanks(df)
    findings = []

    def add(rule, severity, column, count, description):
        if count:
            findings.append({
                "Rule": rule,
                "Severity": severity,
                "Column": str(column),
                "Affected Rows": int(count),
                "Description": description,
            })

    for col in required:
        add(
            "Required field",
            "High",
            col,
            int(audit[col].isna().sum()),
            "Required values are missing.",
        )

    for col in unique_fields:
        values = audit[col].dropna()
        add(
            "Uniqueness",
            "Medium",
            col,
            int(values.duplicated(keep=False).sum()),
            "Values repeat in a field marked unique.",
        )

    for col in numeric_fields:
        converted = pd.to_numeric(audit[col], errors="coerce")
        invalid = int((audit[col].notna() & converted.isna()).sum())

        add(
            "Numeric validity",
            "Medium",
            col,
            invalid,
            "Non-null values cannot be interpreted as numbers.",
        )

    add(
        "Duplicate records",
        "Medium",
        "(entire row)",
        int(audit.duplicated().sum()),
        "Rows repeat across all columns.",
    )

    return pd.DataFrame(
        findings,
        columns=[
            "Rule", "Severity", "Column", "Affected Rows", "Description"
        ],
    )


def clean_data(source: pd.DataFrame, options: dict):
    result = source.copy(deep=True)
    changes = []

    if options["trim"]:
        changed = 0

        for col in result.select_dtypes(
            include=["object", "string"]
        ).columns:
            before = result[col].copy()
            result[col] = result[col].map(
                lambda value: value.strip()
                if isinstance(value, str) else value
            )
            changed += int((before != result[col]).fillna(False).sum())

        changes.append(f"Whitespace trimming changed {changed} cell(s).")

    if options["normalize"]:
        before = int(result.isna().sum().sum())
        result = normalize_blanks(result)
        after = int(result.isna().sum().sum())

        changes.append(
            f"Blank normalization identified "
            f"{after - before} additional missing cell(s)."
        )

    if options["deduplicate"]:
        before = len(result)
        result = result.drop_duplicates().copy()
        changes.append(f"Removed {before - len(result)} duplicate row(s).")

    if options["drop_empty"]:
        empty = [col for col in result.columns if result[col].isna().all()]
        result = result.drop(columns=empty)
        changes.append(f"Removed {len(empty)} completely empty column(s).")

    strategy = options["strategy"]

    for col in options["columns"]:
        if col not in result.columns:
            continue

        missing_count = int(result[col].isna().sum())

        if not missing_count or strategy == "Keep missing values":
            continue

        if strategy == "Drop rows with missing values":
            before = len(result)
            result = result.dropna(subset=[col]).copy()
            changes.append(
                f"{col}: removed {before - len(result)} row(s)."
            )

        elif strategy in (
            "Fill numeric with median",
            "Fill numeric with mean",
        ):
            if not pd.api.types.is_numeric_dtype(result[col]):
                changes.append(f"{col}: skipped because it is not numeric.")
                continue

            fill_value = (
                result[col].median()
                if strategy == "Fill numeric with median"
                else result[col].mean()
            )

            if pd.isna(fill_value):
                changes.append(f"{col}: no usable fill value exists.")
                continue

            result[col] = result[col].fillna(fill_value)
            changes.append(
                f"{col}: filled {missing_count} missing cell(s)."
            )

        elif strategy == "Fill text with mode":
            mode = result[col].mode(dropna=True)

            if mode.empty:
                changes.append(f"{col}: no mode value available.")
                continue

            result[col] = result[col].fillna(mode.iloc[0])
            changes.append(
                f"{col}: filled {missing_count} missing cell(s) with mode."
            )

    if not changes:
        changes.append("No transformations were selected.")

    return result, changes


def excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")

    return buffer.getvalue()


def metric_card(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="dp-card">
            <div class="dp-card-label">{label}</div>
            <div class="dp-card-value">{value}</div>
            <div class="dp-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="dp-brand">
            <div class="dp-logo">D</div>
            <div>
                <div class="dp-brand-name">DataPilot AI</div>
                <div class="dp-brand-sub">DATA OPERATIONS PLATFORM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📂 DATA SOURCE")

    uploaded = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx"],
        key="dataset_uploader",
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
        label_visibility="visible",
    )

    st.caption(f"CSV / XLSX · Maximum {MAX_UPLOAD_MB} MB")

    st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🛡️ QUALITY CONTROLS")

    st.markdown(
        """
        <div style="color:#C0CDE0;font-size:13px;line-height:2">
            • Missing-value checks<br>
            • Duplicate detection<br>
            • Field validation<br>
            • Cleaning preview<br>
            • Reversible changes
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="dp-footer">
            <span style="color:#48E0C2">●</span> LOCAL DEVELOPMENT<br>
            Version {APP_VERSION}<br>
            Session-based processing
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    '<div class="dp-eyebrow">DATA QUALITY · GOVERNANCE · PREPARATION</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dp-title">Enterprise Data Workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dp-subtitle">'
    'Profile, validate, clean and export datasets through a reviewable workflow.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 6. EMPTY STATE
# ============================================================

if uploaded is None:
    st.markdown(
        """
        <div class="dp-card">
            <div class="dp-eyebrow">WORKSPACE READY</div>
            <h3>Bring your data into focus.</h3>
            <p style="color:#C0CDE0;line-height:1.9">
                Upload a CSV or Excel dataset using the clearly marked
                upload control in the left sidebar. DataPilot AI will help
                you inspect data quality, review cleaning changes and export
                the prepared dataset.
            </p>
            <span class="dp-pill">CSV</span>
            &nbsp; <span class="dp-pill">EXCEL</span>
            &nbsp; <span class="dp-pill">PREVIEW BEFORE APPLY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("01 · PROFILE", "Understand", "Explore fields and statistics.")

    with c2:
        metric_card("02 · VALIDATE", "Investigate", "Identify data quality issues.")

    with c3:
        metric_card("03 · PREPARE", "Transform", "Preview and approve changes.")

    st.stop()


# ============================================================
# 7. INGESTION AND BASIC VALIDATION
# ============================================================

if uploaded.size > MAX_UPLOAD_BYTES:
    st.error(f"Upload rejected. Maximum allowed size is {MAX_UPLOAD_MB} MB.")
    st.stop()

raw_bytes = uploaded.getvalue()
file_hash = hashlib.sha256(raw_bytes).hexdigest()

try:
    if uploaded.name.lower().endswith(".csv"):
        try:
            source_df = pd.read_csv(
                BytesIO(raw_bytes),
                encoding="utf-8-sig",
                on_bad_lines="error",
            )
        except UnicodeDecodeError:
            source_df = pd.read_csv(
                BytesIO(raw_bytes),
                encoding="cp1252",
                on_bad_lines="error",
            )
    else:
        source_df = pd.read_excel(
            BytesIO(raw_bytes),
            engine="openpyxl",
        )

except Exception:
    logger.exception("Dataset import failed")
    st.error("Could not read the file. Check its format and contents.")
    st.stop()

if source_df.empty or source_df.shape[1] == 0:
    st.error("The uploaded file contains no usable data.")
    st.stop()

if len(source_df) > MAX_ROWS or len(source_df.columns) > MAX_COLUMNS:
    st.error(
        f"Dataset exceeds the configured limit of "
        f"{MAX_ROWS:,} rows or {MAX_COLUMNS} columns."
    )
    st.stop()

if source_df.columns.has_duplicates:
    st.error("Duplicate column names detected. Rename them before analysis.")
    st.stop()

if any(not str(col).strip() for col in source_df.columns):
    st.error("Blank column names detected. Rename them before analysis.")
    st.stop()


# ============================================================
# 8. SESSION STATE AND DATA VERSIONING
# ============================================================

if st.session_state.get("file_hash") != file_hash:
    st.session_state["file_hash"] = file_hash
    st.session_state["original_df"] = source_df.copy(deep=True)
    st.session_state["working_df"] = source_df.copy(deep=True)
    st.session_state["undo_stack"] = []
    st.session_state["preview_df"] = None
    st.session_state["preview_changes"] = []
    st.session_state["history"] = []
    st.session_state["run_id"] = str(uuid.uuid4())
    st.session_state["quality_findings"] = None
    st.session_state["quality_config"] = None
    st.session_state.pop("schema_validation_result", None)
    st.session_state.pop("schema_validation_duration", None)

original_df = st.session_state["original_df"]
working_df = st.session_state["working_df"]
metrics = get_metrics(working_df)

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:10px;margin:8px 0 18px">
        <span class="dp-pill">● DATASET LOADED</span>
        <span style="color:#C0CDE0;font-size:12px">
            {uploaded.name} · {uploaded.size / 1024:.1f} KB
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. EXECUTIVE KPI SUMMARY
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    metric_card("TOTAL ROWS", f"{metrics['rows']:,}", "Records")

with k2:
    metric_card("TOTAL COLUMNS", f"{metrics['columns']:,}", "Fields")

with k3:
    metric_card("MISSING CELLS", f"{metrics['missing']:,}", "Null or blank cells")

with k4:
    metric_card("MISSING RATE", f"{metrics['missing_pct']}%", "Share of all cells")

with k5:
    metric_card("DUPLICATE ROWS", f"{metrics['duplicates']:,}", "Repeated records")


# ============================================================
# 10. WORKSPACE TABS
# ============================================================

overview, quality, schema, cleaning, history = st.tabs([
    "◈  Overview",
    "⌕  Quality Audit",
    "🛡️  Schema Validation",
    "✦  Cleaning Pipeline",
    "↺  Audit History",
])


# ============================================================
# 11. OVERVIEW AND PROFILING
# ============================================================

with overview:
    st.markdown("### Dataset Preview")

    st.dataframe(
        working_df.head(PREVIEW_ROWS),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Column Quality Profile")

    st.dataframe(
        profile_columns(working_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Descriptive Statistics")

    st.dataframe(
        working_df.describe(include="all").transpose(),
        use_container_width=True,
    )

    numeric_columns = working_df.select_dtypes(
        include="number"
    ).columns.tolist()

    st.markdown("### Visual Explorer")

    if numeric_columns:
        chart_col, chart_type_col = st.columns([2, 1])

        with chart_col:
            selected_column = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="visual_column",
            )

        with chart_type_col:
            chart_type = st.selectbox(
                "Visualization",
                ["Bar", "Line", "Area", "Histogram"],
                key="visual_type",
            )

        values = working_df[selected_column].dropna()

        if values.empty:
            st.info("No numeric values are available.")

        elif chart_type == "Histogram":
            bins = min(20, max(5, int(values.nunique())))
            distribution = values.value_counts(bins=bins).sort_index()
            st.bar_chart(distribution)

        elif chart_type == "Line":
            st.line_chart(values.reset_index(drop=True))

        elif chart_type == "Area":
            st.area_chart(values.reset_index(drop=True))

        else:
            st.bar_chart(values.reset_index(drop=True))

    else:
        st.info("No numeric columns were detected.")

# ============================================================
# 12. SCHEMA AND TYPE VALIDATION
# ============================================================


with schema:
    st.markdown("### Schema & Type Validation")
    st.caption(
        "Validate required columns, expected data types and business rules. "
        "This check is read-only and will not modify your working dataset."
    )

    schema_columns = list(working_df.columns)


    required_text = st.text_input(
        "Required columns (comma-separated)",
        placeholder="customer_id, age, email",
        key="schema_required_text",
        help="Enter every column that must exist, including missing columns.",
    )
    required_columns = [
        item.strip() for item in required_text.split(",") if item.strip()
    ]

    st.markdown("#### Expected Data Types")
    supported_types = ["Do not check", "string", "integer", "numeric", "date", "boolean"]
    expected_types = {}
    type_columns = st.columns(2)

    for i, column in enumerate(schema_columns):
        with type_columns[i % 2]:
            selected_type = st.selectbox(
                f"Expected type: {column}",
                supported_types,
                key=f"schema_expected_type_{i}",
            )
        if selected_type != "Do not check":
            expected_types[column] = selected_type

    st.markdown("#### Column-Level Rules")
    rule_columns = st.multiselect(
        "Select columns that need additional rules",
        options=schema_columns,
        key="schema_rule_columns",
    )
    rules = {}

    for i, column in enumerate(rule_columns):
        with st.expander(f"Rules for {column}", expanded=False):
            not_null = st.checkbox(
                "Values must not be missing", key=f"schema_not_null_{i}"
            )
            use_min = st.checkbox(
                "Set minimum numeric value", key=f"schema_use_min_{i}"
            )
            if use_min:
                min_value = st.number_input(
                    f"Minimum for {column}", value=0.0, key=f"schema_min_{i}"
                )
            else:
                min_value = None

            use_max = st.checkbox(
                "Set maximum numeric value", key=f"schema_use_max_{i}"
            )
            if use_max:
                max_value = st.number_input(
                    f"Maximum for {column}", value=100.0, key=f"schema_max_{i}"
                )
            else:
                max_value = None

            use_allowed = st.checkbox(
                "Restrict to allowed values", key=f"schema_use_allowed_{i}"
            )
            allowed_values = []
            if use_allowed:
                allowed_text = st.text_input(
                    f"Allowed values for {column}",
                    placeholder="Active, Inactive, Pending",
                    key=f"schema_allowed_{i}",
                )
                allowed_values = [
                    value.strip() for value in allowed_text.split(",") if value.strip()
                ]

            rule = {}
            if not_null:
                rule["not_null"] = True
            if use_min:
                rule["min"] = min_value
            if use_max:
                rule["max"] = max_value
            if use_allowed:
                rule["allowed_values"] = allowed_values
            if rule:
                rules[column] = rule

    if st.button(
        "Run Schema Validation",
        type="primary",
        key="run_schema_validation",
        use_container_width=True,
    ):
        try:
            started = time.perf_counter()
            result = validate_schema(
                df=working_df,
                required_columns=required_columns,
                expected_types=expected_types,
                rules=rules,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            st.session_state["schema_validation_result"] = result
            st.session_state["schema_validation_duration"] = elapsed_ms
            logger.info(
                "schema_validation run_id=%s rows=%d errors=%d warnings=%d duration_ms=%.2f",
                st.session_state["run_id"],
                len(working_df),
                result["error_count"],
                result["warning_count"],
                elapsed_ms,
            )
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop("schema_validation_result", None)
            st.error(f"Could not run validation: {exc}")
        except Exception:
            logger.exception("Schema validation failed")
            st.error("Schema validation failed. Check the application logs.")

    result = st.session_state.get("schema_validation_result")
    if result is not None:
        st.markdown("---")
        st.markdown("#### Validation Summary")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Rows Checked", result["total_rows"])
        metric2.metric("Columns Checked", result["total_columns"])
        metric3.metric("Errors", result["error_count"])
        metric4.metric("Warnings", result["warning_count"])
        st.caption(
            "Execution time: "
            f"{st.session_state.get('schema_validation_duration', 0)} ms"
        )

        if result["valid"]:
            st.success("No validation errors found for the selected rules.")
        else:
            st.error(
                f"Validation found {result['error_count']} error(s). "
                "Review the issue report below."
            )

        issues = result["issues"]
        if issues.empty:
            st.success("No issues to display.")
        else:
            st.markdown("#### Validation Issue Report")
            issue_filter = st.selectbox(
                "Filter report",
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
                visible_issues, use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download Validation Report (CSV)",
                data=visible_issues.to_csv(index=False).encode("utf-8-sig"),
                file_name="datapilot_schema_validation_report.csv",
                mime="text/csv",
                key="download_schema_validation_report",
                use_container_width=True,
            )
        st.info(
            "Validation reports issues but does not automatically change your dataset."
        )


# ============================================================
# 13. QUALITY AUDIT
# ============================================================

with quality:
    st.markdown("### Configure Data Quality Rules")

    columns = list(working_df.columns)

    required_fields = st.multiselect(
        "Required fields",
        columns,
        key="required_fields",
        help="Flag rows with missing values in selected fields.",
    )

    unique_fields = st.multiselect(
        "Fields expected to be unique",
        columns,
        key="unique_fields",
        help="Example: employee ID or transaction ID.",
    )

    numeric_fields = st.multiselect(
        "Fields expected to be numeric",
        columns,
        key="numeric_fields",
    )

    if st.button("Run Quality Audit", type="primary"):
        started = time.perf_counter()

        findings = run_quality_checks(
            working_df,
            required_fields,
            unique_fields,
            numeric_fields,
        )

        elapsed = round((time.perf_counter() - started) * 1000, 2)

        st.session_state["quality_findings"] = findings
        st.session_state["quality_config"] = {
            "required": required_fields,
            "unique": unique_fields,
            "numeric": numeric_fields,
        }
        st.session_state["quality_duration"] = elapsed

        logger.info(
            "quality_audit run_id=%s rows=%d findings=%d duration_ms=%.2f",
            st.session_state["run_id"],
            len(working_df),
            len(findings),
            elapsed,
        )

    findings = st.session_state.get("quality_findings")

    if findings is not None:
        a, b, c = st.columns(3)

        a.metric("Total Findings", len(findings))
        b.metric(
            "High Severity",
            int((findings["Severity"] == "High").sum()),
        )
        c.metric(
            "Execution Time",
            f"{st.session_state.get('quality_duration', 0)} ms",
        )

        if findings.empty:
            st.success("No issues were found by the configured rules.")

        else:
            st.dataframe(
                findings,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Export Quality Findings",
                findings.to_csv(index=False).encode("utf-8-sig"),
                file_name="datapilot_quality_findings.csv",
                mime="text/csv",
            )


# ============================================================
# 14. CLEANING PIPELINE
# ============================================================

with cleaning:
    st.markdown("### Reviewable Cleaning Pipeline")

    st.warning(
        "Review the preview before applying changes. "
        "Dropping rows or imputing values can affect analysis results."
    )

    with st.form("cleaning_form"):
        left, right = st.columns(2)

        with left:
            trim = st.checkbox("Trim text whitespace", value=True)
            normalize = st.checkbox("Normalize blank strings", value=True)
            deduplicate = st.checkbox("Remove duplicate rows", value=False)

        with right:
            drop_empty = st.checkbox(
                "Drop completely empty columns",
                value=False,
            )

            strategy = st.selectbox(
                "Missing-value treatment",
                [
                    "Keep missing values",
                    "Drop rows with missing values",
                    "Fill numeric with median",
                    "Fill numeric with mean",
                    "Fill text with mode",
                ],
            )

        normalized = normalize_blanks(working_df)
        missing_columns = [
            col for col in normalized.columns
            if normalized[col].isna().any()
        ]

        target_columns = st.multiselect(
            "Columns for missing-value treatment",
            missing_columns,
            default=missing_columns,
        )

        generate_preview = st.form_submit_button(
            "Generate Cleaning Preview",
            type="primary",
            use_container_width=True,
        )

    if generate_preview:
        started = time.perf_counter()

        try:
            preview_df, changes = clean_data(
                working_df,
                {
                    "trim": trim,
                    "normalize": normalize,
                    "deduplicate": deduplicate,
                    "drop_empty": drop_empty,
                    "strategy": strategy,
                    "columns": target_columns,
                },
            )

            st.session_state["preview_df"] = preview_df
            st.session_state["preview_changes"] = changes
            st.session_state["preview_duration"] = round(
                (time.perf_counter() - started) * 1000, 2
            )

        except Exception:
            logger.exception("Cleaning preview failed")
            st.error("Cleaning preview failed. Check the selected options.")
            st.stop()

    preview_df = st.session_state.get("preview_df")

    if preview_df is not None:
        st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Before vs After")

        before = get_metrics(working_df)
        after = get_metrics(preview_df)

        comparison = pd.DataFrame({
            "Metric": [
                "Rows",
                "Columns",
                "Missing Cells",
                "Missing Rate (%)",
                "Duplicate Rows",
            ],
            "Before": [
                before["rows"],
                before["columns"],
                before["missing"],
                before["missing_pct"],
                before["duplicates"],
            ],
            "After": [
                after["rows"],
                after["columns"],
                after["missing"],
                after["missing_pct"],
                after["duplicates"],
            ],
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # CLEANING IMPACT ANALYSIS
        # ====================================================

        st.markdown("### 🔍 Cleaning Impact Analysis")
        st.caption(
            "Review the detailed impact of proposed cleaning "
            "changes before approving them."
        )

        try:
            summary, column_summary, cell_changes = (
                compare_dataframes(working_df, preview_df)
            )

            st.markdown("#### Overall Impact")
            if isinstance(summary, dict):
                summary_df = pd.DataFrame(
                    [
                        {
                            "Metric": str(key).replace("_", " ").title(),
                            "Value": value,
                        }
                        for key, value in summary.items()
                    ]
                )
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )
            elif isinstance(summary, pd.DataFrame):
                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write(summary)

            st.markdown("#### Column-wise Impact")
            if isinstance(column_summary, pd.DataFrame):
                if column_summary.empty:
                    st.info("No column-level changes detected.")
                else:
                    st.dataframe(
                        column_summary,
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.write(column_summary)

            st.markdown("#### Changed Cells")
            if isinstance(cell_changes, pd.DataFrame):
                if cell_changes.empty:
                    st.success("No individual cell changes detected.")
                else:
                    st.caption(
                        f"Showing {len(cell_changes):,} detected "
                        "cell-level change(s)."
                    )
                    st.dataframe(
                        cell_changes.head(500),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if len(cell_changes) > 500:
                        st.info(
                            "For performance, only the first 500 "
                            "cell changes are displayed."
                        )
            else:
                st.write(cell_changes)

        except Exception:
            logger.exception("Cleaning impact analysis failed")
            st.error(
                "Impact analysis could not be generated. "
                "Check src/datapilot/impact.py and its return format."
            )

        st.markdown("### Proposed Transformations")

        for change in st.session_state["preview_changes"]:
            st.markdown(f"- {change}")

        st.caption(
            f"Preview execution time: "
            f"{st.session_state.get('preview_duration', 0)} ms"
        )

        st.markdown("### Preview Result")

        st.dataframe(
            preview_df.head(PREVIEW_ROWS),
            use_container_width=True,
            hide_index=True,
        )

        apply_col, discard_col = st.columns(2)

        with apply_col:
            if st.button(
                "Approve & Apply Changes",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["undo_stack"].append(
                    working_df.copy(deep=True)
                )

                st.session_state["working_df"] = preview_df.copy(deep=True)

                st.session_state["history"].append({
                    "Timestamp UTC": datetime.now(timezone.utc).isoformat(),
                    "Operation": "APPLY",
                    "Run ID": st.session_state["run_id"],
                    "Rows Before": len(working_df),
                    "Rows After": len(preview_df),
                    "Changes": " | ".join(
                        st.session_state["preview_changes"]
                    ),
                })

                logger.info(
                    "cleaning_applied run_id=%s rows_before=%d rows_after=%d",
                    st.session_state["run_id"],
                    len(working_df),
                    len(preview_df),
                )

                st.session_state["preview_df"] = None
                st.session_state["preview_changes"] = []
                st.rerun()

        with discard_col:
            if st.button(
                "Discard Preview",
                use_container_width=True,
            ):
                st.session_state["preview_df"] = None
                st.session_state["preview_changes"] = []
                st.rerun()


# ============================================================
# 15. AUDIT HISTORY, UNDO AND RESET
# ============================================================

with history:
    st.markdown("### Transformation Audit History")

    events = st.session_state["history"]

    if events:
        history_df = pd.DataFrame(list(reversed(events)))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export Audit History",
            pd.DataFrame(events).to_csv(index=False).encode("utf-8-sig"),
            file_name="datapilot_audit_history.csv",
            mime="text/csv",
        )

    else:
        st.info("No transformations have been applied in this session.")

    if st.session_state["undo_stack"]:
        if st.button("Undo Last Transformation"):
            previous_df = st.session_state["undo_stack"].pop()
            rows_before = len(working_df)

            st.session_state["working_df"] = previous_df.copy(deep=True)
            st.session_state["preview_df"] = None
            st.session_state["preview_changes"] = []

            st.session_state["history"].append({
                "Timestamp UTC": datetime.now(timezone.utc).isoformat(),
                "Operation": "UNDO",
                "Run ID": st.session_state["run_id"],
                "Rows Before": rows_before,
                "Rows After": len(previous_df),
                "Changes": "Restored previous working dataset.",
            })

            st.rerun()

    if st.button("Reset to Original Uploaded Data"):
        st.session_state["working_df"] = original_df.copy(deep=True)
        st.session_state["undo_stack"] = []
        st.session_state["preview_df"] = None
        st.session_state["preview_changes"] = []

        st.session_state["history"].append({
            "Timestamp UTC": datetime.now(timezone.utc).isoformat(),
            "Operation": "RESET",
            "Run ID": st.session_state["run_id"],
            "Rows Before": len(working_df),
            "Rows After": len(original_df),
            "Changes": "Restored original uploaded dataset.",
        })

        st.rerun()


# ============================================================
# 16. EXPORT CENTER
# ============================================================

st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)
st.markdown("### Export Center")

format_col, info_col = st.columns([1, 2])

with format_col:
    export_format = st.radio(
        "Export format",
        ["CSV", "Excel"],
        horizontal=True,
    )

with info_col:
    st.caption(
        "Download the current working dataset after approved transformations."
    )

try:
    if export_format == "CSV":
        export_data = working_df.to_csv(index=False).encode("utf-8-sig")
        export_filename = "datapilot_cleaned_dataset.csv"
        export_mime = "text/csv"

    else:
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            working_df.to_excel(
                writer,
                index=False,
                sheet_name="Cleaned Data",
            )

        export_data = buffer.getvalue()
        export_filename = "datapilot_cleaned_dataset.xlsx"
        export_mime = (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )

    st.download_button(
        "Download Prepared Dataset",
        data=export_data,
        file_name=export_filename,
        mime=export_mime,
        type="primary",
        use_container_width=True,
    )

except Exception:
    logger.exception("Dataset export failed")
    st.error("Unable to prepare the export. Please try again.")


# ============================================================
# 17. FOOTER
# ============================================================

st.markdown('<div class="dp-divider"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="dp-footer">
        <b style="color:#48E0C2">DATAPILOT AI</b>
        · Data Operations Workspace · v{APP_VERSION}<br>
        Local development build. History is session-based and is not a
        persistent audit database. Production use requires authentication,
        access controls, durable audit storage, security testing and deployment
        hardening.
    </div>
    """,
    unsafe_allow_html=True,
)
