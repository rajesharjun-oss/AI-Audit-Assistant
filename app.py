from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from models import CompanyProfile, ReviewOptions
from reviewer import build_ai_review_memo, findings_to_markdown, review_pdf


st.set_page_config(page_title="AI Audit Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --ink: #101820;
        --muted: #627181;
        --line: #d9e0e8;
        --panel: #ffffff;
        --panel-soft: #f6f8fb;
        --navy: #071629;
        --navy-2: #0d243d;
        --gold: #b9934a;
        --gold-soft: #efe4cf;
        --green: #126c55;
        --red: #9d2433;
    }

    .stApp {
        background:
            linear-gradient(180deg, #f4f6f9 0%, #fbfcfd 38%, #ffffff 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244,246,249,.82);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }

    .premium-hero {
        background:
            linear-gradient(135deg, rgba(7,22,41,.98), rgba(13,36,61,.94)),
            radial-gradient(circle at 85% 12%, rgba(185,147,74,.28), transparent 34%);
        border: 1px solid rgba(185,147,74,.22);
        border-radius: 8px;
        padding: 34px 38px 30px;
        box-shadow: 0 24px 70px rgba(7,22,41,.18);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: var(--gold);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .premium-title {
        color: #ffffff !important;
        font-size: 42px;
        line-height: 1.05;
        font-weight: 700;
        margin: 0;
    }

    .premium-subtitle {
        color: #c9d5df !important;
        max-width: 780px;
        font-size: 16px;
        line-height: 1.6;
        margin: 14px 0 0;
    }

    .module-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 28px;
    }

    .module-card {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 18px 16px;
        min-height: 176px;
        box-shadow: 0 14px 38px rgba(16,24,32,.07);
    }

    .module-index {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .module-title {
        color: var(--ink);
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .module-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }

    .section-label {
        color: var(--ink);
        font-size: 18px;
        font-weight: 760;
        margin: 8px 0 8px;
    }

    .upload-shell {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px 8px;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
        margin-bottom: 22px;
    }

    .profile-shell {
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 20px 22px 8px;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
        margin-bottom: 24px;
    }

    .profile-heading {
        color: var(--ink);
        font-size: 18px;
        font-weight: 760;
        margin-bottom: 4px;
    }

    .profile-copy {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 16px;
    }

    .memo-panel {
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid var(--line);
        border-left: 4px solid var(--gold);
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 14px 38px rgba(16,24,32,.07);
        margin: 10px 0 18px;
    }

    .memo-kicker {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .memo-text {
        color: var(--ink);
        font-size: 15px;
        line-height: 1.65;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff, #f7f9fb);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 14px 34px rgba(16,24,32,.07);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 780;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #cbd5df !important;
        border-radius: 6px !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8a97a5 !important;
    }

    .stTextInput label p,
    .stTextArea label p,
    .stNumberInput label p,
    .stSelectbox label p,
    .stFileUploader label p,
    .stMultiSelect label p,
    .stSlider label p,
    .stToggle label p {
        color: var(--ink) !important;
        font-weight: 700;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    label,
    label *,
    .stToggle p,
    .stToggle span {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    .control-label {
        color: var(--ink);
        font-size: 14px;
        font-weight: 760;
        margin: 2px 0 8px;
    }

    .stToggle,
    .stToggle *,
    .stSlider,
    .stSlider *,
    .stNumberInput,
    .stNumberInput * {
        color: var(--ink) !important;
    }

    .stNumberInput button {
        background: #f3f6f9 !important;
        color: var(--ink) !important;
        border-color: #cbd5df !important;
        box-shadow: none !important;
    }

    .stSlider [data-baseweb="slider"] div {
        color: var(--ink) !important;
    }

    [data-testid="stTooltipIcon"] {
        color: #627181 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #252630 !important;
        border: 1px solid #252630 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #f7f9fb !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #ffffff !important;
        border: 1px solid #d9e0e8 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileName"] *,
    [data-testid="stFileUploaderFileSize"],
    [data-testid="stFileUploaderFileSize"] * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] svg {
        color: #627181 !important;
        fill: #627181 !important;
    }

    .stDownloadButton button,
    .stButton button {
        background: linear-gradient(135deg, var(--navy), var(--navy-2));
        color: #ffffff;
        border: 1px solid rgba(185,147,74,.48);
        border-radius: 6px;
        min-height: 42px;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(7,22,41,.18);
    }

    .stDownloadButton button:hover,
    .stButton button:hover {
        border-color: var(--gold);
        color: #ffffff;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
    }

    .streamlit-expanderHeader {
        font-weight: 700;
        color: var(--ink);
    }

    @media (max-width: 1100px) {
        .module-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .premium-title {
            font-size: 34px;
        }
    }

    @media (max-width: 680px) {
        .module-grid {
            grid-template-columns: 1fr;
        }
        .premium-hero {
            padding: 26px 22px;
        }
        .premium-title {
            font-size: 30px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="premium-hero">
        <div class="eyebrow">Financial Statement Intelligence</div>
        <div class="premium-title">AI Audit Assistant</div>
        <p class="premium-subtitle">
            A high-assurance review workspace for prepared financial statements, built to surface arithmetic,
            presentation, note-agreement, policy, and standards-checklist exceptions before final sign-off.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
        <div class="profile-heading">Engagement profile</div>
        <div class="profile-copy">Set the company context so the assistant can tailor policy and standards checks.</div>
        """,
        unsafe_allow_html=True,
    )
    profile_cols = st.columns([1.2, 1, 0.8, 0.8])
    company_name = profile_cols[0].text_input("Company name")
    industry = profile_cols[1].text_input("Industry")
    reporting_currency = profile_cols[2].text_input("Reporting currency", placeholder="Example: NGN")
    presentation_standard = profile_cols[3].selectbox("Presentation standard", ["IFRS", "Local GAAP"], index=0)
    detail_cols = st.columns(3)
    expected_policies_text = detail_cols[0].text_area(
        "Expected policies",
        placeholder="Example: revenue, financial instruments, tax",
        help="Comma-separated policy areas that are expected even if balances are not obvious in the extracted PDF text.",
    )
    significant_transactions_text = detail_cols[1].text_area(
        "Significant transactions",
        placeholder="Example: leases, share-based payments, foreign currency loans",
        help="Comma-separated transactions or balances that should have tailored accounting policy coverage.",
    )
    checklist_areas_text = detail_cols[2].text_area(
        "Force checklist areas",
        placeholder="Example: IFRS 15, IFRS 16, revenue, leases, EPS",
        help="Comma-separated standards or areas to check even if the PDF text does not clearly trigger them.",
    )
    ocr_cols = st.columns([1, 1, 1])
    ocr_cols[0].markdown('<div class="control-label">OCR scanned PDFs</div>', unsafe_allow_html=True)
    use_ocr = ocr_cols[0].toggle(
        "Enable OCR for scanned PDFs",
        value=True,
        label_visibility="collapsed",
        help="When text coverage is low, render pages in memory and run local Tesseract OCR. No OCR output or images are saved.",
    )
    ocr_max_pages = ocr_cols[1].number_input(
        "OCR page limit",
        min_value=1,
        max_value=300,
        value=60,
        step=5,
        help="Limits OCR work for very large PDFs. Increase if the full report is scanned.",
    )
    ocr_dpi = ocr_cols[2].select_slider(
        "OCR quality",
        options=[150, 200, 250, 300],
        value=200,
        help="Higher DPI can improve OCR accuracy but takes longer.",
    )

st.markdown('<div class="section-label">Audit review modules</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="module-grid">
        <div class="module-card">
            <div class="module-index">Module 01</div>
            <div class="module-title">Totals and rounding</div>
            <div class="module-copy">Totals, subtotals, cross-footings, duplicate totals, and $000s / millions labels.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 02</div>
            <div class="module-title">Formatting</div>
            <div class="module-copy">Number styles, brackets for negatives, currency markers, comparatives, and headings.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 03</div>
            <div class="module-title">Notes agreement</div>
            <div class="module-copy">Face statement references, segment totals, EPS, tax, depreciation, and note totals.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 04</div>
            <div class="module-title">Accounting policies</div>
            <div class="module-copy">Irrelevant policies, boilerplate wording, missing policies, and superseded standards.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 05</div>
            <div class="module-title">Standards checklist</div>
            <div class="module-copy">Triggered IFRS disclosure checks for presentation and significant transaction areas.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload prepared financial statement PDF", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.info("Upload a PDF to start the review.")
    st.stop()

expected_policies = tuple(item.strip() for item in expected_policies_text.split(",") if item.strip())
significant_transactions = tuple(item.strip() for item in significant_transactions_text.split(",") if item.strip())
checklist_areas = tuple(item.strip() for item in checklist_areas_text.split(",") if item.strip())
profile = CompanyProfile(
    company_name=company_name.strip(),
    industry=industry.strip(),
    reporting_currency=reporting_currency.strip(),
    expected_policies=expected_policies,
    significant_transactions=significant_transactions,
    presentation_standard=presentation_standard,
    checklist_areas=checklist_areas,
)

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    temp_file.write(uploaded.getbuffer())
    temp_path = Path(temp_file.name)

try:
    with st.spinner("Extracting PDF text, running OCR if needed, and performing review checks..."):
        result = review_pdf(
            temp_path,
            profile,
            ReviewOptions(use_ocr=use_ocr, ocr_max_pages=int(ocr_max_pages), ocr_dpi=int(ocr_dpi)),
        )
finally:
    temp_path.unlink(missing_ok=True)

st.markdown('<div class="section-label">Review dashboard</div>', unsafe_allow_html=True)
metric_cols = st.columns(9)
metric_cols[0].metric("Pages", result.metrics["pages"])
metric_cols[1].metric("Text coverage", result.metrics.get("extraction_coverage", "0%"))
metric_cols[2].metric("Confidence", result.metrics.get("extraction_confidence", "0%"))
metric_cols[3].metric("OCR pages", result.metrics.get("ocr_pages", 0))
metric_cols[4].metric("OCR tables", result.metrics.get("ocr_tables", 0))
metric_cols[5].metric("Tables", result.metrics["tables"])
metric_cols[6].metric("Findings", result.metrics["findings"])
metric_cols[7].metric("High", result.metrics["high"])
metric_cols[8].metric("Medium", result.metrics["medium"])

st.markdown(
    f"""
    <section class="memo-panel">
        <div class="memo-kicker">Executive review memo</div>
        <div class="memo-text">{build_ai_review_memo(result)}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

markdown_report = findings_to_markdown(result)
st.download_button(
    "Download review report",
    markdown_report,
    file_name="financial_statement_review.md",
    mime="text/markdown",
)

if not result.findings:
    st.success("No issues were detected by the automated checks.")
    st.stop()

severity_order = ["High", "Medium", "Low"]
filter_cols = st.columns([1.4, 1])
category_filter = filter_cols[0].multiselect(
    "Category",
    sorted({finding.category for finding in result.findings}),
)
severity_filter = filter_cols[1].multiselect("Severity", severity_order, default=severity_order)

filtered = [
    finding
    for finding in result.findings
    if (not category_filter or finding.category in category_filter)
    and (not severity_filter or finding.severity in severity_filter)
]

rows = [
    {
        "Severity": finding.severity,
        "Category": finding.category,
        "Location": finding.location,
        "Issue": finding.issue,
        "Evidence": finding.evidence,
        "Recommendation": finding.recommendation,
    }
    for finding in filtered
]
st.markdown('<div class="section-label">Exception register</div>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

for finding in filtered:
    with st.expander(f"{finding.severity}: {finding.issue}", expanded=finding.severity == "High"):
        st.write(f"**Category:** {finding.category}")
        st.write(f"**Location:** {finding.location}")
        st.write(f"**Evidence:** {finding.evidence}")
        st.write(f"**Recommendation:** {finding.recommendation}")
