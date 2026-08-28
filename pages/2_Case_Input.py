import streamlit as st
import pandas as pd
import re
import requests

from datetime import datetime

import pytesseract
from pdf2image import convert_from_bytes

from database.repository import (
    get_cases,
    add_case
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Case Registration & ML Prediction",
    page_icon="📥",
    layout="wide"
)


# ============================================================
# ML BACKEND
# ============================================================

ML_API_URL = "http://127.0.0.1:8000/predict"


def get_ml_prediction(case_data):

    try:

        response = requests.post(
            ML_API_URL,
            json=case_data,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            "ML backend is not running.\n\n"
            "Start it using:\n"
            "uvicorn backend.main:app --reload --port 8000"
        )

        return None

    except requests.exceptions.RequestException as exc:

        st.error(
            f"ML prediction error: {exc}"
        )

        return None


# ============================================================
# OCR CONFIGURATION
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

POPPLER_PATH = (
    r"C:\poppler\poppler-24.08.0\Library\bin"
)


# ============================================================
# LOAD POSTGRESQL DATA
# ============================================================

@st.cache_data(ttl=30)
def load_cases():

    return get_cases()


try:

    df = load_cases()

except Exception as exc:

    st.error(
        "Unable to connect to PostgreSQL."
    )

    st.exception(exc)

    st.stop()


if df.empty:

    st.warning(
        "No cases found in PostgreSQL."
    )

    st.stop()


# ============================================================
# HELPERS
# ============================================================

def get_districts():

    if "district_name" not in df.columns:

        return []

    return sorted(
        df["district_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_stages():

    if "stage_of_case" not in df.columns:

        return []

    return sorted(
        df["stage_of_case"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_workload(district):

    if "district_workload_level" not in df.columns:

        return "Medium"

    rows = df[
        df["district_name"]
        .astype(str)
        .str.lower()
        ==
        str(district).lower()
    ]

    if rows.empty:

        return "Medium"

    values = (
        rows[
            "district_workload_level"
        ]
        .dropna()
        .astype(str)
    )

    if values.empty:

        return "Medium"

    return str(
        values.mode().iloc[0]
    )


# ============================================================
# DISPLAY ML RESULT
# ============================================================

def display_ml_result(result):

    if not result:

        return

    st.divider()

    st.subheader(
        "🤖 XGBoost Prediction"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        duration = float(
            result.get(
                "predicted_duration_years",
                0
            )
        )

        st.metric(
            "Predicted Duration",
            f"{duration:.2f} years"
        )

    with col2:

        predicted_risk = result.get(
            "predicted_risk",
            "Unknown"
        )

        st.metric(
            "Predicted Delay Risk",
            predicted_risk
        )

    with col3:

        probabilities = result.get(
            "risk_probabilities",
            {}
        )

        high_probability = float(
            probabilities.get(
                "High",
                0
            )
        )

        st.metric(
            "High-Risk Probability",
            f"{high_probability * 100:.1f}%"
        )

    # ========================================================
    # RISK PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Risk Probabilities"
    )

    if probabilities:

        for category, probability in probabilities.items():

            probability = float(
                probability
            )

            st.write(
                f"**{category}:** "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    explanation = result.get(
        "explanation",
        []
    )

    if explanation:

        st.subheader(
            "🔍 SHAP Explanation"
        )

        explanation_df = pd.DataFrame(
            explanation
        )

        if "shap_value" in explanation_df.columns:

            explanation_df[
                "shap_value"
            ] = pd.to_numeric(
                explanation_df[
                    "shap_value"
                ],
                errors="coerce"
            ).round(4)

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "SHAP values show which model features "
            "contributed most to the prediction."
        )

    else:

        st.info(
            "No SHAP explanation was returned."
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📥 Case Registration & ML Prediction"
)

st.markdown(
    "### Enter a new case or extract case details "
    "from an eCourts PDF."
)

st.caption(
    "Cases are stored in PostgreSQL and evaluated using "
    "XGBoost-based risk and duration predictions."
)


# ============================================================
# INPUT METHOD
# ============================================================

input_method = st.radio(
    "Choose Input Method",
    [
        "Manual Entry",
        "PDF Upload"
    ],
    horizontal=True
)


# ============================================================
# MANUAL ENTRY
# ============================================================

if input_method == "Manual Entry":

    st.subheader(
        "✍️ Manual Case Entry"
    )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        districts = get_districts()

        if districts:

            district = st.selectbox(
                "District",
                districts
            )

        else:

            district = st.text_input(
                "District"
            )

    with col2:

        court_name = st.text_input(
            "Court Name"
        )

    col1, col2 = st.columns(2)

    with col1:

        stages = get_stages()

        if stages:

            stage = st.selectbox(
                "Stage of Case",
                stages
            )

        else:

            stage = st.text_input(
                "Stage of Case"
            )

    with col2:

        case_status = st.selectbox(
            "Case Status",
            [
                "PENDING",
                "DISPOSED"
            ]
        )

    # ========================================================
    # CASE DETAILS
    # ========================================================

    st.subheader(
        "📋 Case Details"
    )

    col1, col2 = st.columns(2)

    with col1:

        cnr_number = st.text_input(
            "CNR Number"
        )

        registration_number = st.text_input(
            "Registration Number"
        )

        case_type = st.text_input(
            "Case Type",
            value="Criminal"
        )

        act_name = st.text_input(
            "Act Name",
            value="IPC"
        )

    with col2:

        section = st.text_input(
            "Section",
            value="Unknown"
        )

        police_station = st.text_input(
            "Police Station",
            value="Unknown"
        )

        source_type = st.selectbox(
            "Source Type",
            [
                "MANUAL"
            ]
        )

    # ========================================================
    # DATES
    # ========================================================

    st.subheader(
        "📅 Case Dates"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        filing_date = st.date_input(
            "Filing Date"
        )

    with col2:

        first_hearing_date = st.date_input(
            "First Hearing Date"
        )

    with col3:

        decision_date = st.date_input(
            "Decision Date",
            value=None
        )

    # ========================================================
    # CALCULATED FEATURES
    # ========================================================

    filing_datetime = datetime.combine(
        filing_date,
        datetime.min.time()
    )

    today = datetime.today()

    case_age_days = max(
        0,
        (
            today
            -
            filing_datetime
        ).days
    )

    first_hearing_datetime = datetime.combine(
        first_hearing_date,
        datetime.min.time()
    )

    hearing_gap_days = max(
        0,
        (
            today
            -
            first_hearing_datetime
        ).days
    )

    first_hearing_delay_days = max(
        0,
        (
            first_hearing_datetime
            -
            filing_datetime
        ).days
    )

    filing_year = filing_date.year

    filing_month = filing_date.month

    st.subheader(
        "🧮 Calculated ML Features"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Case Age",
            f"{case_age_days} days"
        )

    with c2:

        st.metric(
            "First Hearing Delay",
            f"{first_hearing_delay_days} days"
        )

    with c3:

        st.metric(
            "Hearing Gap",
            f"{hearing_gap_days} days"
        )

    # ========================================================
    # ADD CASE
    # ========================================================

    if st.button(
        "➕ Add Case",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        errors = []

        if not district.strip():

            errors.append(
                "District is required."
            )

        if not court_name.strip():

            errors.append(
                "Court Name is required."
            )

        if not stage.strip():

            errors.append(
                "Stage of Case is required."
            )

        if not cnr_number.strip():

            errors.append(
                "CNR Number is required."
            )

        if errors:

            for error in errors:

                st.error(
                    error
                )

            st.stop()

        # ====================================================
        # ML INPUT
        # ====================================================

        ml_case = {

            "district_name":
                district,

            "district_workload_level":
                get_workload(
                    district
                ),

            "court_name":
                court_name,

            "case_type":
                case_type,

            "stage_of_case":
                stage,

            "act_name":
                act_name,

            "section":
                section,

            "police_station":
                police_station,

            "case_age_days":
                case_age_days,

            "hearing_gap_days":
                hearing_gap_days,

            "filing_year":
                filing_year,

            "filing_month":
                filing_month,

            "filing_month_feature":
                filing_month,

            "first_hearing_delay_days":
                first_hearing_delay_days
        }

        # ====================================================
        # ML PREDICTION
        # ====================================================

        with st.spinner(
            "Running XGBoost prediction..."
        ):

            ml_result = get_ml_prediction(
                ml_case
            )

        display_ml_result(
            ml_result
        )

        # ====================================================
        # PREPARE DATABASE RECORD
        # ====================================================

        case_data = {

            "cnr_number":
                cnr_number.strip(),

            "registration_number":
                registration_number.strip(),

            "district_name":
                district.strip(),

            "court_name":
                court_name.strip(),

            "stage_of_case":
                stage.strip(),

            "case_type":
                case_type.strip(),

            "act_name":
                act_name.strip(),

            "section":
                section.strip(),

            "police_station":
                police_station.strip(),

            "filing_date":
                filing_date,

            "first_hearing_date":
                first_hearing_date,

            "decision_date":
                decision_date,

            "case_status":
                case_status,

            "disposal_type":
                None,

            "source_type":
                source_type
        }

        # ====================================================
        # INSERT INTO POSTGRESQL
        # ====================================================

        try:

            new_case_id = add_case(
                case_data
            )

            st.success(
                "✅ Case added successfully to PostgreSQL."
            )

            st.info(
                f"Case ID: {new_case_id}"
            )

            # ------------------------------------------------
            # SHOW ML RESULT AGAIN
            # ------------------------------------------------

            if ml_result:

                st.subheader(
                    "🤖 Saved Case ML Result"
                )

                predicted_risk = ml_result.get(
                    "predicted_risk",
                    "Unknown"
                )

                predicted_duration = float(
                    ml_result.get(
                        "predicted_duration_years",
                        0
                    )
                )

                probabilities = ml_result.get(
                    "risk_probabilities",
                    {}
                )

                high_probability = float(
                    probabilities.get(
                        "High",
                        0
                    )
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "ML Risk",
                        predicted_risk
                    )

                with c2:

                    st.metric(
                        "Predicted Duration",
                        f"{predicted_duration:.2f} years"
                    )

                with c3:

                    st.metric(
                        "High-Risk Probability",
                        f"{high_probability * 100:.1f}%"
                    )

            # ------------------------------------------------
            # REFRESH
            # ------------------------------------------------

            st.cache_data.clear()

        except Exception as exc:

            st.error(
                "Unable to save the case to PostgreSQL."
            )

            st.exception(exc)


# ============================================================
# PDF UPLOAD
# ============================================================

else:

    st.subheader(
        "📄 Upload eCourts PDF"
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_pdf:

        # ====================================================
        # OCR
        # ====================================================

        text = ""

        try:

            pdf_bytes = uploaded_pdf.read()

            images = convert_from_bytes(
                pdf_bytes,
                poppler_path=POPPLER_PATH
            )

            progress = st.progress(
                0
            )

            total = len(images)

            for index, image in enumerate(images):

                extracted = (
                    pytesseract.image_to_string(
                        image
                    )
                )

                text += (
                    extracted
                    +
                    "\n"
                )

                progress.progress(
                    (index + 1) / total
                )

            progress.empty()

        except Exception as exc:

            st.error(
                f"OCR Error: {exc}"
            )

            st.stop()

        # ====================================================
        # OCR TEXT
        # ====================================================

        st.subheader(
            "📑 Extracted OCR Text"
        )

        st.text_area(
            "OCR Output",
            text,
            height=300
        )

        if not text.strip():

            st.warning(
                "No text could be extracted from the PDF."
            )

            st.stop()

        lower_text = text.lower()

        # ====================================================
        # DISTRICT
        # ====================================================

        district = "Unknown"

        district_map = {

            "mysuru":
                "Mysuru",

            "mysore":
                "Mysuru",

            "bengaluru":
                "Bengaluru Urban",

            "bangalore":
                "Bengaluru Urban",

            "tumakuru":
                "Tumakuru",

            "tumkur":
                "Tumakuru",

            "belagavi":
                "Belagavi",

            "kalaburagi":
                "Kalaburagi"
        }

        for key, value in district_map.items():

            if key in lower_text:

                district = value

                break

        # ====================================================
        # COURT
        # ====================================================

        court_name = ""

        court_patterns = [

            r"PRL\.?\s*.*?COURT",

            r"COURT\s+OF\s+.*",

            r"COURT.*?MYSURU",

            r"COURT.*?BENGALURU",

            r"COURT.*?TUMAKURU"
        ]

        for pattern in court_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                court_name = (
                    match.group(0)
                    .strip()
                )

                break

        # ====================================================
        # STAGE
        # ====================================================

        stage = "Unknown"

        stage_keywords = [

            "RE-ISSUE SUMMONS",

            "REISSUE SUMMONS",

            "EVIDENCE",

            "HEARING",

            "ARGUMENT",

            "JUDGMENT",

            "SUMMONS",

            "APPEARANCE",

            "NBW",

            "STEPS"
        ]

        for keyword in stage_keywords:

            if keyword.lower() in lower_text:

                stage = keyword

                break

        # ====================================================
        # STATUS
        # ====================================================

        case_status = "PENDING"

        if (
            "disposed" in lower_text
            or
            "case disposed" in lower_text
            or
            "closed" in lower_text
        ):

            case_status = "DISPOSED"

        # ====================================================
        # CNR
        # ====================================================

        cnr_number = ""

        cnr_patterns = [

            r"\b[A-Z]{4}\d{12}\b",

            r"\b[A-Z]{4}\d{10,14}\b"
        ]

        for pattern in cnr_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                cnr_number = (
                    match.group(0)
                    .upper()
                )

                break

        # ====================================================
        # REGISTRATION NUMBER
        # ====================================================

        registration_number = ""

        reg_patterns = [

            r"C\.C\.\s*No\.?\s*\d+\s*/\s*\d{4}",

            r"\b\d{1,6}\s*/\s*\d{4}\b"
        ]

        for pattern in reg_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                registration_number = (
                    match.group(0)
                    .strip()
                )

                break

        # ====================================================
        # DATES
        # ====================================================

        all_dates = re.findall(
            r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
            text
        )

        def parse_date_string(value):

            value = value.replace(
                "/",
                "-"
            )

            try:

                return datetime.strptime(
                    value,
                    "%d-%m-%Y"
                ).date()

            except ValueError:

                return None

        parsed_dates = []

        for value in all_dates:

            parsed = parse_date_string(
                value
            )

            if parsed:

                parsed_dates.append(
                    parsed
                )

        filing_date = (

            parsed_dates[0]

            if parsed_dates

            else datetime.today().date()

        )

        first_hearing_date = (

            parsed_dates[1]

            if len(parsed_dates) >= 2

            else filing_date

        )

        decision_date = None

        if case_status == "DISPOSED":

            if len(parsed_dates) >= 3:

                decision_date = (
                    parsed_dates[-1]
                )

        # ====================================================
        # CALCULATED FEATURES
        # ====================================================

        filing_datetime = datetime.combine(
            filing_date,
            datetime.min.time()
        )

        first_hearing_datetime = datetime.combine(
            first_hearing_date,
            datetime.min.time()
        )

        case_age_days = max(
            0,
            (
                datetime.today()
                -
                filing_datetime
            ).days
        )

        first_hearing_delay_days = max(
            0,
            (
                first_hearing_datetime
                -
                filing_datetime
            ).days
        )

        hearing_gap_days = max(
            0,
            (
                datetime.today()
                -
                first_hearing_datetime
            ).days
        )

        filing_year = (
            filing_date.year
        )

        filing_month = (
            filing_date.month
        )

        # ====================================================
        # VERIFY INFORMATION
        # ====================================================

        st.subheader(
            "✅ Verify Extracted Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            district = st.text_input(
                "District",
                value=district
            )

            court_name = st.text_input(
                "Court Name",
                value=court_name
            )

            cnr_number = st.text_input(
                "CNR Number",
                value=cnr_number
            )

            registration_number = st.text_input(
                "Registration Number",
                value=registration_number
            )

            stage = st.text_input(
                "Stage of Case",
                value=stage
            )

        with col2:

            case_status = st.selectbox(
                "Case Status",
                [
                    "PENDING",
                    "DISPOSED"
                ],
                index=(
                    1
                    if case_status == "DISPOSED"
                    else 0
                )
            )

            filing_date = st.date_input(
                "Filing Date",
                value=filing_date
            )

            first_hearing_date = st.date_input(
                "First Hearing Date",
                value=first_hearing_date
            )

            if decision_date:

                decision_date = st.date_input(
                    "Decision Date",
                    value=decision_date
                )

            else:

                decision_date = st.date_input(
                    "Decision Date",
                    value=None
                )

        # ====================================================
        # ADDITIONAL ML INFORMATION
        # ====================================================

        st.subheader(
            "🧠 ML Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            case_type = st.text_input(
                "Case Type",
                value="Criminal"
            )

            act_name = st.text_input(
                "Act Name",
                value="IPC"
            )

        with col2:

            section = st.text_input(
                "Section",
                value="Unknown"
            )

            police_station = st.text_input(
                "Police Station",
                value="Unknown"
            )

        # ====================================================
        # RECALCULATE FEATURES
        # ====================================================

        filing_datetime = datetime.combine(
            filing_date,
            datetime.min.time()
        )

        first_hearing_datetime = datetime.combine(
            first_hearing_date,
            datetime.min.time()
        )

        case_age_days = max(
            0,
            (
                datetime.today()
                -
                filing_datetime
            ).days
        )

        first_hearing_delay_days = max(
            0,
            (
                first_hearing_datetime
                -
                filing_datetime
            ).days
        )

        hearing_gap_days = max(
            0,
            (
                datetime.today()
                -
                first_hearing_datetime
            ).days
        )

        filing_year = filing_date.year

        filing_month = filing_date.month

        # ====================================================
        # SHOW FEATURES
        # ====================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Case Age",
                f"{case_age_days} days"
            )

        with c2:

            st.metric(
                "First Hearing Delay",
                f"{first_hearing_delay_days} days"
            )

        with c3:

            st.metric(
                "Hearing Gap",
                f"{hearing_gap_days} days"
            )

        # ====================================================
        # ADD EXTRACTED CASE
        # ====================================================

        if st.button(
            "➕ Add Extracted Case",
            type="primary",
            use_container_width=True
        ):

            errors = []

            if not district.strip():

                errors.append(
                    "District is required."
                )

            if not court_name.strip():

                errors.append(
                    "Court Name is required."
                )

            if not stage.strip():

                errors.append(
                    "Stage of Case is required."
                )

            if not cnr_number.strip():

                errors.append(
                    "CNR Number is required."
                )

            if errors:

                for error in errors:

                    st.error(
                        error
                    )

                st.stop()

            # =================================================
            # ML CASE
            # =================================================

            ml_case = {

                "district_name":
                    district,

                "district_workload_level":
                    get_workload(
                        district
                    ),

                "court_name":
                    court_name,

                "case_type":
                    case_type,

                "stage_of_case":
                    stage,

                "act_name":
                    act_name,

                "section":
                    section,

                "police_station":
                    police_station,

                "case_age_days":
                    case_age_days,

                "hearing_gap_days":
                    hearing_gap_days,

                "filing_year":
                    filing_year,

                "filing_month":
                    filing_month,

                "filing_month_feature":
                    filing_month,

                "first_hearing_delay_days":
                    first_hearing_delay_days
            }

            # =================================================
            # ML PREDICTION
            # =================================================

            with st.spinner(
                "Running XGBoost prediction..."
            ):

                ml_result = get_ml_prediction(
                    ml_case
                )

            display_ml_result(
                ml_result
            )

            # =================================================
            # DATABASE RECORD
            # =================================================

            case_data = {

                "cnr_number":
                    cnr_number.strip(),

                "registration_number":
                    registration_number.strip(),

                "district_name":
                    district.strip(),

                "court_name":
                    court_name.strip(),

                "stage_of_case":
                    stage.strip(),

                "case_type":
                    case_type.strip(),

                "act_name":
                    act_name.strip(),

                "section":
                    section.strip(),

                "police_station":
                    police_station.strip(),

                "filing_date":
                    filing_date,

                "first_hearing_date":
                    first_hearing_date,

                "decision_date":
                    decision_date,

                "case_status":
                    case_status,

                "disposal_type":
                    None,

                "source_type":
                    "PDF_OCR"
            }

            # =================================================
            # INSERT INTO POSTGRESQL
            # =================================================

            try:

                new_case_id = add_case(
                    case_data
                )

                st.success(
                    "✅ Extracted case added successfully to PostgreSQL."
                )

                st.info(
                    f"Case ID: {new_case_id}"
                )

                st.cache_data.clear()

            except Exception as exc:

                st.error(
                    "Unable to save extracted case to PostgreSQL."
                )

                st.exception(exc)


# ============================================================
# END
# ============================================================

st.divider()

st.caption(
    "Case registration and extracted case data are stored in "
    "PostgreSQL. XGBoost provides duration and delay-risk "
    "predictions, while SHAP provides model explanations."
)