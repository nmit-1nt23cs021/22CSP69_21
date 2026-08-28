import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import pytesseract

from pdf2image import convert_from_bytes

from database.repository import get_cases, add_case


# ============================================================
# CONFIGURATION
# ============================================================

ML_API_URL = "http://127.0.0.1:8000/predict"

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

POPPLER_PATH = (
    r"C:\poppler\poppler-24.08.0\Library\bin"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Case Input",
    page_icon="📥",
    layout="wide"
)


# ============================================================
# LOAD CASES FROM POSTGRESQL
# ============================================================

@st.cache_data(ttl=30)
def load_cases():

    try:

        data = get_cases()

        return data

    except Exception as e:

        st.error(
            f"Unable to load cases from PostgreSQL: {e}"
        )

        return pd.DataFrame()


df = load_cases()


# ============================================================
# BASIC DATABASE CHECK
# ============================================================

if df.empty:

    st.warning(
        "No case data was found in PostgreSQL."
    )

    st.info(
        "Make sure PostgreSQL is running and the "
        "database connection is working."
    )


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "📥 Dynamic Case Input & OCR Extraction"
)

st.markdown(
    "### Add a new case using manual entry or eCourts PDF OCR."
)

st.caption(
    "All cases are stored in PostgreSQL. "
    "No CSV files are used."
)


# ============================================================
# HELPER — DISTRICTS
# ============================================================

def get_districts():

    if (
        not df.empty
        and "district_name" in df.columns
    ):

        districts = (
            df["district_name"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        districts = [
            value
            for value in districts.unique()
            if value
            and value.lower() != "unknown"
        ]

        return sorted(districts)

    return []


# ============================================================
# HELPER — STAGES
# ============================================================

def get_stages():

    if (
        not df.empty
        and "stage_of_case" in df.columns
    ):

        stages = (
            df["stage_of_case"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        stages = [
            value
            for value in stages.unique()
            if value
            and value.lower() != "unknown"
        ]

        return sorted(stages)

    return []


# ============================================================
# HELPER — DISTRICT WORKLOAD
# ============================================================

def get_district_workload(district):

    if (
        not df.empty
        and "district_name" in df.columns
        and "district_workload_level" in df.columns
    ):

        matching_rows = df[
            df["district_name"].astype(str).str.strip()
            == str(district).strip()
        ]

        if not matching_rows.empty:

            values = (
                matching_rows[
                    "district_workload_level"
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            values = values[
                values.str.lower() != "unknown"
            ]

            if not values.empty:

                return str(
                    values.mode().iloc[0]
                )

    return "Medium"


# ============================================================
# HELPER — CALCULATE CASE AGE
# ============================================================

def calculate_case_age(filing_date):

    if not filing_date:
        return 0

    try:

        filing_dt = pd.to_datetime(
            filing_date,
            errors="coerce"
        )

        if pd.isna(filing_dt):
            return 0

        today = pd.Timestamp.today().normalize()

        age = (
            today - filing_dt.normalize()
        ).days

        return max(
            0,
            int(age)
        )

    except Exception:

        return 0


# ============================================================
# HELPER — CALL ML BACKEND
# ============================================================

def get_ml_prediction(case_data):

    try:

        response = requests.post(
            ML_API_URL,
            json=case_data,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            "ML backend is not running.\n\n"
            "Start it in another terminal using:\n"
            "uvicorn backend.main:app --reload --port 8000"
        )

        return None

    except requests.exceptions.Timeout:

        st.error(
            "The ML backend took too long to respond."
        )

        return None

    except requests.exceptions.HTTPError:

        try:

            detail = response.json()

        except Exception:

            detail = response.text

        st.error(
            f"ML backend returned an error: {detail}"
        )

        return None

    except requests.exceptions.RequestException as e:

        st.error(
            f"ML prediction error: {e}"
        )

        return None


# ============================================================
# HELPER — DISPLAY ML RESULT
# ============================================================

def display_ml_result(ml_result):

    if not ml_result:
        return

    st.divider()

    st.subheader(
        "🤖 Machine Learning Prediction"
    )

    # --------------------------------------------------------
    # DURATION AND RISK
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        predicted_duration = float(
            ml_result.get(
                "predicted_duration_years",
                0
            )
        )

        st.metric(
            "Predicted Case Duration",
            f"{predicted_duration:.2f} years"
        )

    with col2:

        predicted_risk = ml_result.get(
            "predicted_risk",
            "Unknown"
        )

        st.metric(
            "ML Predicted Delay Risk",
            predicted_risk
        )

    # --------------------------------------------------------
    # RISK PROBABILITIES
    # --------------------------------------------------------

    st.subheader(
        "📊 Risk Probabilities"
    )

    probabilities = ml_result.get(
        "risk_probabilities",
        {}
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
                max(
                    0.0,
                    min(
                        1.0,
                        probability
                    )
                )
            )

    else:

        st.info(
            "No risk probabilities were returned."
        )

    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "🔍 ML Explanation — SHAP"
    )

    explanation = ml_result.get(
        "explanation",
        []
    )

    if explanation:

        explanation_df = pd.DataFrame(
            explanation
        )

        if "shap_value" in explanation_df.columns:

            explanation_df[
                "shap_value"
            ] = explanation_df[
                "shap_value"
            ].round(4)

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "SHAP values show which model features "
            "had the greatest influence on the prediction."
        )

    else:

        st.info(
            "No SHAP explanation was returned."
        )


# ============================================================
# HELPER — SAVE CASE TO POSTGRESQL
# ============================================================

def save_case_to_database(case_data):

    try:

        case_id = add_case(
            case_data
        )

        return case_id

    except Exception as e:

        st.error(
            f"Unable to save case to PostgreSQL: {e}"
        )

        return None


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
        "✍ Manual Case Entry"
    )

    # --------------------------------------------------------
    # DATABASE OPTIONS
    # --------------------------------------------------------

    districts = get_districts()

    stages = get_stages()

    # --------------------------------------------------------
    # BASIC CASE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "#### Basic Case Information"
    )

    col1, col2 = st.columns(2)

    with col1:

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
                "Pending",
                "Disposed"
            ]
        )

    # --------------------------------------------------------
    # IDENTIFICATION
    # --------------------------------------------------------

    st.markdown(
        "#### Case Identification"
    )

    col1, col2 = st.columns(2)

    with col1:

        cnr_number = st.text_input(
            "CNR Number"
        )

    with col2:

        registration_number = st.text_input(
            "Registration Number"
        )

    # --------------------------------------------------------
    # CASE DETAILS
    # --------------------------------------------------------

    st.markdown(
        "#### Case Details"
    )

    col1, col2 = st.columns(2)

    with col1:

        case_type = st.text_input(
            "Case Type",
            value="Criminal"
        )

    with col2:

        act_name = st.text_input(
            "Act Name",
            value="IPC"
        )

    col1, col2 = st.columns(2)

    with col1:

        section = st.text_input(
            "Section",
            value="Unknown"
        )

    with col2:

        police_station = st.text_input(
            "Police Station",
            value="Unknown"
        )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    st.markdown(
        "#### Case Dates"
    )

    filing_date = st.date_input(
        "Filing Date",
        value=datetime.today()
    )

    first_hearing_date = st.date_input(
        "First Hearing Date",
        value=None
    )

    # --------------------------------------------------------
    # ML INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "#### 🧠 ML Information"
    )

    case_age = calculate_case_age(
        filing_date
    )

    st.info(
        f"Calculated Case Age: **{case_age} days**"
    )

    hearing_gap = st.number_input(
        "Average Hearing Gap (Days)",
        min_value=0,
        value=0,
        step=1
    )

    # --------------------------------------------------------
    # FILING INFORMATION
    # --------------------------------------------------------

    filing_year = int(
        filing_date.year
    )

    filing_month = int(
        filing_date.month
    )

    district_workload = (
        get_district_workload(
            district
        )
    )

    st.info(
        f"District Workload Level: "
        f"**{district_workload}**"
    )

    # --------------------------------------------------------
    # PREDICT + SAVE
    # --------------------------------------------------------

    if st.button(
        "🤖 Predict and Add Case",
        type="primary",
        key="manual_predict_add"
    ):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not district.strip():

            st.error(
                "Please enter/select a district."
            )

            st.stop()

        if not court_name.strip():

            st.error(
                "Please enter the court name."
            )

            st.stop()

        if not stage.strip():

            st.error(
                "Please enter/select the stage."
            )

            st.stop()

        if not case_type.strip():

            st.error(
                "Please enter the case type."
            )

            st.stop()

        # ----------------------------------------------------
        # ML INPUT
        # ----------------------------------------------------

        ml_case = {

            "district_name":
                district.strip(),

            "district_workload_level":
                district_workload,

            "court_name":
                court_name.strip(),

            "case_type":
                case_type.strip(),

            "stage_of_case":
                stage.strip(),

            "act_name":
                act_name.strip(),

            "section":
                section.strip(),

            "police_station":
                police_station.strip(),

            "case_age_days":
                float(case_age),

            "hearing_gap_days":
                float(hearing_gap),

            "filing_year":
                filing_year,

            "filing_month":
                filing_month
        }

        # ----------------------------------------------------
        # CALL ML
        # ----------------------------------------------------

        with st.spinner(
            "Running XGBoost prediction and SHAP explanation..."
        ):

            ml_result = get_ml_prediction(
                ml_case
            )

        if ml_result:

            # ------------------------------------------------
            # DISPLAY ML RESULT
            # ------------------------------------------------

            display_ml_result(
                ml_result
            )

            # ------------------------------------------------
            # SAVE TO POSTGRESQL
            # ------------------------------------------------

            first_hearing_value = None

            if first_hearing_date:

                first_hearing_value = (
                    first_hearing_date
                )

            case_to_save = {

                "cnr_number":
                    cnr_number.strip()
                    if cnr_number
                    else None,

                "registration_number":
                    registration_number.strip()
                    if registration_number
                    else None,

                "district_name":
                    district.strip(),

                "court_name":
                    court_name.strip(),

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
                    first_hearing_value,

                "decision_date":
                    None,

                "case_status":
                    case_status,

                "stage_of_case":
                    stage.strip(),

                "disposal_type":
                    None,

                "source_type":
                    "MANUAL"
            }

            case_id = save_case_to_database(
                case_to_save
            )

            if case_id:

                st.success(
                    f"Case added successfully to PostgreSQL. "
                    f"Case ID: {case_id}"
                )

                st.info(
                    "The case is now stored in PostgreSQL. "
                    "No CSV file was created or modified."
                )

                st.cache_data.clear()


# ============================================================
# PDF UPLOAD
# ============================================================

elif input_method == "PDF Upload":

    st.subheader(
        "📄 Upload eCourts PDF"
    )

    uploaded_pdf = st.file_uploader(
        "Upload eCourts PDF",
        type=["pdf"]
    )

    if uploaded_pdf:

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        st.subheader(
            "🔎 OCR Processing"
        )

        text = ""

        try:

            pdf_bytes = (
                uploaded_pdf.getvalue()
            )

            images = convert_from_bytes(
                pdf_bytes,
                poppler_path=POPPLER_PATH
            )

            progress = st.progress(
                0
            )

            total_pages = len(
                images
            )

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
                    (index + 1) / total_pages
                )

            progress.empty()

        except Exception as e:

            st.error(
                f"OCR Error: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # SHOW OCR TEXT
        # ----------------------------------------------------

        st.subheader(
            "📑 Extracted OCR Text"
        )

        st.text_area(
            "OCR Output",
            text,
            height=300
        )

        # ====================================================
        # EXTRACT INFORMATION
        # ====================================================

        text_lower = text.lower()

        # ----------------------------------------------------
        # DISTRICT
        # ----------------------------------------------------

        district = "Unknown"

        district_map = {

            "mysuru":
                "Mysuru",

            "bengaluru":
                "Bengaluru Urban",

            "tumakuru":
                "Tumakuru",

            "belagavi":
                "Belagavi",

            "kalaburagi":
                "Kalaburagi"
        }

        for key, value in district_map.items():

            if key in text_lower:

                district = value

                break

        # ----------------------------------------------------
        # COURT
        # ----------------------------------------------------

        court_name = ""

        court_patterns = [

            r"PRL.*?NANJANGUD",

            r"COURT.*?MYSURU",

            r"JUDGE.*?MYSURU"
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

        # ----------------------------------------------------
        # STAGE
        # ----------------------------------------------------

        stage = "Unknown"

        stage_keywords = [

            "EVIDENCE",

            "HEARING",

            "ARGUMENT",

            "JUDGMENT",

            "SUMMONS",

            "RE ISSUE SUMMONS",

            "STEPS",

            "NBW"
        ]

        for keyword in stage_keywords:

            if keyword.lower() in text_lower:

                stage = (
                    keyword.title()
                )

                break

        # ----------------------------------------------------
        # CASE STATUS
        # ----------------------------------------------------

        case_status = "Pending"

        if "disposed" in text_lower:

            case_status = "Disposed"

        # ----------------------------------------------------
        # REGISTRATION NUMBER
        # ----------------------------------------------------

        registration_number = ""

        reg_patterns = [

            r"\d{1,6}/\d{4}",

            r"C\.C\.\s*No\.\s*\d+/\d+"
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

        # ----------------------------------------------------
        # FILING DATE
        # ----------------------------------------------------

        filing_date = ""

        date_patterns = [

            r"\b\d{2}-\d{2}-\d{4}\b",

            r"\b\d{2}/\d{2}/\d{4}\b"
        ]

        date_match = None

        for pattern in date_patterns:

            date_match = re.search(
                pattern,
                text
            )

            if date_match:
                break

        if date_match:

            filing_date = (
                date_match.group(0)
            )

        # ----------------------------------------------------
        # CASE AGE
        # ----------------------------------------------------

        case_age = 0

        parsed_filing_date = None

        try:

            if filing_date:

                if "-" in filing_date:

                    parsed_filing_date = (
                        datetime.strptime(
                            filing_date,
                            "%d-%m-%Y"
                        ).date()
                    )

                elif "/" in filing_date:

                    parsed_filing_date = (
                        datetime.strptime(
                            filing_date,
                            "%d/%m/%Y"
                        ).date()
                    )

                if parsed_filing_date:

                    case_age = (
                        datetime.today().date()
                        -
                        parsed_filing_date
                    ).days

                    case_age = max(
                        0,
                        case_age
                    )

        except Exception:

            case_age = 0

        # ----------------------------------------------------
        # HEARING GAP
        # ----------------------------------------------------

        hearing_gap = 0

        all_dates = re.findall(
            r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
            text
        )

        parsed_dates = []

        for date_string in all_dates:

            try:

                if "-" in date_string:

                    parsed_date = (
                        datetime.strptime(
                            date_string,
                            "%d-%m-%Y"
                        ).date()
                    )

                else:

                    parsed_date = (
                        datetime.strptime(
                            date_string,
                            "%d/%m/%Y"
                        ).date()
                    )

                parsed_dates.append(
                    parsed_date
                )

            except Exception:

                continue

        if len(parsed_dates) >= 2:

            hearing_gap = abs(
                (
                    parsed_dates[-1]
                    -
                    parsed_dates[-2]
                ).days
            )

        # ----------------------------------------------------
        # DEFAULT FILING DATE
        # ----------------------------------------------------

        if parsed_filing_date:

            filing_year = (
                parsed_filing_date.year
            )

            filing_month = (
                parsed_filing_date.month
            )

        else:

            filing_year = (
                datetime.today().year
            )

            filing_month = (
                datetime.today().month
            )

        # ====================================================
        # VERIFY EXTRACTED DATA
        # ====================================================

        st.subheader(
            "✅ Verify Extracted Information"
        )

        st.caption(
            "OCR can make mistakes. Verify the information "
            "before adding the case to PostgreSQL."
        )

        district = st.text_input(
            "District",
            value=district
        )

        court_name = st.text_input(
            "Court Name",
            value=court_name
        )

        registration_number = st.text_input(
            "Registration Number",
            value=registration_number
        )

        stage = st.text_input(
            "Stage of Case",
            value=stage
        )

        case_status = st.selectbox(
            "Case Status",
            [
                "Pending",
                "Disposed"
            ],
            index=(
                1
                if case_status == "Disposed"
                else 0
            )
        )

        filing_date_input = st.text_input(
            "Filing Date",
            value=filing_date
        )

        # ----------------------------------------------------
        # CASE DETAILS
        # ----------------------------------------------------

        st.subheader(
            "🧠 ML Information"
        )

        case_type = st.text_input(
            "Case Type",
            value="Criminal"
        )

        act_name = st.text_input(
            "Act Name",
            value="IPC"
        )

        section = st.text_input(
            "Section",
            value="Unknown"
        )

        police_station = st.text_input(
            "Police Station",
            value="Unknown"
        )

        case_age = st.number_input(
            "Case Age (Days)",
            min_value=0,
            value=int(case_age),
            step=1
        )

        hearing_gap = st.number_input(
            "Average Hearing Gap (Days)",
            min_value=0,
            value=int(hearing_gap),
            step=1
        )

        # ----------------------------------------------------
        # ADD EXTRACTED CASE
        # ----------------------------------------------------

        if st.button(
            "🤖 Predict and Add Extracted Case",
            type="primary",
            key="add_extracted_case"
        ):

            # ------------------------------------------------
            # PARSE FILING DATE
            # ------------------------------------------------

            parsed_filing_date = None

            try:

                if "-" in filing_date_input:

                    parsed_filing_date = (
                        datetime.strptime(
                            filing_date_input,
                            "%d-%m-%Y"
                        ).date()
                    )

                elif "/" in filing_date_input:

                    parsed_filing_date = (
                        datetime.strptime(
                            filing_date_input,
                            "%d/%m/%Y"
                        ).date()
                    )

            except Exception:

                parsed_filing_date = None

            if parsed_filing_date:

                filing_year = (
                    parsed_filing_date.year
                )

                filing_month = (
                    parsed_filing_date.month
                )

            else:

                st.warning(
                    "Filing date could not be parsed. "
                    "Current year/month will be used."
                )

                filing_year = (
                    datetime.today().year
                )

                filing_month = (
                    datetime.today().month
                )

            # ------------------------------------------------
            # DISTRICT WORKLOAD
            # ------------------------------------------------

            district_workload = (
                get_district_workload(
                    district
                )
            )

            # ------------------------------------------------
            # ML INPUT
            # ------------------------------------------------

            ml_case = {

                "district_name":
                    district.strip(),

                "district_workload_level":
                    district_workload,

                "court_name":
                    court_name.strip(),

                "case_type":
                    case_type.strip(),

                "stage_of_case":
                    stage.strip(),

                "act_name":
                    act_name.strip(),

                "section":
                    section.strip(),

                "police_station":
                    police_station.strip(),

                "case_age_days":
                    float(case_age),

                "hearing_gap_days":
                    float(hearing_gap),

                "filing_year":
                    int(filing_year),

                "filing_month":
                    int(filing_month)
            }

            # ------------------------------------------------
            # CALL ML
            # ------------------------------------------------

            with st.spinner(
                "Running XGBoost prediction and SHAP explanation..."
            ):

                ml_result = get_ml_prediction(
                    ml_case
                )

            if ml_result:

                # ------------------------------------------------
                # DISPLAY PREDICTION
                # ------------------------------------------------

                display_ml_result(
                    ml_result
                )

                # ------------------------------------------------
                # SAVE TO POSTGRESQL
                # ------------------------------------------------

                case_to_save = {

                    "cnr_number":
                        None,

                    "registration_number":
                        registration_number.strip()
                        if registration_number
                        else None,

                    "district_name":
                        district.strip(),

                    "court_name":
                        court_name.strip(),

                    "case_type":
                        case_type.strip(),

                    "act_name":
                        act_name.strip(),

                    "section":
                        section.strip(),

                    "police_station":
                        police_station.strip(),

                    "filing_date":
                        parsed_filing_date,

                    "first_hearing_date":
                        None,

                    "decision_date":
                        None,

                    "case_status":
                        case_status,

                    "stage_of_case":
                        stage.strip(),

                    "disposal_type":
                        None,

                    "source_type":
                        "OCR"
                }

                case_id = (
                    save_case_to_database(
                        case_to_save
                    )
                )

                if case_id:

                    st.success(
                        "PDF case added successfully "
                        "to PostgreSQL."
                    )

                    st.info(
                        f"Case ID: {case_id}"
                    )

                    st.info(
                        "No CSV file was created or modified."
                    )

                    st.cache_data.clear()


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🗄 PostgreSQL Database"
)

if not df.empty:

    st.write(
        f"Current cases available in PostgreSQL: "
        f"**{len(df)}**"
    )

    preview_columns = [

        "case_id",
        "cnr_number",
        "registration_number",
        "district_name",
        "court_name",
        "case_type",
        "case_status",
        "stage_of_case",
        "filing_date"
    ]

    available_columns = [
        column
        for column in preview_columns
        if column in df.columns
    ]

    if available_columns:

        st.dataframe(
            df[
                available_columns
            ].head(20),
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "No PostgreSQL case records available."
    )