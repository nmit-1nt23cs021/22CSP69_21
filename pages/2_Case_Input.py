import os
import re
import shutil
from datetime import datetime, date

import pandas as pd
import requests
import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes

from database.repository import get_cases, add_case


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Case Registration & ML Prediction",
    page_icon="📥",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

ML_API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:8000/predict")

# These paths work with the original Windows setup, while also
# allowing the page to run on Linux/Render when the executables
# are available in PATH.
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
POPPLER_PATH = os.getenv(
    "POPPLER_PATH",
    r"C:\poppler\poppler-24.08.0\Library\bin",
)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


def get_poppler_path():
    if os.path.isdir(POPPLER_PATH):
        return POPPLER_PATH
    if shutil.which("pdftoppm"):
        return None
    return None


# ============================================================
# POSTGRESQL
# ============================================================

@st.cache_data(ttl=30)
def load_cases():
    return get_cases()


try:
    df = load_cases()
except Exception as exc:
    st.error("Unable to connect to PostgreSQL.")
    st.exception(exc)
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_districts():
    if "district_name" not in df.columns:
        return []
    values = (
        df["district_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted([x for x in values.unique() if x and x.lower() != "unknown"])


def get_stages():
    if "stage_of_case" not in df.columns:
        return []
    values = (
        df["stage_of_case"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted([x for x in values.unique() if x and x.lower() != "unknown"])


def get_workload(district):
    if "district_workload_level" not in df.columns:
        return "Medium"

    rows = df[
        df["district_name"].astype(str).str.strip().str.lower()
        == str(district).strip().lower()
    ]

    if rows.empty:
        return "Medium"

    values = (
        rows["district_workload_level"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return str(values.mode().iloc[0]) if not values.empty else "Medium"


def safe_date(value):
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.date()
    except Exception:
        return None


def calculate_features(filing_date, first_hearing_date, hearing_dates=None):
    today = pd.Timestamp.today().normalize()
    filing = pd.Timestamp(filing_date)
    first_hearing = pd.Timestamp(first_hearing_date)

    case_age_days = max(0, int((today - filing).days))
    first_hearing_delay_days = max(0, int((first_hearing - filing).days))

    # IMPORTANT:
    # The ML training pipeline defines hearing_gap_days as the average
    # interval between consecutive hearings. The old page incorrectly
    # calculated it as TODAY - FIRST_HEARING_DATE.
    if hearing_dates:
        ordered = sorted(set(hearing_dates))
        gaps = [
            (b - a).days
            for a, b in zip(ordered, ordered[1:])
            if (b - a).days >= 0
        ]
        hearing_gap_days = int(round(sum(gaps) / len(gaps))) if gaps else 0
    else:
        hearing_gap_days = 0

    return {
        "case_age_days": case_age_days,
        "first_hearing_delay_days": first_hearing_delay_days,
        "hearing_gap_days": hearing_gap_days,
        "filing_year": filing.year,
        "filing_month": filing.month,
    }


def get_ml_prediction(case_data):
    try:
        response = requests.post(
            ML_API_URL,
            json=case_data,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "ML backend is not running. Start it with:\n\n"
            "uvicorn backend.main:app --reload --port 8000"
        )
        return None

    except requests.exceptions.Timeout:
        st.error("The ML backend took too long to respond.")
        return None

    except requests.exceptions.HTTPError:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        st.error(f"ML backend returned an error: {detail}")
        return None

    except requests.exceptions.RequestException as exc:
        st.error(f"ML prediction error: {exc}")
        return None


def display_ml_result(result):
    if not result:
        return

    st.divider()
    st.subheader("🤖 XGBoost Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = float(result.get("predicted_duration_years", 0))
        st.metric("Predicted Duration", f"{duration:.2f} years")

    with col2:
        st.metric("Predicted Delay Risk", result.get("predicted_risk", "Unknown"))

    probabilities = result.get("risk_probabilities", {}) or {}

    with col3:
        high_probability = float(probabilities.get("High", 0))
        st.metric("High-Risk Probability", f"{high_probability * 100:.1f}%")

    st.subheader("📊 Risk Probabilities")
    for category, probability in probabilities.items():
        probability = float(probability)
        st.write(f"**{category}:** {probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    explanation = result.get("explanation", []) or []
    st.subheader("🔍 SHAP Explanation")

    if explanation:
        explanation_df = pd.DataFrame(explanation)
        if "shap_value" in explanation_df.columns:
            explanation_df["shap_value"] = pd.to_numeric(
                explanation_df["shap_value"], errors="coerce"
            ).round(4)
        st.dataframe(explanation_df, use_container_width=True, hide_index=True)
        st.caption("SHAP values show which model features contributed most to the prediction.")
    else:
        st.info("No SHAP explanation was returned.")


# ============================================================
# OCR HELPERS
# ============================================================

DATE_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b")
DATE_PREFIX_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]$")
YEAR_RE = re.compile(r"^\d{4}$")
MONTH_RE = re.compile(
    r"\b(\d{1,2})[- ](January|February|March|April|May|June|July|August|"
    r"September|October|November|December)[- ](\d{4})\b",
    re.IGNORECASE,
)


def parse_numeric_date(value):
    value = clean_text(value).replace("/", "-")
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def parse_named_date(value):
    match = MONTH_RE.search(value)
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{match.group(1)} {match.group(2)} {match.group(3)}",
            "%d %B %Y",
        ).date()
    except ValueError:
        return None


def ocr_page(image):
    """Return normal OCR text and word-level OCR data."""
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, config=config)
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    return text, data


def data_words(data):
    words = []
    for i, raw in enumerate(data.get("text", [])):
        word = clean_text(raw)
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0
        words.append({
            "text": word,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "conf": conf,
        })
    return words


def date_candidates_from_data(data):
    """Recover full dates even when Tesseract splits dd-mm-yyyy over two lines."""
    words = data_words(data)
    candidates = []

    # Normal full date tokens.
    for w in words:
        match = DATE_RE.search(w["text"])
        if match:
            parsed = parse_numeric_date(match.group(0))
            if parsed:
                candidates.append((parsed, w["x"], w["y"], w["w"], w["h"]))

    # Reconstruct tokens such as "21-11-" + "2022".
    prefixes = [w for w in words if DATE_PREFIX_RE.match(w["text"])]
    years = [w for w in words if YEAR_RE.match(w["text"])]

    for p in prefixes:
        best = None
        best_score = 10**9
        for y in years:
            if abs(y["x"] - p["x"]) > 80:
                continue
            dy = y["y"] - (p["y"] + p["h"])
            if -15 <= dy <= 80:
                score = abs(y["x"] - p["x"]) + abs(dy)
                if score < best_score:
                    best = y
                    best_score = score
        if best:
            parsed = parse_numeric_date(p["text"] + best["text"])
            if parsed:
                candidates.append((parsed, p["x"], p["y"], p["w"], p["h"] + best["h"]))

    # De-duplicate same date/position.
    unique = {}
    for item in candidates:
        key = (item[0], round(item[1] / 10), round(item[2] / 10))
        unique[key] = item
    return list(unique.values())


def hearing_dates_from_pages(page_data, page_width):
    """
    eCourts pages 2+ contain a two-column date table:
    Business On Date is around x=780 on the supplied PDF and
    Hearing Date is around x=1030 on a 1700px OCR image.
    We therefore use the right-hand date column instead of all dates.
    """
    dates = []
    hearing_x_min = page_width * 0.56

    for data in page_data:
        for parsed, x, y, _, _ in date_candidates_from_data(data):
            # Ignore the page timestamp at the very top.
            if y < page_width * 0.08:
                continue
            if x >= hearing_x_min:
                dates.append(parsed)

    return sorted(set(dates))


def extract_case_fields(page_texts, page_data, page_sizes):
    """Extract fields from the eCourts case-status layout.

    The PDF contains a structured first-page case row.  OCR sometimes splits
    dates across two tokens (for example ``14-February-`` + ``2023``) and can
    also slightly misspell words such as ``ACCUSSED``.  This function therefore
    uses OCR coordinates for the first-page table and text matching for the
    labelled sections on page 2.
    """

    first_page = page_texts[0] if page_texts else ""
    second_page = page_texts[1] if len(page_texts) > 1 else ""
    first_lower = first_page.lower()

    fields = {
        "district": "Unknown",
        "court_name": "Unknown",
        "case_type": "Unknown",
        "cnr_number": "",
        "registration_number": "",
        "filing_date": None,
        "first_hearing_date": None,
        "decision_date": None,
        "case_status": "PENDING",
        "stage": "Unknown",
        "act_name": "Unknown",
        "section": "Unknown",
        "police_station": "Unknown",
    }

    # ---------------- DISTRICT ----------------
    district_map = {
        "kalaburagi": "Kalaburagi",
        "gulbarga": "Kalaburagi",
        "mysuru": "Mysuru",
        "mysore": "Mysuru",
        "bengaluru": "Bengaluru Urban",
        "bangalore": "Bengaluru Urban",
        "tumakuru": "Tumakuru",
        "tumkur": "Tumakuru",
        "belagavi": "Belagavi",
    }
    for key, value in district_map.items():
        if key in first_lower:
            fields["district"] = value
            break

    # ---------------- COURT ----------------
    court_lines = [
        clean_text(line)
        for line in first_page.splitlines()
        if "COURT" in line.upper()
    ]
    preferred = [
        line for line in court_lines
        if "JUDGE" in line.upper() and "COURT" in line.upper()
    ]
    if preferred:
        fields["court_name"] = preferred[0]
    elif court_lines:
        non_generic = [
            line for line in court_lines
            if "DISTRICT COURT" not in line.upper()
        ]
        if non_generic:
            fields["court_name"] = non_generic[0]

    # ---------------- FIRST PAGE TABLE ROW ----------------
    # Locate the row containing the case type.  In both supplied PDFs the
    # filing number, registration number and CNR sit on the same OCR row.
    words0 = data_words(page_data[0]) if page_data else []
    if page_sizes:
        first_width, first_height = page_sizes[0]
        case_row_words = [
            w for w in words0
            if first_height * 0.69 <= w["y"] <= first_height * 0.78
        ]
    else:
        first_width, first_height = 1700, 2200
        case_row_words = words0

    # ---------------- CNR ----------------
    cnr_candidates = [
        w["text"] for w in case_row_words
        if re.fullmatch(r"[A-Z]{2,4}\d{10,16}", w["text"], re.IGNORECASE)
    ]
    if cnr_candidates:
        fields["cnr_number"] = cnr_candidates[-1].upper()
    else:
        cnr_match = re.search(r"\b[A-Z]{2,4}\d{10,16}\b", first_page, re.IGNORECASE)
        if cnr_match:
            fields["cnr_number"] = cnr_match.group(0).upper()

    # ---------------- REGISTRATION NUMBER ----------------
    # There are TWO nnnnn/yyyy values in the first-page row:
    #   Filing Number       Registration Number
    # Always use the second one, immediately before the registration date.
    row_reg_numbers = [
        w for w in case_row_words
        if re.fullmatch(r"\d{1,6}/\d{4}", w["text"])
    ]
    if len(row_reg_numbers) >= 2:
        fields["registration_number"] = row_reg_numbers[1]["text"]
    elif row_reg_numbers:
        # Safe fallback: use the number closest to the CNR on the same row.
        fields["registration_number"] = row_reg_numbers[-1]["text"]

    # ---------------- CASE TYPE ----------------
    # Prefer the explicit Criminal/Civil value in the case-type row.
    # OCR can distort the surrounding table text, so check Criminal first.
    if re.search(r"\bCRIMINAL\b", first_page, re.IGNORECASE):
        fields["case_type"] = "Criminal"
    elif re.search(r"\bCIVIL\b", first_page, re.IGNORECASE):
        fields["case_type"] = "Civil"
    else:
        case_type_match = re.search(
            r"\b(Original|Miscellaneous|MISC|C\.C\.)\b",
            first_page,
            re.IGNORECASE,
        )
        if case_type_match:
            fields["case_type"] = case_type_match.group(1).title()

    # ---------------- FILING DATE ----------------
    # Find the date in the Filing Date column (x between the filing-number
    # and registration-number columns).  eCourts OCR can split 24-12-2022
    # into 24-12- + 2022, so reconstruct it from nearby words.
    if page_data and page_sizes:
        width, height = page_sizes[0]
        candidates = date_candidates_from_data(page_data[0])

        filing_candidates = [
            item for item in candidates
            if 0.30 <= item[1] / width <= 0.46
            and 0.68 <= item[2] / height <= 0.82
        ]
        if filing_candidates:
            # Prefer the candidate closest to the case row.
            filing_candidates.sort(key=lambda item: abs(item[2] - 1600))
            fields["filing_date"] = filing_candidates[0][0]

        # If the OCR reconstructed-date helper missed the split token,
        # explicitly join a prefix and year in the filing-date x range.
        if fields["filing_date"] is None:
            prefix_words = [
                w for w in case_row_words
                if 0.30 <= w["x"] / width <= 0.46
                and DATE_PREFIX_RE.match(w["text"])
            ]
            year_words = [
                w for w in words0
                if YEAR_RE.match(w["text"])
                and 0.30 <= w["x"] / width <= 0.46
                and 1570 <= w["y"] <= 1655
            ]
            for prefix in prefix_words:
                nearby = [
                    y for y in year_words
                    if abs(y["x"] - prefix["x"]) < 100
                    and -15 <= y["y"] - (prefix["y"] + prefix["h"]) <= 80
                ]
                if nearby:
                    parsed = parse_numeric_date(prefix["text"] + nearby[0]["text"])
                    if parsed:
                        fields["filing_date"] = parsed
                        break

    # Text fallback: the first date in the case row is the filing date.
    if fields["filing_date"] is None:
        row_dates = []
        for item in date_candidates_from_data(page_data[0]) if page_data else []:
            if 0.30 <= item[1] / page_sizes[0][0] <= 0.46:
                row_dates.append(item[0])
        if row_dates:
            fields["filing_date"] = row_dates[0]

    # ---------------- FIRST HEARING DATE ----------------
    # Do NOT take the first date on the page.  The case table has a labelled
    # First Hearing column, and OCR may split a named date over two lines.
    if page_data and page_sizes:
        width, height = page_sizes[0]
        words = words0

        # First-hearing date is in the leftmost date column below the
        # 'First Hearing / Date' header.
        fh_tokens = [
            w for w in words
            if w["x"] < width * 0.23 and w["y"] > height * 0.83
        ]

        # Named date: e.g. 14-February- + 2023.
        for i, w in enumerate(fh_tokens):
            if re.fullmatch(
                r"\d{1,2}-(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)-",
                w["text"],
                re.IGNORECASE,
            ):
                for y in fh_tokens[i + 1:]:
                    if abs(y["x"] - w["x"]) < 90 and 0 <= y["y"] - w["y"] <= 80:
                        parsed = parse_named_date(f"{w['text']}{y['text']}")
                        if parsed:
                            fields["first_hearing_date"] = parsed
                            break
                if fields["first_hearing_date"]:
                    break

        # Numeric date fallback in the same column.
        if fields["first_hearing_date"] is None:
            fh_dates = [
                item for item in date_candidates_from_data(page_data[0])
                if item[1] < width * 0.23 and item[2] > height * 0.83
            ]
            if fh_dates:
                fields["first_hearing_date"] = sorted(
                    fh_dates, key=lambda item: item[2]
                )[0][0]

    # Text fallback: specifically search after the First Hearing header.
    if fields["first_hearing_date"] is None:
        named_matches = re.findall(
            r"(\d{1,2}-[A-Za-z]+-\s*\d{4})",
            first_page,
        )
        for raw in named_matches:
            parsed = parse_named_date(raw)
            if parsed:
                fields["first_hearing_date"] = parsed
                break

    # ---------------- STATUS ----------------
    fields["case_status"] = (
        "DISPOSED"
        if re.search(r"\b(disposed|case disposed|closed)\b", first_page, re.IGNORECASE)
        else "PENDING"
    )

    # ---------------- CURRENT STAGE ----------------
    # OCR on these PDFs sometimes returns ACCUSSED instead of ACCUSED.
    # Match the stable 'APPEARANCE OF' phrase and tolerate the OCR typo.
    # First try the actual Stage-of-Case column using OCR coordinates.
    # The OCR reading order places the court-name column between
    # "APPEARANCE OF" and "ACCUSSED", so a plain text regex is unreliable.
    if page_data and page_sizes:
        width, height = page_sizes[0]
        stage_words = [
            w for w in words0
            if w["x"] >= width * 0.45 and w["y"] >= height * 0.84
        ]
        appearance_words = [
            w for w in stage_words
            if re.fullmatch(r"APPEARANCE", w["text"], re.IGNORECASE)
        ]
        accused_words = [
            w for w in stage_words
            if re.fullmatch(r"ACCUS+ED", w["text"], re.IGNORECASE)
        ]
        if appearance_words and accused_words:
            for a in appearance_words:
                nearby = [
                    b for b in accused_words
                    if 0 <= b["y"] - a["y"] <= 100
                    and abs(b["x"] - a["x"]) <= width * 0.18
                ]
                if nearby:
                    fields["stage"] = "APPEARANCE OF ACCUSED"
                    break

    # Text fallback for less noisy OCR layouts.
    if fields["stage"] == "Unknown":
        normalized_stage_text = re.sub(r"ACCUS+ED", "ACCUSED", first_page, flags=re.IGNORECASE)
        stage_patterns = [
            (r"APPEARANCE\s+OF\s+ACCUSED", "APPEARANCE OF ACCUSED"),
            (r"FRAMING\s+OF\s+CHARGE\s+OR\s+PLEA", "FRAMING OF CHARGE OR PLEA"),
            (r"RE-?ISSUE\s+SUMMONS", "RE-ISSUE SUMMONS"),
            (r"EVIDENCE", "EVIDENCE"),
            (r"ARGUMENT", "ARGUMENT"),
            (r"JUDGMENT", "JUDGMENT"),
            (r"SUMMONS", "SUMMONS"),
            (r"NBW", "NBW"),
            (r"STEPS", "STEPS"),
        ]
        for pattern, label in stage_patterns:
            if re.search(pattern, normalized_stage_text, re.IGNORECASE):
                fields["stage"] = label
                break

    # ---------------- ACT / SECTION ----------------
    act_match = re.search(
        r"Under Act\(s\).*?Under Section\(s\).*?\n\s*([A-Za-z][^\n]+?)\s+([0-9][^\n]+)",
        second_page,
        re.IGNORECASE | re.DOTALL,
    )
    if act_match:
        fields["act_name"] = clean_text(act_match.group(1)).upper()
        fields["section"] = clean_text(act_match.group(2)).upper()

    if fields["act_name"] == "Unknown":
        act_match = re.search(
            r"\b(INDIAN PENAL CODE|INDIAN PENAL|IPC)\b",
            second_page,
            re.IGNORECASE,
        )
        if act_match:
            fields["act_name"] = clean_text(act_match.group(1)).upper()

    if fields["section"] == "Unknown":
        section_match = re.search(
            r"\b(\d{2,4}\s*,\s*\d{2,4}(?:\s*,\s*\d{2,4})?(?:\s*,\s*\d{2,4})?\s*R?/?W?\s*34IPC)\b",
            second_page,
            re.IGNORECASE,
        )
        if section_match:
            fields["section"] = re.sub(r"\s+", " ", section_match.group(1)).upper()
        else:
            section_match = re.search(
                r"\b(\d{2,4}\s*,\s*\d{2,4})\s*IPC\b",
                second_page,
                re.IGNORECASE,
            )
            if section_match:
                fields["section"] = f"{section_match.group(1).replace(' ', '')}IPC"

    # ---------------- POLICE STATION ----------------
    police_match = re.search(
        r"Police Station\s+FIR Number\s+Year.*?\n\s*([A-Za-z0-9 ._-]+?)\s+\d+\s+\d{4}",
        second_page,
        re.IGNORECASE | re.DOTALL,
    )
    if police_match:
        fields["police_station"] = clean_text(police_match.group(1)).upper()
    else:
        # Prefer a station name ending in PS; this handles both supplied PDFs.
        police_match = re.search(r"\b([A-Z][A-Z0-9 ._-]+\s+PS)\b", second_page)
        if police_match:
            fields["police_station"] = clean_text(police_match.group(1)).upper()

    # ---------------- DECISION DATE ----------------
    if fields["case_status"] == "DISPOSED":
        all_numeric_dates = []
        for data in page_data:
            all_numeric_dates.extend(item[0] for item in date_candidates_from_data(data))
        all_numeric_dates = sorted(set(all_numeric_dates))
        if all_numeric_dates:
            fields["decision_date"] = all_numeric_dates[-1]

    # ---------------- HEARING HISTORY ----------------
    # Pages 2+ contain a table whose 'Hearing Date' column is the right-hand
    # date column. Use OCR x-coordinates so filing/next/order/process dates
    # elsewhere in the PDF do not contaminate the average gap.
    hearing_dates = hearing_dates_from_pages(
        page_data[1:],
        page_sizes[1][0] if len(page_sizes) > 1 else page_sizes[0][0],
    )

    return fields, hearing_dates


# ============================================================
# TITLE
# ============================================================

st.title("📥 Case Registration & ML Prediction")
st.markdown("### Enter a new case or extract case details from an eCourts PDF.")
st.caption(
    "Cases are stored in PostgreSQL and evaluated using XGBoost-based "
    "risk and duration predictions."
)


# ============================================================
# INPUT METHOD
# ============================================================

input_method = st.radio(
    "Choose Input Method",
    ["Manual Entry", "PDF Upload"],
    horizontal=True,
)


# ============================================================
# MANUAL ENTRY
# ============================================================

if input_method == "Manual Entry":

    st.subheader("✍️ Manual Case Entry")

    col1, col2 = st.columns(2)
    with col1:
        districts = get_districts()
        district = st.selectbox("District", districts) if districts else st.text_input("District")
    with col2:
        court_name = st.text_input("Court Name")

    col1, col2 = st.columns(2)
    with col1:
        stages = get_stages()
        stage = st.selectbox("Stage of Case", stages) if stages else st.text_input("Stage of Case")
    with col2:
        case_status = st.selectbox("Case Status", ["PENDING", "DISPOSED"])

    st.subheader("📋 Case Details")
    col1, col2 = st.columns(2)
    with col1:
        cnr_number = st.text_input("CNR Number")
        registration_number = st.text_input("Registration Number")
        case_type = st.text_input("Case Type", value="Criminal")
        act_name = st.text_input("Act Name", value="IPC")
    with col2:
        section = st.text_input("Section", value="Unknown")
        police_station = st.text_input("Police Station", value="Unknown")
        source_type = st.selectbox("Source Type", ["MANUAL"])

    st.subheader("📅 Case Dates")
    col1, col2, col3 = st.columns(3)
    with col1:
        filing_date = st.date_input("Filing Date", value=datetime.today().date())
    with col2:
        first_hearing_date = st.date_input("First Hearing Date", value=None)
    with col3:
        decision_date = st.date_input("Decision Date", value=None)

    if first_hearing_date is None:
        first_hearing_date = filing_date

    features = calculate_features(filing_date, first_hearing_date)

    st.subheader("🧮 Calculated ML Features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Case Age", f"{features['case_age_days']} days")
    c2.metric("First Hearing Delay", f"{features['first_hearing_delay_days']} days")
    c3.metric("Average Hearing Gap", f"{features['hearing_gap_days']} days")

    if st.button("➕ Add Case", type="primary", use_container_width=True):
        errors = []
        if not str(district).strip():
            errors.append("District is required.")
        if not court_name.strip():
            errors.append("Court Name is required.")
        if not str(stage).strip():
            errors.append("Stage of Case is required.")
        if not cnr_number.strip():
            errors.append("CNR Number is required.")

        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        ml_case = {
            "district_name": district,
            "district_workload_level": get_workload(district),
            "court_name": court_name,
            "case_type": case_type,
            "stage_of_case": stage,
            "act_name": act_name,
            "section": section,
            "police_station": police_station,
            **features,
            "filing_month_feature": features["filing_month"],
        }

        with st.spinner("Running XGBoost prediction..."):
            ml_result = get_ml_prediction(ml_case)
        display_ml_result(ml_result)

        case_data = {
            "cnr_number": cnr_number.strip(),
            "registration_number": registration_number.strip(),
            "district_name": str(district).strip(),
            "court_name": court_name.strip(),
            "stage_of_case": str(stage).strip(),
            "case_type": case_type.strip(),
            "act_name": act_name.strip(),
            "section": section.strip(),
            "police_station": police_station.strip(),
            "filing_date": filing_date,
            "first_hearing_date": first_hearing_date,
            "decision_date": decision_date,
            "case_status": case_status,
            "disposal_type": None,
            "source_type": source_type,
        }

        try:
            new_case_id = add_case(case_data)
            st.success("✅ Case added successfully to PostgreSQL.")
            st.info(f"Case ID: {new_case_id}")
            st.cache_data.clear()
        except Exception as exc:
            st.error("Unable to save the case to PostgreSQL.")
            st.exception(exc)


# ============================================================
# PDF UPLOAD
# ============================================================

else:

    st.subheader("📄 Upload eCourts PDF")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload an eCourts case-status PDF. The page will extract fields from the eCourts table layout.",
    )

    if uploaded_pdf:

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------
        try:
            pdf_bytes = uploaded_pdf.getvalue()
            images = convert_from_bytes(
                pdf_bytes,
                dpi=200,
                poppler_path=get_poppler_path(),
            )

            page_texts = []
            page_data = []
            page_sizes = []

            progress = st.progress(0)
            total = len(images)

            for index, image in enumerate(images):
                text, data = ocr_page(image)
                page_texts.append(text)
                page_data.append(data)
                page_sizes.append(image.size)
                progress.progress((index + 1) / total)

            progress.empty()

        except Exception as exc:
            st.error(f"OCR Error: {exc}")
            st.stop()

        text = "\n\n".join(page_texts)

        st.subheader("📑 Extracted OCR Text")
        st.text_area("OCR Output", text, height=300)

        if not text.strip():
            st.warning("No text could be extracted from the PDF.")
            st.stop()

        # ----------------------------------------------------
        # STRUCTURED EXTRACTION
        # ----------------------------------------------------
        extracted, hearing_dates = extract_case_fields(
            page_texts,
            page_data,
            page_sizes,
        )

        filing_date = extracted["filing_date"] or datetime.today().date()
        first_hearing_date = extracted["first_hearing_date"] or filing_date
        decision_date = extracted["decision_date"]

        features = calculate_features(
            filing_date,
            first_hearing_date,
            hearing_dates=hearing_dates,
        )

        # ----------------------------------------------------
        # VERIFY INFORMATION
        # ----------------------------------------------------
        st.subheader("✅ Verify Extracted Information")
        st.caption(
            "The values below are extracted from the PDF. You can correct any value before saving."
        )

        col1, col2 = st.columns(2)

        with col1:
            district = st.text_input("District", value=extracted["district"])
            court_name = st.text_input("Court Name", value=extracted["court_name"])
            cnr_number = st.text_input("CNR Number", value=extracted["cnr_number"])
            registration_number = st.text_input(
                "Registration Number",
                value=extracted["registration_number"],
            )
            stage = st.text_input("Stage of Case", value=extracted["stage"])

        with col2:
            case_status = st.selectbox(
                "Case Status",
                ["PENDING", "DISPOSED"],
                index=1 if extracted["case_status"] == "DISPOSED" else 0,
            )
            filing_date = st.date_input("Filing Date", value=filing_date)
            first_hearing_date = st.date_input(
                "First Hearing Date",
                value=first_hearing_date,
            )
            decision_date = st.date_input(
                "Decision Date",
                value=decision_date,
            )

        st.subheader("🧠 ML Information")

        col1, col2 = st.columns(2)
        with col1:
            case_type = st.text_input("Case Type", value=extracted["case_type"])
            act_name = st.text_input("Act Name", value=extracted["act_name"])
        with col2:
            section = st.text_input("Section", value=extracted["section"])
            police_station = st.text_input(
                "Police Station",
                value=extracted["police_station"],
            )

        # Recalculate from the verified values.
        features = calculate_features(
            filing_date,
            first_hearing_date,
            hearing_dates=hearing_dates,
        )

        st.subheader("🧮 Calculated ML Features")
        c1, c2, c3 = st.columns(3)
        c1.metric("Case Age", f"{features['case_age_days']} days")
        c2.metric(
            "First Hearing Delay",
            f"{features['first_hearing_delay_days']} days",
        )
        c3.metric(
            "Average Hearing Gap",
            f"{features['hearing_gap_days']} days",
        )

        if hearing_dates:
            st.caption(
                f"Detected {len(hearing_dates)} hearing dates from the Hearing Date column. "
                f"Average interval is used for the ML hearing_gap_days feature."
            )

        # ----------------------------------------------------
        # ADD EXTRACTED CASE
        # ----------------------------------------------------
        if st.button(
            "➕ Add Extracted Case",
            type="primary",
            use_container_width=True,
        ):
            errors = []

            if not district.strip():
                errors.append("District is required.")
            if not court_name.strip():
                errors.append("Court Name is required.")
            if not stage.strip():
                errors.append("Stage of Case is required.")
            if not cnr_number.strip():
                errors.append("CNR Number is required.")

            if errors:
                for error in errors:
                    st.error(error)
                st.stop()

            ml_case = {
                "district_name": district.strip(),
                "district_workload_level": get_workload(district),
                "court_name": court_name.strip(),
                "case_type": case_type.strip(),
                "stage_of_case": stage.strip(),
                "act_name": act_name.strip(),
                "section": section.strip(),
                "police_station": police_station.strip(),
                **features,
                "filing_month_feature": features["filing_month"],
            }

            with st.spinner("Running XGBoost prediction..."):
                ml_result = get_ml_prediction(ml_case)
            display_ml_result(ml_result)

            case_data = {
                "cnr_number": cnr_number.strip(),
                "registration_number": registration_number.strip(),
                "district_name": district.strip(),
                "court_name": court_name.strip(),
                "stage_of_case": stage.strip(),
                "case_type": case_type.strip(),
                "act_name": act_name.strip(),
                "section": section.strip(),
                "police_station": police_station.strip(),
                "filing_date": filing_date,
                "first_hearing_date": first_hearing_date,
                "decision_date": decision_date,
                "case_status": case_status,
                "disposal_type": None,
                "source_type": "PDF_OCR",
            }

            try:
                new_case_id = add_case(case_data)
                st.success("✅ Extracted case added successfully to PostgreSQL.")
                st.info(f"Case ID: {new_case_id}")
                st.cache_data.clear()
            except Exception as exc:
                st.error("Unable to save extracted case to PostgreSQL.")
                st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Case registration and extracted case data are stored in PostgreSQL. "
    "XGBoost provides duration and delay-risk predictions, while SHAP provides model explanations."
)
