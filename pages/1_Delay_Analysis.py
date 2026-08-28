import streamlit as st
import pandas as pd
import plotly.express as px

from database.repository import get_cases
from ml.predict import predict_dataframe


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Delay & Risk Analysis",
    page_icon="⚠️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚠️ Delay & Risk Analysis"
)

st.markdown(
    "### Identifying Delay Patterns and High-Risk Cases"
)

st.caption(
    "Detailed analysis of case delays, hearing gaps, "
    "predicted risk, and priority cases."
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
# ML PREDICTIONS
# ============================================================

try:

    df = predict_dataframe(df)

except Exception as exc:

    st.error(
        "Unable to generate XGBoost predictions."
    )

    st.code(
        "python -m ml.train"
    )

    st.exception(exc)

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df["district_name"] = (
    df["district_name"]
    .fillna("Unknown")
    .astype(str)
)

df["stage_of_case"] = (
    df["stage_of_case"]
    .fillna("Unknown")
    .astype(str)
)

df["court_name"] = (
    df["court_name"]
    .fillna("Unknown")
    .astype(str)
)

df["predicted_risk"] = (
    df["predicted_risk"]
    .fillna("Unknown")
    .astype(str)
)


df["predicted_duration_years"] = pd.to_numeric(
    df["predicted_duration_years"],
    errors="coerce"
).fillna(0)


df["hearing_gap_days"] = pd.to_numeric(
    df.get("hearing_gap_days", 0),
    errors="coerce"
).fillna(0)


df["case_age_days"] = pd.to_numeric(
    df.get("case_age_days", 0),
    errors="coerce"
).fillna(0)


# ============================================================
# HIGH-RISK PROBABILITY
# ============================================================

if "risk_probability_high" in df.columns:

    df["high_risk_probability"] = pd.to_numeric(

        df["risk_probability_high"],

        errors="coerce"

    ).fillna(0)

else:

    df["high_risk_probability"] = 0.0


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🔎 Analysis Filters"
)


districts = sorted(
    df["district_name"]
    .unique()
    .tolist()
)


selected_districts = st.sidebar.multiselect(

    "District",

    districts,

    default=districts

)


selected_risks = st.sidebar.multiselect(

    "Predicted Risk Level",

    [
        "Low",
        "Medium",
        "High"
    ],

    default=[
        "Low",
        "Medium",
        "High"
    ]

)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df[

    df["district_name"].isin(
        selected_districts
    )

    &

    df["predicted_risk"].isin(
        selected_risks
    )

].copy()


if filtered.empty:

    st.warning(
        "No cases match the selected filters."
    )

    st.stop()


# ============================================================
# DELAY OVERVIEW
# ============================================================

st.subheader(
    "📊 Delay & Risk Overview"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Cases",
        f"{len(filtered):,}"
    )


with c2:

    high_risk_count = int(

        (
            filtered["predicted_risk"]
            == "High"
        ).sum()

    )

    st.metric(
        "High-Risk Cases",
        f"{high_risk_count:,}"
    )


with c3:

    avg_duration = (

        filtered[
            "predicted_duration_years"
        ].mean()

    )

    st.metric(
        "Avg. Predicted Duration",
        f"{avg_duration:.2f} years"
    )


with c4:

    avg_gap = (

        filtered[
            "hearing_gap_days"
        ].mean()

    )

    st.metric(
        "Avg. Hearing Gap",
        f"{avg_gap:.0f} days"
    )


# ============================================================
# 1. HIGH-RISK CASES BY DISTRICT
# ============================================================

st.subheader(
    "🚨 High-Risk Cases by District"
)


district_risk = (

    filtered

    .assign(

        high_risk=lambda x:

        (
            x["predicted_risk"]
            == "High"
        ).astype(int)

    )

    .groupby(
        "district_name"
    )

    .agg(

        high_risk_cases=(
            "high_risk",
            "sum"
        )

    )

    .reset_index()

)


district_risk.columns = [
    "District",
    "High-Risk Cases"
]


district_risk = district_risk.sort_values(

    "High-Risk Cases",

    ascending=False

)


fig = px.bar(

    district_risk,

    x="District",

    y="High-Risk Cases",

    text="High-Risk Cases",

    title="High-Risk Cases Requiring Greater Attention by District"

)


fig.update_traces(
    textposition="outside"
)


fig.update_layout(

    xaxis_title="District",

    yaxis_title="Number of High-Risk Cases"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# 2. RISK BY CASE STAGE
# ============================================================

st.subheader(
    "⚠️ Predicted Risk by Current Case Stage"
)


stage_risk = (

    filtered

    .groupby(

        [
            "stage_of_case",
            "predicted_risk"
        ]

    )

    .size()

    .reset_index(
        name="Cases"
    )

)


fig = px.bar(

    stage_risk,

    x="stage_of_case",

    y="Cases",

    color="predicted_risk",

    barmode="group",

    text="Cases",

    title="Predicted Risk Distribution Across Case Stages"

)


fig.update_layout(

    xaxis_title="Current Case Stage",

    yaxis_title="Number of Cases",

    legend_title="Predicted Risk"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# 3. HEARING GAP ANALYSIS
# ============================================================

st.subheader(
    "📅 Hearing Gap Distribution by District"
)


fig = px.box(

    filtered,

    x="district_name",

    y="hearing_gap_days",

    title="Distribution of Time Between Hearings"

)


fig.update_layout(

    xaxis_title="District",

    yaxis_title="Hearing Gap (Days)"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# 4. PRIORITY CASES
# ============================================================

st.subheader(
    "🚨 Highest-Priority Cases"
)


priority_cases = (

    filtered

    .sort_values(

        [
            "high_risk_probability",
            "predicted_duration_years"
        ],

        ascending=False

    )

    .head(20)

)


display_columns = [

    "cnr_number",

    "district_name",

    "court_name",

    "stage_of_case",

    "predicted_risk",

    "high_risk_probability",

    "predicted_duration_years",

    "case_age_days",

    "hearing_gap_days"

]


available_columns = [

    column

    for column in display_columns

    if column in priority_cases.columns

]


display_df = priority_cases[
    available_columns
].copy()


# ============================================================
# FRIENDLY COLUMN NAMES
# ============================================================

display_df = display_df.rename(

    columns={

        "cnr_number":
            "CNR Number",

        "district_name":
            "District",

        "court_name":
            "Court",

        "stage_of_case":
            "Current Stage",

        "predicted_risk":
            "Predicted Risk",

        "high_risk_probability":
            "High-Risk Probability (%)",

        "predicted_duration_years":
            "Predicted Duration (Years)",

        "case_age_days":
            "Case Age (Days)",

        "hearing_gap_days":
            "Hearing Gap (Days)"

    }

)


if "High-Risk Probability (%)" in display_df.columns:

    display_df[
        "High-Risk Probability (%)"
    ] = (

        display_df[
            "High-Risk Probability (%)"
        ]

        * 100

    ).round(2)


if "Predicted Duration (Years)" in display_df.columns:

    display_df[
        "Predicted Duration (Years)"
    ] = (

        display_df[
            "Predicted Duration (Years)"
        ]

        .round(2)

    )


st.dataframe(

    display_df,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# FOOTER
# ============================================================

st.caption(

    "Risk predictions are generated by the trained XGBoost "
    "models using case data retrieved from PostgreSQL. "
    "Predictions are analytical estimates and not guaranteed outcomes."

)