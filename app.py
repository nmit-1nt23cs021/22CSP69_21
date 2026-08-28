import streamlit as st
import pandas as pd
import plotly.express as px

from database.repository import get_cases
from ml.predict import predict_dataframe


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Judicial Intelligence Dashboard",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚖️ Judicial Intelligence Dashboard"
)

st.markdown(
    "### Karnataka Judiciary — Data Analytics & ML-Based Case Intelligence"
)

st.caption(
    "Interactive analysis of judicial case data using "
    "PostgreSQL and XGBoost-based predictions."
)


# ============================================================
# LOAD DATA FROM POSTGRESQL
# ============================================================

try:

    df = get_cases()

except Exception as exc:

    st.error(
        "PostgreSQL connection failed."
    )

    st.exception(exc)

    st.stop()


if df.empty:

    st.warning(
        "No cases available in PostgreSQL."
    )

    st.stop()


# ============================================================
# XGBOOST PREDICTIONS
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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Dashboard Filters"
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
# OVERVIEW KPIs
# ============================================================

st.subheader(
    "📌 Dashboard Overview"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Total Cases",
        f"{len(filtered):,}"
    )


with c2:

    high_risk = int(
        (
            filtered["predicted_risk"]
            == "High"
        ).sum()
    )

    st.metric(
        "High-Risk Cases",
        f"{high_risk:,}"
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
# 1. CASE DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Case Distribution by District"
)


district_cases = (

    filtered[
        "district_name"
    ]

    .value_counts()

    .reset_index()
)


district_cases.columns = [
    "District",
    "Cases"
]


fig = px.bar(

    district_cases,

    x="District",

    y="Cases",

    text="Cases",

    title="Number of Cases by District"

)


fig.update_traces(
    textposition="outside"
)


fig.update_layout(

    xaxis_title="District",

    yaxis_title="Number of Cases"

)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 2. OVERALL RISK DISTRIBUTION
# ============================================================

st.subheader(
    "🤖 Predicted Case Risk Distribution"
)


risk = (

    filtered[
        "predicted_risk"
    ]

    .value_counts()

    .reindex(
        [
            "Low",
            "Medium",
            "High"
        ],
        fill_value=0
    )

    .reset_index()
)


risk.columns = [
    "Risk Level",
    "Cases"
]


fig = px.pie(

    risk,

    names="Risk Level",

    values="Cases",

    title="Distribution Across Predicted Risk Levels"

)


fig.update_traces(

    textposition="inside",

    textinfo="percent+label"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# 3. PREDICTED DURATION BY DISTRICT
# ============================================================

st.subheader(
    "📅 Predicted Case Duration by District"
)


duration = (

    filtered

    .groupby(
        "district_name"
    )[

        "predicted_duration_years"

    ]

    .mean()

    .reset_index()
)


duration.columns = [
    "District",
    "Predicted Duration (Years)"
]


duration = duration.sort_values(

    "Predicted Duration (Years)",

    ascending=False

)


fig = px.bar(

    duration,

    x="District",

    y="Predicted Duration (Years)",

    text="Predicted Duration (Years)",

    title="Average Predicted Case Duration by District"

)


fig.update_traces(

    texttemplate="%{text:.2f} years",

    textposition="outside"

)


fig.update_layout(

    xaxis_title="District",

    yaxis_title="Predicted Duration (Years)"

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Predictions are generated by trained XGBoost models "
    "using case data retrieved from PostgreSQL."
)