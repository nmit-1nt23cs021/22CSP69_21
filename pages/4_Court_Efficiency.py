import streamlit as st
import pandas as pd
import plotly.express as px

from database.repository import get_cases
from ml.predict import predict_dataframe


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Court Performance",
    page_icon="🏛️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏛️ Court Performance Analysis"
)

st.markdown(
    "### Comparing court workload, predicted delay risk and case duration"
)

st.caption(
    "Court-level performance is analyzed using PostgreSQL case data "
    "and XGBoost predictions."
)


# ============================================================
# LOAD DATA
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
        "XGBoost models are not available."
    )

    st.code(
        "python -m ml.train"
    )

    st.exception(exc)

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

for column in [
    "court_name",
    "district_name",
    "stage_of_case",
    "predicted_risk"
]:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )


df["predicted_duration_years"] = pd.to_numeric(
    df["predicted_duration_years"],
    errors="coerce"
).fillna(0)


if "risk_probability_high" in df.columns:

    df["high_risk_probability"] = pd.to_numeric(
        df["risk_probability_high"],
        errors="coerce"
    ).fillna(0)

else:

    df["high_risk_probability"] = 0.0


df["case_age_days"] = pd.to_numeric(
    df.get("case_age_days", 0),
    errors="coerce"
).fillna(0)


df["hearing_gap_days"] = pd.to_numeric(
    df.get("hearing_gap_days", 0),
    errors="coerce"
).fillna(0)


# ============================================================
# ACTIVE CASES
# ============================================================

inactive_statuses = {
    "DISPOSED",
    "CLOSED",
    "DISMISSED"
}


if "case_status" in df.columns:

    active_df = df[
        ~df["case_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(inactive_statuses)
    ].copy()

else:

    active_df = df.copy()


if active_df.empty:

    st.warning(
        "No active cases available for court performance analysis."
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🔎 Filters"
)


districts = sorted(
    active_df["district_name"]
    .unique()
    .tolist()
)


selected_district = st.sidebar.selectbox(
    "District",
    ["All"] + districts
)


if selected_district != "All":

    active_df = active_df[
        active_df["district_name"]
        ==
        selected_district
    ].copy()


if active_df.empty:

    st.warning(
        "No active cases match the selected district."
    )

    st.stop()


# ============================================================
# COURT STATISTICS
# ============================================================

court_stats = (

    active_df

    .groupby("court_name")

    .agg(

        total_cases=(
            "case_id",
            "count"
        ),

        high_risk_cases=(
            "predicted_risk",
            lambda x:
            (x == "High").sum()
        ),

        average_high_risk_probability=(
            "high_risk_probability",
            "mean"
        ),

        average_predicted_duration=(
            "predicted_duration_years",
            "mean"
        ),

        average_case_age=(
            "case_age_days",
            "mean"
        ),

        average_hearing_gap=(
            "hearing_gap_days",
            "mean"
        )
    )

    .reset_index()
)


# ============================================================
# HIGH-RISK RATE
# ============================================================

court_stats["high_risk_rate"] = (

    court_stats["high_risk_cases"]

    /

    court_stats["total_cases"]

).fillna(0)


# ============================================================
# PERFORMANCE INDEX
# ============================================================
#
# This is a simple analytical index:
#
# - lower high-risk rate = better
# - lower predicted duration = better
#
# It is NOT a judicial quality score.
#

max_duration = (
    court_stats[
        "average_predicted_duration"
    ].max()
)


if max_duration > 0:

    duration_component = (

        1
        -
        (
            court_stats[
                "average_predicted_duration"
            ]
            /
            max_duration
        )

    )

else:

    duration_component = 1.0


risk_component = (
    1
    -
    court_stats[
        "high_risk_rate"
    ]
)


court_stats[
    "performance_index"
] = (

    (
        risk_component
        +
        duration_component
    )
    / 2

) * 100


# ============================================================
# DISPLAY VALUES
# ============================================================

court_stats[
    "average_high_risk_probability"
] = (

    court_stats[
        "average_high_risk_probability"
    ] * 100

)


court_stats[
    "high_risk_rate"
] = (

    court_stats[
        "high_risk_rate"
    ] * 100

)


court_stats[
    "average_predicted_duration"
] = (

    court_stats[
        "average_predicted_duration"
    ]

)


court_stats[
    "average_case_age"
] = (

    court_stats[
        "average_case_age"
    ]

)


court_stats[
    "average_hearing_gap"
] = (

    court_stats[
        "average_hearing_gap"
    ]

)


# ============================================================
# RANKINGS
# ============================================================

highest_risk = (

    court_stats
    .sort_values(
        "average_high_risk_probability",
        ascending=False
    )
)


longest_duration = (

    court_stats
    .sort_values(
        "average_predicted_duration",
        ascending=False
    )
)


best_performance = (

    court_stats
    .sort_values(
        "performance_index",
        ascending=False
    )
)


# ============================================================
# KPI
# ============================================================

st.subheader(
    "📊 Court Performance Overview"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Active Courts",
        len(court_stats)
    )


with c2:

    st.metric(
        "Active Cases",
        len(active_df)
    )


with c3:

    st.metric(
        "Avg Predicted Duration",
        f"{active_df['predicted_duration_years'].mean():.2f} yrs"
    )


with c4:

    st.metric(
        "Avg High-Risk Probability",
        f"{active_df['high_risk_probability'].mean() * 100:.1f}%"
    )


# ============================================================
# COURT PERFORMANCE COMPARISON
# ============================================================

st.subheader(
    "📈 Court Performance Comparison"
)

st.caption(
    "Higher performance index indicates a lower predicted "
    "delay-risk and duration profile."
)


performance_chart = (

    court_stats
    .sort_values(
        "performance_index",
        ascending=False
    )
    .head(15)
)


fig = px.bar(

    performance_chart,

    x="court_name",

    y="performance_index",

    text="performance_index",

    title="Court Performance Index"
)


fig.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)


fig.update_yaxes(
    title="Performance Index"
)


fig.update_xaxes(
    title="Court"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HIGH-RISK COURTS
# ============================================================

st.subheader(
    "🚨 Courts with Highest Predicted Delay Risk"
)


risk_chart = (

    highest_risk
    .head(10)
)


fig = px.bar(

    risk_chart,

    x="court_name",

    y="average_high_risk_probability",

    text="average_high_risk_probability",

    title="Average High-Risk Probability by Court"
)


fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig.update_yaxes(
    title="High-Risk Probability (%)"
)


fig.update_xaxes(
    title="Court"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PREDICTED DURATION
# ============================================================

st.subheader(
    "📅 Courts with Longest Predicted Case Duration"
)


duration_chart = (

    longest_duration
    .head(10)
)


fig = px.bar(

    duration_chart,

    x="court_name",

    y="average_predicted_duration",

    text="average_predicted_duration",

    title="Average Predicted Case Duration by Court"
)


fig.update_traces(
    texttemplate="%{text:.2f} yrs",
    textposition="outside"
)


fig.update_yaxes(
    title="Predicted Duration (years)"
)


fig.update_xaxes(
    title="Court"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# CASE STAGE ANALYSIS
# ============================================================

st.subheader(
    "⚠️ Predicted Delay Risk by Case Stage"
)


stage_stats = (

    active_df

    .groupby("stage_of_case")

    .agg(

        cases=(
            "case_id",
            "count"
        ),

        high_risk_probability=(
            "high_risk_probability",
            "mean"
        ),

        predicted_duration=(
            "predicted_duration_years",
            "mean"
        )
    )

    .reset_index()
)


stage_stats[
    "high_risk_probability"
] *= 100


fig = px.bar(

    stage_stats.sort_values(
        "high_risk_probability",
        ascending=False
    ),

    x="stage_of_case",

    y="high_risk_probability",

    text="high_risk_probability",

    title="Average High-Risk Probability by Case Stage"
)


fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig.update_yaxes(
    title="High-Risk Probability (%)"
)


fig.update_xaxes(
    title="Case Stage"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DISTRICT SUMMARY
# ============================================================

st.subheader(
    "🏛️ District-Level Court Performance"
)


district_stats = (

    active_df

    .groupby("district_name")

    .agg(

        cases=(
            "case_id",
            "count"
        ),

        high_risk_cases=(
            "predicted_risk",
            lambda x:
            (x == "High").sum()
        ),

        average_high_risk_probability=(
            "high_risk_probability",
            "mean"
        ),

        average_predicted_duration=(
            "predicted_duration_years",
            "mean"
        )

    )

    .reset_index()
)


district_stats[
    "average_high_risk_probability"
] = (

    district_stats[
        "average_high_risk_probability"
    ] * 100

).round(1)


district_stats[
    "average_predicted_duration"
] = (

    district_stats[
        "average_predicted_duration"
    ]

).round(2)


st.dataframe(

    district_stats,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# COURT PERFORMANCE TABLE
# ============================================================

st.subheader(
    "📋 Court Performance Details"
)


display_table = court_stats[
    [
        "court_name",
        "total_cases",
        "high_risk_cases",
        "average_high_risk_probability",
        "high_risk_rate",
        "average_predicted_duration",
        "average_case_age",
        "average_hearing_gap",
        "performance_index"
    ]
].copy()


display_table.columns = [

    "Court",

    "Active Cases",

    "High-Risk Cases",

    "Avg High-Risk Probability",

    "High-Risk Rate",

    "Avg Predicted Duration",

    "Avg Case Age",

    "Avg Hearing Gap",

    "Performance Index"
]


display_table[
    "Avg High-Risk Probability"
] = display_table[
    "Avg High-Risk Probability"
].round(1)


display_table[
    "High-Risk Rate"
] = display_table[
    "High-Risk Rate"
].round(1)


display_table[
    "Avg Predicted Duration"
] = display_table[
    "Avg Predicted Duration"
].round(2)


display_table[
    "Avg Case Age"
] = display_table[
    "Avg Case Age"
].round(0)


display_table[
    "Avg Hearing Gap"
] = display_table[
    "Avg Hearing Gap"
].round(0)


display_table[
    "Performance Index"
] = display_table[
    "Performance Index"
].round(1)


st.dataframe(

    display_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# KEY FINDING
# ============================================================

st.subheader(
    "📌 Key Performance Finding"
)


worst = highest_risk.iloc[0]

best = best_performance.iloc[0]


col1, col2 = st.columns(2)


with col1:

    st.warning(

        f"""
**Highest predicted delay-risk court**

**{worst['court_name']}**

Average high-risk probability:
**{worst['average_high_risk_probability']:.1f}%**

Average predicted duration:
**{worst['average_predicted_duration']:.2f} years**
"""
    )


with col2:

    st.success(

        f"""
**Strongest predicted performance profile**

**{best['court_name']}**

Performance index:
**{best['performance_index']:.1f}**

Average high-risk probability:
**{best['average_high_risk_probability']:.1f}%**

Average predicted duration:
**{best['average_predicted_duration']:.2f} years**
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "The Performance Index is an analytical comparison based "
    "on predicted delay risk and predicted case duration. "
    "It does not represent judicial quality, legal performance, "
    "or a judicial decision."
)