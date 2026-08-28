import streamlit as st
import pandas as pd

from database.repository import get_cases
from ml.predict import predict_dataframe
from ml.explain import explain_case


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Case Recommendations & Risk Insights",
    page_icon="📌",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📌 Case Recommendations & Risk Insights"
)

st.markdown(
    "### ML-Based Judicial Decision Support"
)

st.caption(
    "Uses PostgreSQL case data, XGBoost risk and duration "
    "predictions, and SHAP explanations to identify "
    "priority cases and support administrative review."
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

    df = predict_dataframe(
        df
    )

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
    "district_name",
    "court_name",
    "stage_of_case"
]:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )


# ============================================================
# PREDICTED DURATION
# ============================================================

df[
    "predicted_duration_years"
] = pd.to_numeric(

    df[
        "predicted_duration_years"
    ],

    errors="coerce"

).fillna(0)


# ============================================================
# HIGH RISK PROBABILITY
# ============================================================

if "risk_probability_high" in df.columns:

    df[
        "high_risk_probability"
    ] = pd.to_numeric(

        df[
            "risk_probability_high"
        ],

        errors="coerce"

    ).fillna(0)

else:

    df[
        "high_risk_probability"
    ] = 0.0


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🔎 Filters"
)


districts = sorted(
    df[
        "district_name"
    ].unique()
)


selected_district = st.sidebar.selectbox(

    "District",

    ["All"] + districts
)


risk_levels = [
    "All",
    "High",
    "Medium",
    "Low"
]


selected_risk = st.sidebar.selectbox(

    "ML Risk",

    risk_levels
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


if selected_district != "All":

    filtered = filtered[
        filtered[
            "district_name"
        ]
        ==
        selected_district
    ]


if selected_risk != "All":

    filtered = filtered[
        filtered[
            "predicted_risk"
        ]
        ==
        selected_risk
    ]


if filtered.empty:

    st.warning(
        "No cases match the selected filters."
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    "📊 Risk & Duration Overview"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Cases",
        len(filtered)
    )


with c2:

    st.metric(

        "High-Risk Cases",

        int(

            (
                filtered[
                    "predicted_risk"
                ]
                ==
                "High"
            ).sum()

        )
    )


with c3:

    st.metric(

        "Avg Predicted Duration",

        f"{filtered['predicted_duration_years'].mean():.2f} yrs"
    )


with c4:

    st.metric(

        "Avg High-Risk Probability",

        f"{filtered['high_risk_probability'].mean() * 100:.1f}%"
    )


# ============================================================
# ML-RANKED CASES
# ============================================================

st.subheader(
    "🚨 Priority Cases"
)

st.caption(
    "Cases are ranked using high-risk probability first, "
    "followed by predicted duration."
)


ranked_cases = (

    filtered

    .sort_values(

        [
            "high_risk_probability",
            "predicted_duration_years"
        ],

        ascending=False
    )

    .reset_index(
        drop=True
    )
)


display_columns = [

    "cnr_number",

    "district_name",

    "court_name",

    "stage_of_case",

    "predicted_risk",

    "high_risk_probability",

    "predicted_duration_years"
]


available_columns = [

    column

    for column in display_columns

    if column in ranked_cases.columns
]


ranking_table = ranked_cases[
    available_columns
].copy()


if "high_risk_probability" in ranking_table.columns:

    ranking_table[
        "high_risk_probability"
    ] = (

        ranking_table[
            "high_risk_probability"
        ]

        * 100

    ).round(2)


if "predicted_duration_years" in ranking_table.columns:

    ranking_table[
        "predicted_duration_years"
    ] = (

        ranking_table[
            "predicted_duration_years"
        ]

    ).round(2)


st.dataframe(

    ranking_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# CASE SELECTION
# ============================================================

st.subheader(
    "🔎 Case-Specific Recommendation"
)


case_labels = []


for index, row in ranked_cases.iterrows():

    cnr = row.get(
        "cnr_number",
        "No CNR"
    )

    district = row.get(
        "district_name",
        "Unknown"
    )

    risk_value = row.get(
        "predicted_risk",
        "Unknown"
    )

    case_labels.append(

        f"{cnr} | {district} | {risk_value}"

    )


selected_label = st.selectbox(

    "Select a case",

    case_labels
)


selected_index = (

    case_labels.index(
        selected_label
    )
)


selected_case = (

    ranked_cases

    .iloc[
        selected_index
    ]
)


# ============================================================
# SELECTED CASE OVERVIEW
# ============================================================

st.subheader(
    "📋 Selected Case"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(

        "ML Risk",

        selected_case[
            "predicted_risk"
        ]
    )


with c2:

    st.metric(

        "High-Risk Probability",

        f"{selected_case['high_risk_probability'] * 100:.1f}%"
    )


with c3:

    st.metric(

        "Predicted Duration",

        f"{selected_case['predicted_duration_years']:.2f} years"
    )


with c4:

    st.metric(

        "District",

        selected_case[
            "district_name"
        ]
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader(
    "🔍 Why Did the Model Predict This?"
)

st.caption(
    "SHAP identifies the features that contributed most "
    "to the model's risk and duration predictions."
)


try:

    explanation = explain_case(
        selected_case.to_dict()
    )


    risk_factors = pd.DataFrame(

        explanation.get(
            "risk_factors",
            []
        )
    )


    duration_factors = pd.DataFrame(

        explanation.get(
            "duration_factors",
            []
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### ⚠️ Delay-Risk Factors"
        )


        if not risk_factors.empty:

            st.dataframe(

                risk_factors,

                use_container_width=True,

                hide_index=True
            )

        else:

            st.info(
                "No SHAP risk factors available."
            )


    with col2:

        st.markdown(
            "### 📅 Duration Factors"
        )


        if not duration_factors.empty:

            st.dataframe(

                duration_factors,

                use_container_width=True,

                hide_index=True
            )

        else:

            st.info(
                "No SHAP duration factors available."
            )


except Exception as exc:

    st.warning(
        f"SHAP explanation unavailable: {exc}"
    )


# ============================================================
# ML RECOMMENDATION
# ============================================================

st.subheader(
    "🎯 Recommended Administrative Action"
)


predicted_risk = str(

    selected_case[
        "predicted_risk"
    ]
)


high_probability = float(

    selected_case[
        "high_risk_probability"
    ]
)


predicted_duration = float(

    selected_case[
        "predicted_duration_years"
    ]
)


# ============================================================
# HIGH RISK
# ============================================================

if predicted_risk == "High":

    st.error(

        f"""
### 🔴 High Priority

The XGBoost model predicts **High Delay Risk**.

**High-risk probability:**  
{high_probability * 100:.1f}%

**Predicted duration:**  
{predicted_duration:.2f} years

**Recommended action:**  
Prioritize this case for administrative review and
examine the SHAP factors contributing to the prediction.
"""
    )


# ============================================================
# MEDIUM RISK
# ============================================================

elif predicted_risk == "Medium":

    st.warning(

        f"""
### 🟠 Moderate Priority

The XGBoost model predicts **Medium Delay Risk**.

**High-risk probability:**  
{high_probability * 100:.1f}%

**Predicted duration:**  
{predicted_duration:.2f} years

**Recommended action:**  
Keep the case under closer monitoring and review
the factors contributing to the prediction.
"""
    )


# ============================================================
# LOW RISK
# ============================================================

else:

    st.success(

        f"""
### 🟢 Normal Priority

The XGBoost model predicts **Low Delay Risk**.

**High-risk probability:**  
{high_probability * 100:.1f}%

**Predicted duration:**  
{predicted_duration:.2f} years

**Recommended action:**  
Continue normal monitoring and reassess the prediction
as new case information becomes available.
"""
    )


# ============================================================
# DISTRICT-LEVEL ANALYSIS
# ============================================================

st.subheader(
    "🏛️ District-Level Risk & Duration"
)


district_summary = (

    filtered

    .groupby(
        "district_name"
    )

    .agg(

        cases=(
            "case_id",
            "count"
        ),

        high_risk_cases=(
            "predicted_risk",

            lambda x:
            (
                x == "High"
            ).sum()
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


district_summary[
    "average_high_risk_probability"
] = (

    district_summary[
        "average_high_risk_probability"
    ]

    * 100

).round(1)


district_summary[
    "average_predicted_duration"
] = (

    district_summary[
        "average_predicted_duration"
    ]

).round(2)


st.dataframe(

    district_summary,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Recommendations are decision-support outputs derived "
    "from XGBoost predictions and SHAP explanations. "
    "They are not legal advice or judicial decisions."
)