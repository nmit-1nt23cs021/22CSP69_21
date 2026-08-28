from pathlib import Path

import joblib
import pandas as pd

from ml.features import prepare_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
)

ARTIFACT_DIR = (
    BASE_DIR / "ml_artifacts"
)

DURATION_MODEL_PATH = (
    ARTIFACT_DIR
    / "duration_model.joblib"
)

RISK_MODEL_PATH = (
    ARTIFACT_DIR
    / "risk_model.joblib"
)


# ============================================================
# MODEL CACHE
# ============================================================

_duration_bundle = None
_risk_bundle = None


# ============================================================
# LOAD DURATION MODEL
# ============================================================

def load_duration_bundle():

    global _duration_bundle

    if _duration_bundle is not None:

        return _duration_bundle

    if not DURATION_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Duration model has not been trained.\n\n"
            "Run:\n"
            "python -m ml.train"
        )

    _duration_bundle = joblib.load(
        DURATION_MODEL_PATH
    )

    return _duration_bundle


# ============================================================
# LOAD RISK MODEL
# ============================================================

def load_risk_bundle():

    global _risk_bundle

    if _risk_bundle is not None:

        return _risk_bundle

    if not RISK_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Risk model has not been trained.\n\n"
            "Run:\n"
            "python -m ml.train"
        )

    _risk_bundle = joblib.load(
        RISK_MODEL_PATH
    )

    return _risk_bundle


# ============================================================
# PREPARE DATA
# ============================================================

def _prepare_dataframe(df):

    if df is None:

        return pd.DataFrame()

    if df.empty:

        return df.copy()

    data = df.copy()

    # IMPORTANT:
    #
    # This creates the same derived features
    # used during model training:
    #
    # case_age_days
    # hearing_gap_days
    # filing_year
    # filing_month
    # first_hearing_delay_days
    #
    data = prepare_features(
        data
    )

    return data


# ============================================================
# PREDICT DURATION FOR ONE CASE
# ============================================================

def predict_duration(case_data):

    bundle = (
        load_duration_bundle()
    )

    input_df = pd.DataFrame(
        [case_data]
    )

    # Create all ML features
    input_df = prepare_features(
        input_df
    )

    features = bundle[
        "features"
    ]

    missing = [
        column
        for column in features
        if column not in input_df.columns
    ]

    if missing:

        raise KeyError(
            "Missing ML features after "
            "prepare_features(): "
            + str(missing)
        )

    X = input_df[
        features
    ]

    X_encoded = (
        bundle[
            "preprocessor"
        ]
        .transform(
            X
        )
    )

    prediction = (
        bundle[
            "model"
        ]
        .predict(
            X_encoded
        )
    )

    predicted_years = max(
        0.0,
        float(
            prediction[0]
        )
    )

    predicted_days = (
        predicted_years
        * 365.25
    )

    return {

        "predicted_duration_years":
            predicted_years,

        "predicted_duration_days":
            predicted_days
    }


# ============================================================
# PREDICT RISK FOR ONE CASE
# ============================================================

def predict_risk(case_data):

    bundle = (
        load_risk_bundle()
    )

    input_df = pd.DataFrame(
        [case_data]
    )

    # Create all ML features
    input_df = prepare_features(
        input_df
    )

    features = bundle[
        "features"
    ]

    missing = [
        column
        for column in features
        if column not in input_df.columns
    ]

    if missing:

        raise KeyError(
            "Missing ML features after "
            "prepare_features(): "
            + str(missing)
        )

    X = input_df[
        features
    ]

    X_encoded = (
        bundle[
            "preprocessor"
        ]
        .transform(
            X
        )
    )

    model = bundle[
        "model"
    ]

    encoded_prediction = (
        model.predict(
            X_encoded
        )
    )

    probabilities = (
        model.predict_proba(
            X_encoded
        )
    )

    label_encoder = (
        bundle[
            "label_encoder"
        ]
    )

    predicted_label = (
        label_encoder
        .inverse_transform(
            encoded_prediction
        )
    )

    probability_map = {}

    for i, label in enumerate(
        label_encoder.classes_
    ):

        probability_map[
            str(label)
        ] = float(
            probabilities[
                0,
                i
            ]
        )

    confidence = float(
        probabilities.max(
            axis=1
        )[0]
    )

    # --------------------------------------------------------
    # High-risk probability
    # --------------------------------------------------------

    high_risk_probability = 0.0

    for i, label in enumerate(
        label_encoder.classes_
    ):

        if (
            str(label)
            .strip()
            .lower()
            == "high"
        ):

            high_risk_probability = float(
                probabilities[
                    0,
                    i
                ]
            )

            break

    return {

        "predicted_risk":
            str(
                predicted_label[0]
            ),

        "risk_confidence":
            confidence,

        "risk_probabilities":
            probability_map,

        "high_risk_probability":
            high_risk_probability
    }


# ============================================================
# PREDICT ENTIRE DATAFRAME
# ============================================================

def predict_dataframe(df):

    if df is None:

        return pd.DataFrame()

    if df.empty:

        return df.copy()

    # --------------------------------------------------------
    # Prepare ALL rows at once
    # --------------------------------------------------------

    data = _prepare_dataframe(
        df
    )

    # ========================================================
    # DURATION MODEL
    # ========================================================

    duration_bundle = (
        load_duration_bundle()
    )

    duration_features = (
        duration_bundle[
            "features"
        ]
    )

    missing_duration = [
        column
        for column in duration_features
        if column not in data.columns
    ]

    if missing_duration:

        raise KeyError(
            "Missing duration features after "
            "prepare_features(): "
            + str(missing_duration)
        )

    X_duration = data[
        duration_features
    ]

    X_duration_encoded = (
        duration_bundle[
            "preprocessor"
        ]
        .transform(
            X_duration
        )
    )

    duration_predictions = (
        duration_bundle[
            "model"
        ]
        .predict(
            X_duration_encoded
        )
    )

    # Prevent negative duration
    duration_predictions = (
        pd.Series(
            duration_predictions
        )
        .clip(
            lower=0
        )
        .to_numpy()
    )

    duration_days = (
        duration_predictions
        * 365.25
    )


    # ========================================================
    # RISK MODEL
    # ========================================================

    risk_bundle = (
        load_risk_bundle()
    )

    risk_features = (
        risk_bundle[
            "features"
        ]
    )

    missing_risk = [
        column
        for column in risk_features
        if column not in data.columns
    ]

    if missing_risk:

        raise KeyError(
            "Missing risk features after "
            "prepare_features(): "
            + str(missing_risk)
        )

    X_risk = data[
        risk_features
    ]

    X_risk_encoded = (
        risk_bundle[
            "preprocessor"
        ]
        .transform(
            X_risk
        )
    )

    risk_model = (
        risk_bundle[
            "model"
        ]
    )

    encoded_predictions = (
        risk_model
        .predict(
            X_risk_encoded
        )
    )

    probabilities = (
        risk_model
        .predict_proba(
            X_risk_encoded
        )
    )

    label_encoder = (
        risk_bundle[
            "label_encoder"
        ]
    )

    risk_labels = (
        label_encoder
        .inverse_transform(
            encoded_predictions
        )
    )

    risk_confidence = (
        probabilities
        .max(
            axis=1
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = df.copy()

    result[
        "predicted_duration_years"
    ] = duration_predictions

    result[
        "predicted_duration_days"
    ] = duration_days

    result[
        "predicted_risk"
    ] = risk_labels

    result[
        "risk_confidence"
    ] = risk_confidence


    # ========================================================
    # RISK PROBABILITIES
    # ========================================================

    for i, label in enumerate(
        label_encoder.classes_
    ):

        column_name = (
            "risk_probability_"
            +
            str(label)
            .strip()
            .lower()
        )

        result[
            column_name
        ] = probabilities[
            :,
            i
        ]


    # ========================================================
    # HIGH-RISK PROBABILITY
    # ========================================================
    #
    # The Streamlit pages use:
    #
    #     high_risk_probability
    #
    # So create it here centrally.
    #
    # This is the XGBoost probability of the
    # High-risk class.
    # ========================================================

    high_risk_index = None

    for i, label in enumerate(
        label_encoder.classes_
    ):

        if (
            str(label)
            .strip()
            .lower()
            == "high"
        ):

            high_risk_index = i

            break


    if high_risk_index is not None:

        result[
            "high_risk_probability"
        ] = probabilities[
            :,
            high_risk_index
        ]

    else:

        result[
            "high_risk_probability"
        ] = 0.0


    return result


# ============================================================
# SINGLE CASE - COMPLETE PREDICTION
# ============================================================

def predict_case(case_data):

    duration = (
        predict_duration(
            case_data
        )
    )

    risk = (
        predict_risk(
            case_data
        )
    )

    result = {}

    result.update(
        duration
    )

    result.update(
        risk
    )

    return result