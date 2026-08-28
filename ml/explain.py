import joblib
import pandas as pd
import shap

from pathlib import Path

from ml.features import get_X


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

DURATION_MODEL_PATH = (
    ARTIFACT_DIR / "duration_model.joblib"
)

RISK_MODEL_PATH = (
    ARTIFACT_DIR / "risk_model.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

_duration_bundle = None
_risk_bundle = None


def load_duration_bundle():

    global _duration_bundle

    if _duration_bundle is None:

        if not DURATION_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Duration model not found.\n"
                "Run: python -m ml.train"
            )

        _duration_bundle = joblib.load(
            DURATION_MODEL_PATH
        )

    return _duration_bundle


def load_risk_bundle():

    global _risk_bundle

    if _risk_bundle is None:

        if not RISK_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Risk model not found.\n"
                "Run: python -m ml.train"
            )

        _risk_bundle = joblib.load(
            RISK_MODEL_PATH
        )

    return _risk_bundle


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(case_data, bundle):

    # Convert dictionary into one-row DataFrame
    raw_df = pd.DataFrame(
        [case_data]
    )

    # IMPORTANT:
    # This creates:
    #
    # filing_year
    # filing_month
    # first_hearing_delay_days
    # case_age_days
    # hearing_gap_days
    #
    # and cleans categorical values.
    X = get_X(
        raw_df
    )

    # Use exactly the features stored
    # with the trained model.
    features = bundle.get(
        "features"
    )

    if features is not None:

        for feature in features:

            if feature not in X.columns:

                # Safety fallback
                X[feature] = 0

        X = X[
            features
        ]

    return X


# ============================================================
# GET SHAP VALUES
# ============================================================

def get_shap_values(
    model,
    transformed_data
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        transformed_data
    )

    return explainer, shap_values


# ============================================================
# FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    preprocessor
):

    try:

        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        return []


# ============================================================
# RISK EXPLANATION
# ============================================================

def explain_risk(case_data):

    bundle = load_risk_bundle()

    model = bundle[
        "model"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]

    label_encoder = bundle[
        "label_encoder"
    ]

    # --------------------------------------------------------
    # Prepare raw case correctly
    # --------------------------------------------------------

    X = prepare_model_input(
        case_data,
        bundle
    )

    # --------------------------------------------------------
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    X_encoded = preprocessor.transform(
        X
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_encoded
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    if not feature_names:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                X_encoded.shape[1]
            )
        ]

    # --------------------------------------------------------
    # Handle multiclass SHAP output
    # --------------------------------------------------------

    predicted_encoded = int(
        model.predict(
            X_encoded
        )[0]
    )

    predicted_class = label_encoder.inverse_transform(
        [predicted_encoded]
    )[0]

    if isinstance(
        shap_values,
        list
    ):

        # XGBoost multiclass:
        # one SHAP array per class

        class_values = shap_values[
            predicted_encoded
        ][0]

    else:

        shap_array = shap_values

        if len(shap_array.shape) == 3:

            class_values = shap_array[
                0,
                :,
                predicted_encoded
            ]

        else:

            class_values = shap_array[
                0
            ]

    # --------------------------------------------------------
    # Build explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": class_values
        }
    )

    explanation[
        "absolute_impact"
    ] = explanation[
        "shap_value"
    ].abs()

    explanation = (
        explanation
        .sort_values(
            "absolute_impact",
            ascending=False
        )
        .head(10)
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Convert technical feature names into readable names
    # --------------------------------------------------------

    explanation[
        "direction"
    ] = explanation[
        "shap_value"
    ].apply(

        lambda value:
        "Increases predicted risk"
        if value > 0
        else "Decreases predicted risk"
    )

    return {

        "predicted_risk":
            predicted_class,

        "factors":
            explanation.to_dict(
                orient="records"
            )
    }


# ============================================================
# DURATION EXPLANATION
# ============================================================

def explain_duration(case_data):

    bundle = load_duration_bundle()

    model = bundle[
        "model"
    ]

    preprocessor = bundle[
        "preprocessor"
    ]

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = prepare_model_input(
        case_data,
        bundle
    )

    # --------------------------------------------------------
    # Same preprocessing as training
    # --------------------------------------------------------

    X_encoded = preprocessor.transform(
        X
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_encoded
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    if not feature_names:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                X_encoded.shape[1]
            )
        ]

    # --------------------------------------------------------
    # SHAP values
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[
            0
        ]

    else:

        values = shap_values[
            0
        ]

    # --------------------------------------------------------
    # Explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values
        }
    )

    explanation[
        "absolute_impact"
    ] = explanation[
        "shap_value"
    ].abs()

    explanation = (
        explanation
        .sort_values(
            "absolute_impact",
            ascending=False
        )
        .head(10)
        .reset_index(
            drop=True
        )
    )

    explanation[
        "direction"
    ] = explanation[
        "shap_value"
    ].apply(

        lambda value:
        "Increases predicted duration"
        if value > 0
        else "Decreases predicted duration"
    )

    return {

        "factors":
            explanation.to_dict(
                orient="records"
            )
    }


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def explain_case(case_data):

    risk_result = explain_risk(
        case_data
    )

    duration_result = explain_duration(
        case_data
    )

    return {

        "predicted_risk":
            risk_result[
                "predicted_risk"
            ],

        "risk_factors":
            risk_result[
                "factors"
            ],

        "duration_factors":
            duration_result[
                "factors"
            ]
    }