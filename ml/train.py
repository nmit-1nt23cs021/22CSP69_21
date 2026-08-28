import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from database.repository import get_cases

from ml.features import (
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    get_X,
    get_duration_target,
    get_risk_target
)


BASE_DIR = Path(
    __file__
).resolve().parent.parent


ARTIFACT_DIR = (
    BASE_DIR / "ml_artifacts"
)


ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    return ColumnTransformer(

        transformers=[

            (
                "numeric",

                "passthrough",

                NUMERIC_FEATURES
            ),

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                CATEGORICAL_FEATURES
            )
        ]
    )


# ============================================================
# DURATION MODEL
# ============================================================

def train_duration(df):

    target = get_duration_target(
        df
    )


    mask = (
        target.notna()
        &
        (target > 0)
    )


    X = get_X(
        df.loc[mask]
    )


    y = target.loc[mask]


    if len(y) < 20:

        raise ValueError(

            f"Only {len(y)} valid duration "
            "records found. Need at least 20."
        )


    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42
        )
    )


    preprocessor = (
        create_preprocessor()
    )


    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )


    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )


    model = XGBRegressor(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.85,

        colsample_bytree=0.85,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=2
    )


    model.fit(
        X_train_encoded,
        y_train
    )


    predictions = model.predict(
        X_test_encoded
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = mean_squared_error(

        y_test,

        predictions
    ) ** 0.5


    r2 = r2_score(
        y_test,
        predictions
    )


    joblib.dump(

        {
            "model": model,

            "preprocessor":
                preprocessor,

            "features":
                FEATURES
        },

        ARTIFACT_DIR
        /
        "duration_model.joblib"
    )


    metadata = {

        "model":
            "XGBRegressor",

        "training_rows":
            len(y),

        "mae":
            float(mae),

        "rmse":
            float(rmse),

        "r2":
            float(r2)
    }


    with open(

        ARTIFACT_DIR
        /
        "duration_metadata.json",

        "w"

    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


    print(
        "\nDURATION MODEL"
    )

    print(
        f"Rows: {len(y)}"
    )

    print(
        f"MAE: {mae:.3f}"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"R2: {r2:.3f}"
    )


# ============================================================
# RISK MODEL
# ============================================================

def train_risk(df):

    target = get_risk_target(
        df
    )


    mask = target.notna()


    X = get_X(
        df.loc[mask]
    )


    y = target.loc[mask]


    if len(y) < 30:

        raise ValueError(

            f"Only {len(y)} risk records found. "
            "Need at least 30."
        )


    label_encoder = (
        LabelEncoder()
    )


    y_encoded = (
        label_encoder
        .fit_transform(y)
    )


    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y_encoded,

            test_size=0.20,

            random_state=42,

            stratify=y_encoded
        )
    )


    preprocessor = (
        create_preprocessor()
    )


    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )


    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )


    number_of_classes = (
        len(
            label_encoder.classes_
        )
    )


    model = XGBClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.85,

        colsample_bytree=0.85,

        objective="multi:softprob",

        num_class=number_of_classes,

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=2
    )


    model.fit(

        X_train_encoded,

        y_train
    )


    predictions = (
        model.predict(
            X_test_encoded
        )
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    weighted_f1 = f1_score(

        y_test,

        predictions,

        average="weighted"
    )


    joblib.dump(

        {
            "model": model,

            "preprocessor":
                preprocessor,

            "label_encoder":
                label_encoder,

            "features":
                FEATURES
        },

        ARTIFACT_DIR
        /
        "risk_model.joblib"
    )


    metadata = {

        "model":
            "XGBClassifier",

        "training_rows":
            len(y),

        "accuracy":
            float(accuracy),

        "weighted_f1":
            float(weighted_f1),

        "classes":
            label_encoder
            .classes_
            .tolist()
    }


    with open(

        ARTIFACT_DIR
        /
        "risk_metadata.json",

        "w"

    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


    print(
        "\nRISK MODEL"
    )

    print(
        f"Rows: {len(y)}"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.3f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Loading training data from PostgreSQL..."
    )


    df = get_cases()


    if df.empty:

        raise ValueError(
            "PostgreSQL returned no cases."
        )


    print(
        f"Loaded {len(df)} cases."
    )


    train_duration(
        df
    )


    train_risk(
        df
    )


    print(
        "\n================================"
    )

    print(
        "XGBOOST TRAINING COMPLETE"
    )

    print(
        "================================"
    )