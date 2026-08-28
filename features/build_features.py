import pandas as pd


def build_case_features(df):

    df = df.copy()

    # ------------------------------------------------
    # Temporal features
    # ------------------------------------------------

    if "filing_date" in df.columns:

        snapshot_date = pd.Timestamp.today().normalize()

        df["case_age_days"] = (
            snapshot_date -
            pd.to_datetime(
                df["filing_date"],
                errors="coerce"
            )
        ).dt.days

        df["filing_year"] = (
            pd.to_datetime(
                df["filing_date"],
                errors="coerce"
            ).dt.year
        )

        df["filing_month"] = (
            pd.to_datetime(
                df["filing_date"],
                errors="coerce"
            ).dt.month
        )

    # ------------------------------------------------
    # Duration target
    # ------------------------------------------------

    if {
        "filing_date",
        "decision_date"
    }.issubset(df.columns):

        df["actual_duration_days"] = (
            pd.to_datetime(df["decision_date"])
            -
            pd.to_datetime(df["filing_date"])
        ).dt.days

    return df