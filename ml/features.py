import pandas as pd


NUMERIC_FEATURES = [

    "case_age_days",

    "hearing_gap_days",

    "filing_year",

    "filing_month",

    "first_hearing_delay_days"
]


CATEGORICAL_FEATURES = [

    "district_name",

    "district_workload_level",

    "court_name",

    "case_type",

    "stage_of_case"
]


FEATURES = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)


def prepare_features(df):

    data = df.copy()


    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    for column in [

        "filing_date",

        "first_hearing_date",

        "decision_date"

    ]:

        if column in data.columns:

            data[column] = pd.to_datetime(
                data[column],
                errors="coerce"
            )


    # --------------------------------------------------------
    # Case age
    # --------------------------------------------------------

    if "case_age_days" not in data.columns:

        data["case_age_days"] = 0


    data["case_age_days"] = pd.to_numeric(

        data["case_age_days"],

        errors="coerce"

    ).fillna(0).clip(lower=0)


    # --------------------------------------------------------
    # Hearing gap
    # --------------------------------------------------------

    if "hearing_gap_days" not in data.columns:

        data["hearing_gap_days"] = 0


    data["hearing_gap_days"] = pd.to_numeric(

        data["hearing_gap_days"],

        errors="coerce"

    ).fillna(0).clip(lower=0)


    # --------------------------------------------------------
    # Filing year
    # --------------------------------------------------------

    if "filing_date" in data.columns:

        data["filing_year"] = (
            data["filing_date"]
            .dt.year
        )

        data["filing_month"] = (
            data["filing_date"]
            .dt.month
        )

    else:

        data["filing_year"] = 0
        data["filing_month"] = 0


    # --------------------------------------------------------
    # First hearing delay
    # --------------------------------------------------------

    if (
        "filing_date" in data.columns
        and
        "first_hearing_date" in data.columns
    ):

        data[
            "first_hearing_delay_days"
        ] = (

            data[
                "first_hearing_date"
            ]

            -

            data[
                "filing_date"
            ]

        ).dt.days

    else:

        data[
            "first_hearing_delay_days"
        ] = 0


    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        data[column] = pd.to_numeric(

            data[column],

            errors="coerce"

        ).fillna(0)


    # --------------------------------------------------------
    # Categorical cleanup
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if column not in data.columns:

            data[column] = "Unknown"


        data[column] = (

            data[column]

            .fillna("Unknown")

            .astype(str)

            .str.strip()
        )


        data.loc[
            data[column] == "",
            column
        ] = "Unknown"


    return data


def get_X(df):

    data = prepare_features(df)

    return data[
        FEATURES
    ].copy()


def get_duration_target(df):

    data = prepare_features(df)


    # Prefer stored duration if available

    if "case_duration_years" in data.columns:

        target = pd.to_numeric(

            data[
                "case_duration_years"
            ],

            errors="coerce"
        )

        return target


    # Otherwise calculate from dates

    if (
        "filing_date" in data.columns
        and
        "decision_date" in data.columns
    ):

        target = (

            data[
                "decision_date"
            ]

            -

            data[
                "filing_date"
            ]

        ).dt.days / 365.25


        return target


    raise ValueError(
        "Duration training requires "
        "case_duration_years or "
        "filing_date + decision_date."
    )


def get_risk_target(df):

    if "delay_risk_category" in df.columns:

        target = (

            df[
                "delay_risk_category"
            ]

            .astype(str)
            .str.strip()
            .str.title()
        )


        valid = [
            "Low",
            "Medium",
            "High"
        ]


        target = target.where(
            target.isin(valid)
        )


        return target


    raise ValueError(

        """
The PostgreSQL cases table does not contain
delay_risk_category.

Add historical Low/Medium/High labels to
the database before training the XGBoost
risk classifier.
"""
    )