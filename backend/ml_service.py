from ml.predict import (
    predict_case
)

from ml.explain import (
    explain_case
)


def get_complete_prediction(
    case_data: dict
):

    prediction = predict_case(
        case_data
    )


    explanation = explain_case(
        case_data
    )


    return {

        **prediction,

        "risk_factors":
            explanation[
                "risk_factors"
            ],

        "duration_factors":
            explanation[
                "duration_factors"
            ]
    }