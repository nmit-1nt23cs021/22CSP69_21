from fastapi import (
    FastAPI,
    HTTPException
)

from backend.models import (
    CasePredictionRequest
)

from backend.ml_service import (
    get_complete_prediction
)


app = FastAPI(

    title=
        "Judicial Analytics ML API",

    description=
        "XGBoost prediction and SHAP explainability",

    version="2.0.0"
)


@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "service":
            "Judicial Analytics ML API"
    }


@app.post(
    "/predict"
)
def predict(
    request:
        CasePredictionRequest
):

    try:

        return get_complete_prediction(

            request.model_dump()
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=str(exc)
        )