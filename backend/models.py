from typing import Optional

from pydantic import BaseModel, Field


class CasePredictionRequest(BaseModel):
    """
    Input data required for Phase-2 ML prediction.
    """

    district_name: str
    district_workload_level: str
    court_name: str
    case_type: str
    stage_of_case: str
    act_name: str
    section: str
    police_station: str

    case_age_days: Optional[float] = Field(
        default=None,
        ge=0
    )

    hearing_gap_days: Optional[float] = Field(
        default=None,
        ge=0
    )

    filing_year: int
    filing_month_feature: int = Field(
        ge=1,
        le=12
    )


class DurationPredictionResponse(BaseModel):
    predicted_duration_years: float


class RiskPredictionResponse(BaseModel):
    predicted_risk: str
    risk_probabilities: dict[str, float]


class ExplanationResponse(BaseModel):
    predicted_risk: str
    top_features: list[dict]


class FullPredictionResponse(BaseModel):
    predicted_duration_years: float
    predicted_risk: str
    risk_probabilities: dict[str, float]
    explanation: list[dict]