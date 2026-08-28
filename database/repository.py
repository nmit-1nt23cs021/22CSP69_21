import uuid

import pandas as pd
from sqlalchemy import text

from database.connection import engine


# ============================================================
# GET ALL CASES
# ============================================================

def get_cases():

    query = text("""
        WITH hearing_ordered AS (

            SELECT
                h.case_id,
                h.hearing_date,

                LAG(h.hearing_date)
                OVER (
                    PARTITION BY h.case_id
                    ORDER BY h.hearing_date
                ) AS previous_hearing_date

            FROM hearings h
        ),

        hearing_stats AS (

            SELECT
                case_id,

                AVG(
                    hearing_date - previous_hearing_date
                ) AS hearing_gap_days

            FROM hearing_ordered

            WHERE previous_hearing_date IS NOT NULL

            GROUP BY case_id
        )

        SELECT

            c.case_id,
            c.cnr_number,
            c.registration_number,

            d.district_name,
            d.workload_level
                AS district_workload_level,

            co.court_name,

            c.case_type,
            c.act_name,
            c.section,
            c.police_station,

            c.filing_date,
            c.first_hearing_date,
            c.decision_date,

            c.case_status,

            s.canonical_stage
                AS stage_of_case,

            c.disposal_type,
            c.source_type,

            -- ML RISK TRAINING TARGET
            c.delay_risk_category,

            CASE

                WHEN c.filing_date IS NULL
                THEN 0

                ELSE
                    CURRENT_DATE - c.filing_date

            END AS case_age_days,

            COALESCE(
                ROUND(
                    hs.hearing_gap_days
                )::INTEGER,
                0
            ) AS hearing_gap_days

        FROM cases c

        LEFT JOIN districts d
            ON c.district_id = d.district_id

        LEFT JOIN courts co
            ON c.court_id = co.court_id

        LEFT JOIN stages s
            ON c.current_stage_id = s.stage_id

        LEFT JOIN hearing_stats hs
            ON c.case_id = hs.case_id

        ORDER BY
            c.filing_date DESC NULLS LAST
    """)


    df = pd.read_sql(
        query,
        engine
    )


    if df.empty:
        return df


    # --------------------------------------------------------
    # Numeric safety
    # --------------------------------------------------------

    for column in [
        "case_age_days",
        "hearing_gap_days"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # Text safety
    # --------------------------------------------------------

    text_columns = [

        "district_name",
        "district_workload_level",
        "court_name",
        "case_type",
        "act_name",
        "section",
        "police_station",
        "case_status",
        "stage_of_case",
        "disposal_type",
        "source_type",
        "delay_risk_category"
    ]


    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )


    return df


# ============================================================
# GET ONE CASE
# ============================================================

def get_case_by_cnr(cnr_number):

    query = text("""
        SELECT

            c.case_id,
            c.cnr_number,
            c.registration_number,

            d.district_name,
            d.workload_level
                AS district_workload_level,

            co.court_name,

            c.case_type,
            c.act_name,
            c.section,
            c.police_station,

            c.filing_date,
            c.first_hearing_date,
            c.decision_date,

            c.case_status,

            s.canonical_stage
                AS stage_of_case,

            c.disposal_type,
            c.source_type,

            c.delay_risk_category

        FROM cases c

        LEFT JOIN districts d
            ON c.district_id = d.district_id

        LEFT JOIN courts co
            ON c.court_id = co.court_id

        LEFT JOIN stages s
            ON c.current_stage_id = s.stage_id

        WHERE c.cnr_number = :cnr

        LIMIT 1
    """)


    return pd.read_sql(
        query,
        engine,
        params={
            "cnr": cnr_number
        }
    )


# ============================================================
# ADD CASE
# ============================================================

def add_case(case_data):

    case_id = case_data.get(
        "case_id",
        "CASE-" + uuid.uuid4().hex[:12].upper()
    )


    district_name = case_data["district_name"]

    court_name = case_data["court_name"]

    stage_name = case_data["stage_of_case"]


    with engine.begin() as conn:

        # ----------------------------------------------------
        # DISTRICT
        # ----------------------------------------------------

        district_id = conn.execute(

            text("""
                SELECT district_id
                FROM districts

                WHERE LOWER(district_name)
                    = LOWER(:name)

                LIMIT 1
            """),

            {
                "name": district_name
            }

        ).scalar()


        if district_id is None:

            district_id = (
                "D-"
                +
                uuid.uuid4()
                .hex[:10]
                .upper()
            )


            conn.execute(

                text("""
                    INSERT INTO districts (
                        district_id,
                        district_name,
                        workload_level,
                        source_type
                    )

                    VALUES (
                        :id,
                        :name,
                        'Medium',
                        'MANUAL'
                    )
                """),

                {
                    "id": district_id,
                    "name": district_name
                }
            )


        # ----------------------------------------------------
        # COURT
        # ----------------------------------------------------

        court_id = conn.execute(

            text("""
                SELECT court_id
                FROM courts

                WHERE LOWER(court_name)
                    = LOWER(:court_name)

                AND district_id = :district_id

                LIMIT 1
            """),

            {
                "court_name": court_name,
                "district_id": district_id
            }

        ).scalar()


        if court_id is None:

            court_id = (
                "C-"
                +
                uuid.uuid4()
                .hex[:10]
                .upper()
            )


            conn.execute(

                text("""
                    INSERT INTO courts (
                        court_id,
                        district_id,
                        court_name,
                        source_type
                    )

                    VALUES (
                        :id,
                        :district_id,
                        :court_name,
                        'MANUAL'
                    )
                """),

                {
                    "id": court_id,
                    "district_id": district_id,
                    "court_name": court_name
                }
            )


        # ----------------------------------------------------
        # STAGE
        # ----------------------------------------------------

        stage_id = conn.execute(

            text("""
                SELECT stage_id
                FROM stages

                WHERE LOWER(canonical_stage)
                    = LOWER(:stage)

                LIMIT 1
            """),

            {
                "stage": stage_name
            }

        ).scalar()


        if stage_id is None:

            stage_id = conn.execute(

                text("""
                    INSERT INTO stages (
                        raw_stage,
                        canonical_stage,
                        stage_group,
                        stage_order
                    )

                    VALUES (
                        :stage,
                        :stage,
                        'OTHER',
                        999
                    )

                    RETURNING stage_id
                """),

                {
                    "stage": stage_name
                }

            ).scalar()


        # ----------------------------------------------------
        # CASE
        # ----------------------------------------------------

        conn.execute(

            text("""
                INSERT INTO cases (

                    case_id,
                    cnr_number,
                    registration_number,

                    district_id,
                    court_id,

                    case_type,
                    act_name,
                    section,
                    police_station,

                    filing_date,
                    first_hearing_date,
                    decision_date,

                    case_status,
                    current_stage_id,

                    disposal_type,
                    source_type,

                    delay_risk_category
                )

                VALUES (

                    :case_id,
                    :cnr_number,
                    :registration_number,

                    :district_id,
                    :court_id,

                    :case_type,
                    :act_name,
                    :section,
                    :police_station,

                    :filing_date,
                    :first_hearing_date,
                    :decision_date,

                    :case_status,
                    :stage_id,

                    :disposal_type,
                    :source_type,

                    :delay_risk_category
                )
            """),

            {

                "case_id": case_id,

                "cnr_number":
                    case_data.get(
                        "cnr_number"
                    ),

                "registration_number":
                    case_data.get(
                        "registration_number"
                    ),

                "district_id":
                    district_id,

                "court_id":
                    court_id,

                "case_type":
                    case_data.get(
                        "case_type"
                    ),

                "act_name":
                    case_data.get(
                        "act_name"
                    ),

                "section":
                    case_data.get(
                        "section"
                    ),

                "police_station":
                    case_data.get(
                        "police_station"
                    ),

                "filing_date":
                    case_data.get(
                        "filing_date"
                    ),

                "first_hearing_date":
                    case_data.get(
                        "first_hearing_date"
                    ),

                "decision_date":
                    case_data.get(
                        "decision_date"
                    ),

                "case_status":
                    case_data.get(
                        "case_status",
                        "Pending"
                    ),

                "stage_id":
                    stage_id,

                "disposal_type":
                    case_data.get(
                        "disposal_type"
                    ),

                "source_type":
                    case_data.get(
                        "source_type",
                        "MANUAL"
                    ),

                # Usually NULL for newly entered cases.
                # ML prediction will calculate the risk.
                "delay_risk_category":
                    case_data.get(
                        "delay_risk_category"
                    )
            }
        )


    return case_id