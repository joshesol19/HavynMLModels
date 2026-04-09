"""
pipeline_run.py — Havyn ML Pipeline
Runs every morning via GitHub Actions cron job.

What this file does:
  1. Connects to Azure PostgreSQL
  2. Loads all required tables
  3. Retrains health, education, and emotional progress models in memory
  4. Runs .predict_proba() on every active resident
  5. Upserts results into ResidentPredictions table
  6. ASP.NET backend reads that table — no ML code in the backend

Usage:
  python pipeline_run.py

Environment variables required (set as GitHub Actions secrets):
  PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT
"""

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ── ml_pipeline.py lives in the same directory as this file ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_pipeline import MLPipeline

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATABASE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_db_url() -> str:
    # Load .env if running locally; GitHub Actions injects secrets as env vars
    for candidate in [Path(__file__).parent / ".env", Path(".env")]:
        if candidate.is_file():
            load_dotenv(candidate, override=True)
            break

    host = os.getenv("PGHOST")
    user = os.getenv("PGUSER")
    pwd  = os.getenv("PGPASSWORD")
    db   = os.getenv("PGDATABASE")
    port = os.getenv("PGPORT", "5432")

    missing = [k for k, v in {"PGHOST": host, "PGUSER": user,
                               "PGPASSWORD": pwd, "PGDATABASE": db}.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"


def _load_table(engine, table_name: str) -> pd.DataFrame:
    """Load a table by Pascal or snake name, normalize columns to snake_case."""
    import re

    def to_snake(s: str) -> str:
        s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return s.replace("__", "_").lower()

    # Try exact name first, then case-insensitive
    with engine.connect() as conn:
        q = text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND (table_name = :t OR lower(table_name) = lower(:t))
            ORDER BY CASE WHEN table_schema = 'public' THEN 0 ELSE 1 END
            LIMIT 1
        """)
        row = conn.execute(q, {"t": table_name}).fetchone()

    if row is None:
        raise ValueError(f"Table not found in database: {table_name}")

    schema, tbl = row[0], row[1]
    df = pd.read_sql_query(f'SELECT * FROM "{schema}"."{tbl}"', con=engine)
    df.columns = [to_snake(c) for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD AND PREPROCESS ALL TABLES
# ─────────────────────────────────────────────────────────────────────────────

def load_all_tables(engine):
    print("Loading tables from database...")

    health_df      = _load_table(engine, "HealthWellbeingRecords")
    edu_df         = _load_table(engine, "EducationRecords")
    sessions_df    = _load_table(engine, "ProcessRecordings")
    visitations_df = _load_table(engine, "HomeVisitations")
    incidents_df   = _load_table(engine, "IncidentReports")
    residents_df   = _load_table(engine, "Residents")
    plans_df       = _load_table(engine, "InterventionPlans")

    # Parse dates
    health_df["record_date"]          = pd.to_datetime(health_df["record_date"],          errors="coerce")
    edu_df["record_date"]             = pd.to_datetime(edu_df["record_date"],              errors="coerce")
    sessions_df["session_date"]       = pd.to_datetime(sessions_df["session_date"],        errors="coerce")
    visitations_df["visit_date"]      = pd.to_datetime(visitations_df["visit_date"],       errors="coerce")
    incidents_df["incident_date"]     = pd.to_datetime(incidents_df["incident_date"],      errors="coerce")
    residents_df["date_of_admission"] = pd.to_datetime(residents_df["date_of_admission"],  errors="coerce")

    # Boolean columns from PostgreSQL can come back as strings
    def _boolify(df, cols):
        mapping = {True: 1, False: 0, "true": 1, "false": 0, "t": 1, "f": 0, 1: 1, 0: 0}
        for col in cols:
            if col in df.columns:
                df[col] = df[col].map(mapping).astype(float)
        return df

    sessions_df    = _boolify(sessions_df,    ["concerns_flagged", "progress_noted", "referral_made"])
    health_df      = _boolify(health_df,      ["medical_checkup_done", "dental_checkup_done", "psychological_checkup_done"])
    visitations_df = _boolify(visitations_df, ["safety_concerns_noted", "follow_up_needed"])
    residents_df   = _boolify(residents_df,   [
        "sub_cat_trafficked", "sub_cat_physical_abuse", "sub_cat_sexual_abuse",
        "sub_cat_osaec", "sub_cat_cicl", "sub_cat_at_risk", "is_pwd",
        "has_special_needs", "family_is_4ps", "family_informal_settler",
    ])

    # ── Shared ordinal encodings ──────────────────────────────────────────────
    EMOTIONAL_STATE_RANK = {
        "Distressed": 1, "Angry": 2, "Withdrawn": 3, "Anxious": 4,
        "Sad": 5, "Calm": 6, "Hopeful": 7, "Happy": 8,
    }
    RISK_LEVEL_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    COMPLETION_RANK = {"NotStarted": 0, "InProgress": 1, "Completed": 2}
    EDU_LEVEL_RANK  = {"Primary": 0, "Secondary": 1, "Vocational": 2, "CollegePrep": 3}

    # Sessions
    sessions_df["emotional_start_rank"]  = sessions_df["emotional_state_observed"].map(EMOTIONAL_STATE_RANK)
    sessions_df["emotional_end_rank"]    = sessions_df["emotional_state_end"].map(EMOTIONAL_STATE_RANK)
    sessions_df["emotional_improved"]    = (sessions_df["emotional_end_rank"] > sessions_df["emotional_start_rank"]).astype(int)
    sessions_df["is_distressed_start"]   = (sessions_df["emotional_start_rank"] <= 2).astype(int)

    # Residents
    residents_df["current_risk_encoded"] = residents_df["current_risk_level"].map(RISK_LEVEL_RANK)
    residents_df["initial_risk_encoded"] = residents_df["initial_risk_level"].map(RISK_LEVEL_RANK)
    residents_df["risk_delta"]           = residents_df["current_risk_encoded"] - residents_df["initial_risk_encoded"]

    # Visitations
    visitations_df["cooperation_encoded"] = visitations_df["family_cooperation_level"].map(
        {"Uncooperative": 1, "Neutral": 2, "Cooperative": 3, "Highly Cooperative": 4}
    )
    visitations_df["cooperation_high"]    = visitations_df["family_cooperation_level"].isin(
        ["Cooperative", "Highly Cooperative"]
    ).astype(int)
    visitations_df["uncooperative_flag"]  = (visitations_df["family_cooperation_level"] == "Uncooperative").astype(int)

    # Education
    edu_df["completion_encoded"] = edu_df["completion_status"].map(COMPLETION_RANK)
    edu_df["edu_level_encoded"]  = edu_df["education_level"].map(EDU_LEVEL_RANK)

    print(f"  health_wellbeing_records : {len(health_df)} rows")
    print(f"  education_records        : {len(edu_df)} rows")
    print(f"  process_recordings       : {len(sessions_df)} rows")
    print(f"  home_visitations         : {len(visitations_df)} rows")
    print(f"  incident_reports         : {len(incidents_df)} rows")
    print(f"  residents                : {len(residents_df)} rows")
    print(f"  intervention_plans       : {len(plans_df)} rows")

    return health_df, edu_df, sessions_df, visitations_df, incidents_df, residents_df, plans_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING — one function per model
# ─────────────────────────────────────────────────────────────────────────────

def compute_health_features(resident_id, window_start, window_end,
                             health_df, sessions_df, visitations_df,
                             incidents_df, residents_df):
    feats = {}

    # HEALTH RECORDS
    h = health_df[
        (health_df["resident_id"] == resident_id) &
        (health_df["record_date"] >= window_start) &
        (health_df["record_date"] <= window_end)
    ].sort_values("record_date")

    if len(h) == 0:
        return None

    feats["health_score_current"]  = h["general_health_score"].iloc[-1]
    feats["health_score_mean"]     = h["general_health_score"].mean()
    feats["avg_nutrition_score"]   = h["nutrition_score"].mean()
    feats["avg_sleep_score"]       = h["sleep_quality_score"].mean()
    feats["avg_energy_score"]      = h["energy_level_score"].mean()
    feats["avg_bmi"]               = h["bmi"].mean() if "bmi" in h.columns else np.nan
    feats["medical_checkup_rate"]  = h["medical_checkup_done"].mean()
    feats["dental_checkup_rate"]   = h["dental_checkup_done"].mean()
    feats["psych_checkup_rate"]    = h["psychological_checkup_done"].mean()

    if len(h) >= 2:
        days_x = (h["record_date"] - h["record_date"].iloc[0]).dt.days.values.astype(float)
        feats["health_score_trend"] = float(np.polyfit(days_x, h["general_health_score"].values, 1)[0]) if days_x[-1] > 0 else 0.0
    else:
        feats["health_score_trend"] = 0.0

    # COUNSELING SESSIONS
    s = sessions_df[
        (sessions_df["resident_id"] == resident_id) &
        (sessions_df["session_date"] >= window_start) &
        (sessions_df["session_date"] <= window_end)
    ]
    feats["session_count"]              = len(s)
    feats["sessions_per_month"]         = len(s) / 2.0
    feats["concerns_flagged_rate"]      = s["concerns_flagged"].mean()         if len(s) > 0 else 0.0
    feats["progress_noted_rate"]        = s["progress_noted"].mean()           if len(s) > 0 else 0.0
    feats["emotional_improvement_rate"] = s["emotional_improved"].mean()       if len(s) > 0 else 0.0
    feats["avg_emotional_start"]        = s["emotional_start_rank"].mean()     if len(s) > 0 else 4.0
    feats["avg_emotional_end"]          = s["emotional_end_rank"].mean()       if len(s) > 0 else 4.0
    feats["avg_session_duration"]       = s["session_duration_minutes"].mean() if len(s) > 0 else 0.0
    feats["pct_distressed_sessions"]    = (s["emotional_start_rank"] <= 2).mean() if len(s) > 0 else 0.0
    feats["referral_rate"]              = s["referral_made"].mean()            if len(s) > 0 else 0.0

    # HOME VISITATIONS
    v = visitations_df[
        (visitations_df["resident_id"] == resident_id) &
        (visitations_df["visit_date"] >= window_start) &
        (visitations_df["visit_date"] <= window_end)
    ]
    feats["visit_count"]            = len(v)
    feats["favorable_visit_rate"]   = (v["visit_outcome"] == "Favorable").mean()   if len(v) > 0 else 0.5
    feats["unfavorable_visit_rate"] = (v["visit_outcome"] == "Unfavorable").mean() if len(v) > 0 else 0.0
    feats["safety_concerns_rate"]   = v["safety_concerns_noted"].mean()             if len(v) > 0 else 0.0
    feats["avg_cooperation"]        = v["cooperation_encoded"].mean()               if len(v) > 0 else 2.5
    feats["emergency_visit_flag"]   = int((v["visit_type"] == "Emergency").any())   if len(v) > 0 else 0

    # INCIDENTS
    inc = incidents_df[
        (incidents_df["resident_id"] == resident_id) &
        (incidents_df["incident_date"] >= window_start) &
        (incidents_df["incident_date"] <= window_end)
    ]
    feats["prior_incident_count"]      = len(inc)
    feats["prior_high_severity_count"] = (inc["severity"] == "High").sum()
    feats["any_incident_flag"]         = int(len(inc) > 0)
    feats["days_since_last_incident"]  = (
        (window_end - inc["incident_date"].max()).days if len(inc) > 0 else 999
    )

    # STATIC RESIDENT
    r = residents_df[residents_df["resident_id"] == resident_id]
    if len(r) == 0:
        return None
    r = r.iloc[0]

    feats["current_risk_encoded"]   = r["current_risk_encoded"]   if not pd.isna(r["current_risk_encoded"])   else 2
    feats["initial_risk_encoded"]   = r["initial_risk_encoded"]   if not pd.isna(r["initial_risk_encoded"])   else 2
    feats["risk_delta"]             = r["risk_delta"]             if not pd.isna(r["risk_delta"])             else 0
    feats["has_special_needs"]      = int(r["has_special_needs"]) if not pd.isna(r["has_special_needs"])      else 0
    feats["is_pwd"]                 = int(r["is_pwd"])            if not pd.isna(r["is_pwd"])                 else 0
    feats["sub_cat_trafficked"]     = int(r.get("sub_cat_trafficked", 0)     or 0)
    feats["sub_cat_sexual_abuse"]   = int(r.get("sub_cat_sexual_abuse", 0)   or 0)
    feats["sub_cat_physical_abuse"] = int(r.get("sub_cat_physical_abuse", 0) or 0)
    feats["family_is_4ps"]          = int(r.get("family_is_4ps", 0)          or 0)
    feats["length_of_stay_days"]    = max(0, (window_end - r["date_of_admission"]).days) if not pd.isna(r.get("date_of_admission")) else 0

    return feats


def compute_edu_features(resident_id, window_start, window_end,
                          edu_df, sessions_df, visitations_df,
                          incidents_df, plans_df, residents_df):
    feats = {}

    # EDUCATION RECORDS
    e = edu_df[
        (edu_df["resident_id"] == resident_id) &
        (edu_df["record_date"] >= window_start) &
        (edu_df["record_date"] <= window_end)
    ].sort_values("record_date")

    if len(e) == 0:
        return None

    feats["progress_percent_current"]  = e["progress_percent"].iloc[-1]
    feats["progress_percent_mean"]     = e["progress_percent"].mean()
    feats["avg_attendance_rate"]       = e["attendance_rate"].mean()
    feats["min_attendance_rate"]       = e["attendance_rate"].min()
    feats["completion_status_encoded"] = e["completion_encoded"].iloc[-1] if not pd.isna(e["completion_encoded"].iloc[-1]) else 1
    feats["edu_level_encoded"]         = e["edu_level_encoded"].iloc[-1]  if not pd.isna(e["edu_level_encoded"].iloc[-1])  else 1

    if len(e) >= 2:
        days_x = (e["record_date"] - e["record_date"].iloc[0]).dt.days.values.astype(float)
        feats["progress_trend"]   = float(np.polyfit(days_x, e["progress_percent"].values, 1)[0]) if days_x[-1] > 0 else 0.0
        feats["attendance_trend"] = float(np.polyfit(days_x, e["attendance_rate"].values, 1)[0])  if days_x[-1] > 0 else 0.0
    else:
        feats["progress_trend"]   = 0.0
        feats["attendance_trend"] = 0.0

    # COUNSELING SESSIONS
    s = sessions_df[
        (sessions_df["resident_id"] == resident_id) &
        (sessions_df["session_date"] >= window_start) &
        (sessions_df["session_date"] <= window_end)
    ]
    feats["session_count"]              = len(s)
    feats["concerns_flagged_rate"]      = s["concerns_flagged"].mean()      if len(s) > 0 else 0.0
    feats["progress_noted_rate"]        = s["progress_noted"].mean()        if len(s) > 0 else 0.0
    feats["avg_emotional_start"]        = s["emotional_start_rank"].mean()  if len(s) > 0 else 4.0
    feats["emotional_improvement_rate"] = s["emotional_improved"].mean()    if len(s) > 0 else 0.0
    feats["referral_rate"]              = s["referral_made"].mean()         if len(s) > 0 else 0.0

    # HOME VISITATIONS
    v = visitations_df[
        (visitations_df["resident_id"] == resident_id) &
        (visitations_df["visit_date"] >= window_start) &
        (visitations_df["visit_date"] <= window_end)
    ]
    feats["favorable_visit_rate"]    = (v["visit_outcome"] == "Favorable").mean() if len(v) > 0 else 0.5
    feats["family_cooperation_rate"] = v["cooperation_high"].mean()               if len(v) > 0 else 0.5
    feats["safety_concerns_rate"]    = v["safety_concerns_noted"].mean()          if len(v) > 0 else 0.0
    feats["visit_count"]             = len(v)

    # INCIDENTS
    inc = incidents_df[
        (incidents_df["resident_id"] == resident_id) &
        (incidents_df["incident_date"] >= window_start) &
        (incidents_df["incident_date"] <= window_end)
    ]
    feats["prior_incident_count"]      = len(inc)
    feats["prior_high_severity_count"] = (inc["severity"] == "High").sum()
    feats["any_incident_flag"]         = int(len(inc) > 0)

    # INTERVENTION PLANS (all time, not windowed)
    try:
        p         = plans_df[plans_df["resident_id"] == resident_id]
        edu_plans = p[p["plan_category"] == "Education"]
        feats["education_plan_active"] = int(len(edu_plans[edu_plans["status"].isin(["Open", "In Progress"])]) > 0)
        feats["pct_plans_achieved"]    = (p["status"] == "Achieved").mean() if len(p) > 0 else 0.0
        feats["plans_on_hold"]         = int((p["status"] == "On Hold").any())
    except Exception:
        feats["education_plan_active"] = 0
        feats["pct_plans_achieved"]    = 0.0
        feats["plans_on_hold"]         = 0

    # STATIC RESIDENT
    r = residents_df[residents_df["resident_id"] == resident_id]
    if len(r) == 0:
        return None
    r = r.iloc[0]

    feats["current_risk_encoded"] = r["current_risk_encoded"]  if not pd.isna(r["current_risk_encoded"])  else 2
    feats["has_special_needs"]    = int(r["has_special_needs"]) if not pd.isna(r["has_special_needs"])     else 0
    feats["is_pwd"]               = int(r.get("is_pwd", 0) or 0)
    feats["sub_cat_trafficked"]   = int(r.get("sub_cat_trafficked", 0)   or 0)
    feats["sub_cat_sexual_abuse"] = int(r.get("sub_cat_sexual_abuse", 0) or 0)
    feats["sub_cat_cicl"]         = int(r.get("sub_cat_cicl", 0)         or 0)
    feats["family_is_4ps"]        = int(r.get("family_is_4ps", 0)        or 0)
    feats["length_of_stay_days"]  = max(0, (window_end - r["date_of_admission"]).days) if not pd.isna(r.get("date_of_admission")) else 0

    return feats


def compute_emotional_features(resident_id, window_start, window_end,
                                sessions_df, health_df, visitations_df,
                                incidents_df, residents_df):
    feats = {}

    # COUNSELING SESSIONS (primary — need at least 3)
    s = sessions_df[
        (sessions_df["resident_id"] == resident_id) &
        (sessions_df["session_date"] >= window_start) &
        (sessions_df["session_date"] <= window_end)
    ].sort_values("session_date")

    if len(s) < 3:
        return None

    feats["session_count"]                    = len(s)
    feats["sessions_per_month"]               = len(s) / 2.0
    feats["avg_session_duration"]             = s["session_duration_minutes"].mean()
    feats["concerns_flagged_rate"]            = s["concerns_flagged"].mean()
    feats["progress_noted_rate_prior"]        = s["progress_noted"].mean()
    feats["referral_rate"]                    = s["referral_made"].mean()
    feats["avg_emotional_start"]              = s["emotional_start_rank"].mean()
    feats["avg_emotional_end"]                = s["emotional_end_rank"].mean()
    feats["emotional_improvement_rate_prior"] = s["emotional_improved"].mean()
    feats["pct_distressed_sessions"]          = s["is_distressed_start"].mean()
    feats["emotional_volatility"]             = s["emotional_start_rank"].std() if len(s) > 1 else 0.0
    feats["pct_individual_sessions"]          = (s["session_type"] == "Individual").mean() if "session_type" in s.columns else 0.5

    if len(s) >= 3:
        days_x = (s["session_date"] - s["session_date"].iloc[0]).dt.days.values.astype(float)
        feats["emotional_start_trend"] = float(np.polyfit(days_x, s["emotional_start_rank"].values, 1)[0]) if days_x[-1] > 0 else 0.0
    else:
        feats["emotional_start_trend"] = 0.0

    # HEALTH RECORDS
    h = health_df[
        (health_df["resident_id"] == resident_id) &
        (health_df["record_date"] >= window_start) &
        (health_df["record_date"] <= window_end)
    ]
    feats["avg_health_score"]    = h["general_health_score"].mean() if len(h) > 0 else 3.0
    feats["avg_sleep_score"]     = h["sleep_quality_score"].mean()  if len(h) > 0 else 3.0
    feats["avg_nutrition_score"] = h["nutrition_score"].mean()      if len(h) > 0 else 3.0
    feats["psych_checkup_rate"]  = h["psychological_checkup_done"].mean() if len(h) > 0 else 0.0

    # HOME VISITATIONS
    v = visitations_df[
        (visitations_df["resident_id"] == resident_id) &
        (visitations_df["visit_date"] >= window_start) &
        (visitations_df["visit_date"] <= window_end)
    ]
    feats["favorable_visit_rate"]      = (v["visit_outcome"] == "Favorable").mean() if len(v) > 0 else 0.5
    feats["safety_concerns_rate"]      = v["safety_concerns_noted"].mean()          if len(v) > 0 else 0.0
    feats["uncooperative_family_rate"] = v["uncooperative_flag"].mean()             if len(v) > 0 else 0.0
    feats["visit_count"]               = len(v)

    # INCIDENTS
    inc = incidents_df[
        (incidents_df["resident_id"] == resident_id) &
        (incidents_df["incident_date"] >= window_start) &
        (incidents_df["incident_date"] <= window_end)
    ]
    feats["prior_high_severity_count"] = (inc["severity"] == "High").sum()
    feats["any_incident_flag"]         = int(len(inc) > 0)
    feats["days_since_last_incident"]  = (
        (window_end - inc["incident_date"].max()).days if len(inc) > 0 else 999
    )

    # STATIC RESIDENT
    r = residents_df[residents_df["resident_id"] == resident_id]
    if len(r) == 0:
        return None
    r = r.iloc[0]

    feats["current_risk_encoded"]   = r.get("current_risk_encoded", 2)
    feats["initial_risk_encoded"]   = r.get("initial_risk_encoded", 2)
    feats["has_special_needs"]      = int(r.get("has_special_needs", 0) or 0)
    feats["is_pwd"]                 = int(r.get("is_pwd", 0)            or 0)
    feats["sub_cat_trafficked"]     = int(r.get("sub_cat_trafficked", 0)     or 0)
    feats["sub_cat_sexual_abuse"]   = int(r.get("sub_cat_sexual_abuse", 0)   or 0)
    feats["sub_cat_physical_abuse"] = int(r.get("sub_cat_physical_abuse", 0) or 0)
    feats["length_of_stay_days"]    = max(0, (window_end - r["date_of_admission"]).days) if not pd.isna(r.get("date_of_admission")) else 0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD TRAINING DATAFRAME — rolling window per model
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_WINDOW_DAYS = 60
LABEL_WINDOW_DAYS   = 60


def build_health_training_df(health_df, sessions_df, visitations_df, incidents_df, residents_df):
    IMPROVEMENT_THRESHOLD = 0.1
    rows = []

    for resident_id, group in health_df.groupby("resident_id"):
        group = group.sort_values("record_date").reset_index(drop=True)
        for _, row in group.iterrows():
            T            = row["record_date"]
            score_at_T   = row["general_health_score"]
            window_start = T - pd.Timedelta(days=FEATURE_WINDOW_DAYS)
            label_end    = T + pd.Timedelta(days=LABEL_WINDOW_DAYS)

            future = group[(group["record_date"] > T) & (group["record_date"] <= label_end)]
            if len(future) == 0:
                continue

            score_at_T60 = future.sort_values("record_date").iloc[-1]["general_health_score"]
            delta        = score_at_T60 - score_at_T
            y_label      = 1 if delta >= IMPROVEMENT_THRESHOLD else 0

            feats = compute_health_features(
                resident_id, window_start, T,
                health_df, sessions_df, visitations_df, incidents_df, residents_df
            )
            if feats is None:
                continue

            feats["health_improved"] = y_label
            rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"  Health training examples: {len(df)} | Y=1: {df['health_improved'].sum()} ({df['health_improved'].mean()*100:.1f}%)")
    return df


def build_edu_training_df(edu_df, sessions_df, visitations_df, incidents_df, plans_df, residents_df):
    PROGRESS_THRESHOLD   = 5.0
    ATTENDANCE_THRESHOLD = 0.75
    rows = []

    for resident_id, group in edu_df.groupby("resident_id"):
        group = group.sort_values("record_date").reset_index(drop=True)
        for _, row in group.iterrows():
            T            = row["record_date"]
            prog_at_T    = row["progress_percent"]
            window_start = T - pd.Timedelta(days=FEATURE_WINDOW_DAYS)
            label_end    = T + pd.Timedelta(days=LABEL_WINDOW_DAYS)

            future = group[(group["record_date"] > T) & (group["record_date"] <= label_end)]
            if len(future) < 1:
                continue

            prog_at_T60             = future.sort_values("record_date").iloc[-1]["progress_percent"]
            avg_attend_label_window = future["attendance_rate"].mean()
            prog_delta              = prog_at_T60 - prog_at_T
            y_label                 = 1 if (prog_delta >= PROGRESS_THRESHOLD and avg_attend_label_window >= ATTENDANCE_THRESHOLD) else 0

            feats = compute_edu_features(
                resident_id, window_start, T,
                edu_df, sessions_df, visitations_df, incidents_df, plans_df, residents_df
            )
            if feats is None:
                continue

            feats["edu_progressed"] = y_label
            rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"  Education training examples: {len(df)} | Y=1: {df['edu_progressed'].sum()} ({df['edu_progressed'].mean()*100:.1f}%)")
    return df


def build_emotional_training_df(sessions_df, health_df, visitations_df, incidents_df, residents_df):
    EMO_IMPROVEMENT_THRESH = 0.80
    PROGRESS_NOTED_THRESH  = 0.90
    MIN_LABEL_SESSIONS     = 3
    rows = []

    for resident_id, group in sessions_df.groupby("resident_id"):
        group       = group.sort_values("session_date").reset_index(drop=True)
        session_dates = group["session_date"].unique()

        for T in session_dates:
            window_start   = T - pd.Timedelta(days=FEATURE_WINDOW_DAYS)
            label_end      = T + pd.Timedelta(days=LABEL_WINDOW_DAYS)
            label_sessions = group[
                (group["session_date"] > T) &
                (group["session_date"] <= label_end)
            ]
            if len(label_sessions) < MIN_LABEL_SESSIONS:
                continue

            emo_rate      = label_sessions["emotional_improved"].mean()
            progress_rate = label_sessions["progress_noted"].mean()
            y_label       = 1 if (emo_rate >= EMO_IMPROVEMENT_THRESH and progress_rate >= PROGRESS_NOTED_THRESH) else 0

            feats = compute_emotional_features(
                resident_id, window_start, T,
                sessions_df, health_df, visitations_df, incidents_df, residents_df
            )
            if feats is None:
                continue

            feats["emotional_progressed"] = y_label
            rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"  Emotional training examples: {len(df)} | Y=1: {df['emotional_progressed'].sum()} ({df['emotional_progressed'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_model(training_df, target_col, label):
    meta_cols    = [c for c in training_df.columns if c.startswith("_")]
    feature_cols = [c for c in training_df.columns if c not in meta_cols + [target_col]]

    modeling_df = training_df[feature_cols + [target_col]].copy()
    for col in feature_cols:
        if modeling_df[col].isnull().any():
            modeling_df[col].fillna(modeling_df[col].median(), inplace=True)

    pipe = MLPipeline(
        df=modeling_df,
        target=target_col,
        models=["lr", "dt", "knn", "rf", "gb", "ada"],
        tune=True,
        output_path=f"{label}_progress_model_temp.joblib",  # temp file, not used for inference
        cat_strategy="onehot",
        scale=True,
        test_size=0.2,
        random_state=RANDOM_STATE,
        cv_folds=5,
        verbose=False,
    )
    results = pipe.run()
    best_key = pipe.best_model_key
    best_auc = results[best_key]["roc_auc"]
    print(f"  {label} — best model: {best_key} | test AUC: {best_auc:.4f}")

    # Return the trained pipeline and the feature column list
    return pipe.final_pipeline, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD INFERENCE FEATURES — current 60-day window for each active resident
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_rows(residents_df, health_df, edu_df, sessions_df,
                          visitations_df, incidents_df, plans_df):
    """
    Build inference feature rows for each active resident.

    Window anchor: instead of using today's date (which would find nothing
    in a demo dataset whose records end in 2025), we use each resident's
    most recent record date as the anchor. This ensures the 60-day feature
    window always captures real data regardless of when the pipeline runs.
    """
    active = residents_df[residents_df["case_status"] == "Active"]

    health_rows    = []
    edu_rows       = []
    emotional_rows = []

    for _, res in active.iterrows():
        rid = res["resident_id"]

        # ── Health: anchor on last health record date ─────────────────────────
        h_all = health_df[health_df["resident_id"] == rid].sort_values("record_date")
        if len(h_all) > 0:
            anchor       = h_all["record_date"].iloc[-1]
            window_start = anchor - pd.Timedelta(days=60)
            h_feats = compute_health_features(
                rid, window_start, anchor,
                health_df, sessions_df, visitations_df, incidents_df, residents_df
            )
            if h_feats:
                h_feats["resident_id"] = rid
                health_rows.append(h_feats)

        # ── Education: anchor on last education record date ───────────────────
        e_all = edu_df[edu_df["resident_id"] == rid].sort_values("record_date")
        if len(e_all) > 0:
            anchor       = e_all["record_date"].iloc[-1]
            window_start = anchor - pd.Timedelta(days=60)
            e_feats = compute_edu_features(
                rid, window_start, anchor,
                edu_df, sessions_df, visitations_df, incidents_df, plans_df, residents_df
            )
            if e_feats:
                e_feats["resident_id"] = rid
                edu_rows.append(e_feats)

        # ── Emotional: anchor on last session date ────────────────────────────
        s_all = sessions_df[sessions_df["resident_id"] == rid].sort_values("session_date")
        if len(s_all) > 0:
            anchor       = s_all["session_date"].iloc[-1]
            window_start = anchor - pd.Timedelta(days=60)
            emo_feats = compute_emotional_features(
                rid, window_start, anchor,
                sessions_df, health_df, visitations_df, incidents_df, residents_df
            )
            if emo_feats:
                emo_feats["resident_id"] = rid
                emotional_rows.append(emo_feats)

    print(f"  Inference rows — health: {len(health_rows)} | edu: {len(edu_rows)} | emotional: {len(emotional_rows)}")

    # Return empty DataFrames with a resident_id column so downstream
    # code never crashes on a missing column when rows is empty
    def _safe_df(rows):
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["resident_id"])

    return _safe_df(health_rows), _safe_df(edu_rows), _safe_df(emotional_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. RUN PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(health_model, health_feature_cols,
                    edu_model,    edu_feature_cols,
                    emo_model,    emo_feature_cols,
                    health_inf, edu_inf, emo_inf,
                    residents_df):

    active = residents_df[residents_df["case_status"] == "Active"]
    now    = datetime.now(timezone.utc)
    results = []

    for _, res in active.iterrows():
        rid = int(res["resident_id"])

        # ── Health ────────────────────────────────────────────────────────────
        health_prob = None
        health_tag  = None
        if "resident_id" in health_inf.columns and len(health_inf) > 0:
            h_row = health_inf[health_inf["resident_id"] == rid]
            if len(h_row) > 0:
                X_h = h_row.reindex(columns=health_feature_cols).copy()
                health_prob = float(health_model.predict_proba(X_h)[:, 1][0])
                current_score = h_row["health_score_current"].values[0]
                if current_score >= 4.5:
                    health_tag = "Already High — Stable"

        # ── Education ─────────────────────────────────────────────────────────
        edu_prob = None
        if "resident_id" in edu_inf.columns and len(edu_inf) > 0:
            e_row = edu_inf[edu_inf["resident_id"] == rid]
            if len(e_row) > 0:
                X_e = e_row.reindex(columns=edu_feature_cols).copy()
                edu_prob = float(edu_model.predict_proba(X_e)[:, 1][0])

        # ── Emotional ─────────────────────────────────────────────────────────
        emo_prob = None
        if "resident_id" in emo_inf.columns and len(emo_inf) > 0:
            emo_row = emo_inf[emo_inf["resident_id"] == rid]
            if len(emo_row) > 0:
                X_emo = emo_row.reindex(columns=emo_feature_cols).copy()
                emo_prob = float(emo_model.predict_proba(X_emo)[:, 1][0])

        # ── Overall score — average only available probabilities ──────────────
        available = [p for p in [health_prob, edu_prob, emo_prob] if p is not None]
        overall   = float(np.mean(available)) if available else None

        results.append({
            "ResidentId":    rid,
            "HealthProb":    health_prob,
            "EducationProb": edu_prob,
            "EmotionalProb": emo_prob,
            "OverallScore":  overall,
            "HealthTag":     health_tag,
            "PredictedAt":   now,
            "ModelVersion":  "1.0",
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 8. WRITE PREDICTIONS TO DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def upsert_predictions(engine, predictions_df):
    """
    Write predictions to ResidentPredictions table.
    Uses engine.begin() which auto-commits on success and auto-rolls back
    on any exception — the most reliable pattern for Azure PostgreSQL
    with SQLAlchemy 2.0.

    Strategy: CREATE TABLE IF NOT EXISTS, then TRUNCATE + INSERT each run.
    Simpler and more reliable than row-by-row upsert. Safe because this
    table is purely derived output, never a source of truth.
    """
    with engine.begin() as conn:
        # Create table if it does not exist yet
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "ResidentPredictions" (
                "ResidentId"     INTEGER PRIMARY KEY,
                "HealthProb"     FLOAT,
                "EducationProb"  FLOAT,
                "EmotionalProb"  FLOAT,
                "OverallScore"   FLOAT,
                "HealthTag"      VARCHAR(100),
                "PredictedAt"    TIMESTAMPTZ,
                "ModelVersion"   VARCHAR(20)
            )
        """))

        # Clear existing predictions and write fresh ones
        conn.execute(text('TRUNCATE TABLE "ResidentPredictions"'))

        for _, row in predictions_df.iterrows():
            conn.execute(text("""
                INSERT INTO "ResidentPredictions"
                    ("ResidentId", "HealthProb", "EducationProb", "EmotionalProb",
                     "OverallScore", "HealthTag", "PredictedAt", "ModelVersion")
                VALUES
                    (:ResidentId, :HealthProb, :EducationProb, :EmotionalProb,
                     :OverallScore, :HealthTag, :PredictedAt, :ModelVersion)
            """), row.to_dict())

    # If we reach here the transaction committed successfully
    print(f"  Wrote {len(predictions_df)} rows into ResidentPredictions")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"Havyn ML Pipeline — {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}\n")

    # Connect
    engine = create_engine(_build_db_url())
    print("Database connected.\n")

    # Load tables
    (health_df, edu_df, sessions_df,
     visitations_df, incidents_df,
     residents_df, plans_df) = load_all_tables(engine)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Training health model...")
    health_train              = build_health_training_df(health_df, sessions_df, visitations_df, incidents_df, residents_df)
    health_model, health_cols = train_model(health_train, "health_improved", "health")

    print("\n[2/3] Training education model...")
    edu_train              = build_edu_training_df(edu_df, sessions_df, visitations_df, incidents_df, plans_df, residents_df)
    edu_model, edu_cols    = train_model(edu_train, "edu_progressed", "education")

    print("\n[3/3] Training emotional model...")
    emo_train              = build_emotional_training_df(sessions_df, health_df, visitations_df, incidents_df, residents_df)
    emo_model, emo_cols    = train_model(emo_train, "emotional_progressed", "emotional")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nBuilding inference features for active residents...")
    health_inf, edu_inf, emo_inf = build_inference_rows(
        residents_df, health_df, edu_df, sessions_df,
        visitations_df, incidents_df, plans_df
    )

    print("Running predictions...")
    predictions = run_predictions(
        health_model, health_cols,
        edu_model,    edu_cols,
        emo_model,    emo_cols,
        health_inf, edu_inf, emo_inf,
        residents_df,
    )

    # ── Write to DB ───────────────────────────────────────────────────────────
    print("Writing predictions to database...")
    upsert_predictions(engine, predictions)

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed}s — {len(predictions)} residents scored")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()