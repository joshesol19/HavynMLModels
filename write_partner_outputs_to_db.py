"""
write_partner_outputs_to_db.py
Runs after all three partner notebooks complete.

Does two things:
  1. Writes per-resident incident risk rows into ResidentIncidentRisk table
  2. Writes the three aggregate JSON outputs into a single row in MlModelMeta table
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

for candidate in [Path(__file__).parent / ".env", Path(".env")]:
    if candidate.is_file():
        load_dotenv(candidate, override=True)
        break

def _build_db_url():
    host = os.getenv("PGHOST")
    user = os.getenv("PGUSER")
    pwd  = os.getenv("PGPASSWORD")
    db   = os.getenv("PGDATABASE")
    port = os.getenv("PGPORT", "5432")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected notebook output not found: {path}")
    with open(path) as f:
        return json.load(f)

def main():
    # ── Load the three JSON outputs ───────────────────────────────────────────
    incident_payload      = load_json(Path("Partner Models/model_outputs_incident/incident_risk_scores.json"))
    social_media_payload  = load_json(Path("Partner Models/model_outputs/platform_recommendations_simple.json"))
    reintegration_payload = load_json(Path("Partner Models/model_outputs_reintegration/reintegration_model_results.json"))

    engine = create_engine(_build_db_url())
    now    = datetime.now(timezone.utc)

    with engine.begin() as conn:

        # ── Table 1: ResidentIncidentRisk — one row per resident ──────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "ResidentIncidentRisk" (
                "ResidentId"       INTEGER PRIMARY KEY,
                "RiskTier"         VARCHAR(50),
                "FlaggedForReview" BOOLEAN,
                "TopRiskFactors"   JSONB,
                "ScoredAt"         TIMESTAMPTZ
            )
        """))
        conn.execute(text('TRUNCATE TABLE "ResidentIncidentRisk"'))

        for row in incident_payload["resident_scores"]:
            conn.execute(text("""
                INSERT INTO "ResidentIncidentRisk"
                    ("ResidentId", "RiskTier", "FlaggedForReview", "TopRiskFactors", "ScoredAt")
                VALUES
                    (:rid, :tier, :flagged, CAST(:factors AS JSONB), :scored_at)
            """), {
                "rid":       row["resident_id"],
                "tier":      row["risk_tier"],
                "flagged":   row["flagged_for_review"],
                "factors":   json.dumps(row["top_risk_factors"]),
                "scored_at": now,
            })

        print(f"  Wrote {len(incident_payload['resident_scores'])} rows into ResidentIncidentRisk")

        # ── Table 2: MlModelMeta — one row, three jsonb columns ───────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "MlModelMeta" (
                "Id"                      INTEGER PRIMARY KEY DEFAULT 1,
                "IncidentRiskFactors"     JSONB,
                "SocialMediaRecs"         JSONB,
                "ReintegrationModel"      JSONB,
                "UpdatedAt"               TIMESTAMPTZ,
                CONSTRAINT single_row CHECK ("Id" = 1)
            )
        """))

        # Extract just the causal/aggregate parts — not per-resident data
        incident_causal = {
            "risk_factors": incident_payload["risk_factors"],
            "meta":         incident_payload["meta"],
        }

        conn.execute(text("""
            INSERT INTO "MlModelMeta"
                ("Id", "IncidentRiskFactors", "SocialMediaRecs", "ReintegrationModel", "UpdatedAt")
            VALUES
                (1, CAST(:incident AS JSONB), CAST(:social AS JSONB), CAST(:reintegration AS JSONB), :updated_at)
            ON CONFLICT ("Id") DO UPDATE SET
                "IncidentRiskFactors" = EXCLUDED."IncidentRiskFactors",
                "SocialMediaRecs"     = EXCLUDED."SocialMediaRecs",
                "ReintegrationModel"  = EXCLUDED."ReintegrationModel",
                "UpdatedAt"           = EXCLUDED."UpdatedAt"
        """), {
            "incident":      json.dumps(incident_causal),
            "social":        json.dumps(social_media_payload),
            "reintegration": json.dumps(reintegration_payload),
            "updated_at":    now,
        })

        print("  Wrote 1 row into MlModelMeta (IncidentRiskFactors, SocialMediaRecs, ReintegrationModel)")

if __name__ == "__main__":
    main()