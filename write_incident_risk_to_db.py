"""
write_incident_risk_to_db.py
Reads incident_risk_scores.json produced by the incident risk notebook
and writes per-resident rows into ResidentIncidentRisk table in PostgreSQL.
Also writes the global risk factors into MlModelMeta table.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env if running locally
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

def main():
    json_path = Path("Partner Models/model_outputs_incident/incident_risk_scores.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Expected output not found: {json_path}")

    with open(json_path) as f:
        payload = json.load(f)

    resident_scores = payload["resident_scores"]   # per-resident rows
    risk_factors    = payload["risk_factors"]       # global causal factors
    meta            = payload["meta"]

    engine = create_engine(_build_db_url())
    now    = datetime.now(timezone.utc)

    with engine.begin() as conn:

        # ── Table 1: ResidentIncidentRisk — one row per resident ─────────────
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

        for row in resident_scores:
            conn.execute(text("""
                INSERT INTO "ResidentIncidentRisk"
                    ("ResidentId", "RiskTier", "FlaggedForReview", "TopRiskFactors", "ScoredAt")
                VALUES
                    (:rid, :tier, :flagged, :factors::jsonb, :scored_at)
            """), {
                "rid":       row["resident_id"],
                "tier":      row["risk_tier"],
                "flagged":   row["flagged_for_review"],
                "factors":   json.dumps(row["top_risk_factors"]),
                "scored_at": now,
            })

        print(f"  Wrote {len(resident_scores)} rows into ResidentIncidentRisk")

        # ── Table 2: MlModelMeta — global risk factors for analytics page ────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "MlModelMeta" (
                "ModelKey"   VARCHAR(100) PRIMARY KEY,
                "Payload"    JSONB,
                "UpdatedAt"  TIMESTAMPTZ
            )
        """))

        conn.execute(text("""
            INSERT INTO "MlModelMeta" ("ModelKey", "Payload", "UpdatedAt")
            VALUES ('incident_risk_factors', :payload::jsonb, :updated_at)
            ON CONFLICT ("ModelKey") DO UPDATE
                SET "Payload" = EXCLUDED."Payload",
                    "UpdatedAt" = EXCLUDED."UpdatedAt"
        """), {
            "payload":    json.dumps({"risk_factors": risk_factors, "meta": meta}),
            "updated_at": now,
        })

        print("  Wrote incident risk factors into MlModelMeta")

if __name__ == "__main__":
    main()