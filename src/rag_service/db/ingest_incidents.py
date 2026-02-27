from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import insert
from rag_service.core.config import settings
from rag_service.db.session import engine
from rag_service.db.models import Incident, Severity

VALID_SEVERITIES = {s.value for s in Severity}


def _iter_incident_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            yield row.to_dict()
        return

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError("JSON incidents must be a list or {records: [...]} object.")
        for rec in data:
            yield rec
        return

    raise ValueError(f"Unsupported incidents format: {path}")


def ingest_incidents(raw_incidents_dir: str | None = None) -> int:
    base = Path(raw_incidents_dir or settings.raw_incidents_dir)
    files = sorted([p for p in base.glob("*") if p.suffix.lower() in {".csv", ".json"}])

    if not files:
        raise FileNotFoundError(f"No incident files found in {base}")

    inserted = 0
    with engine.begin() as conn:
        for f in files:
            rows = []
            for rec in _iter_incident_rows(f):
                sev = str(rec["severity"]).lower().strip()
                if sev not in VALID_SEVERITIES:
                    raise ValueError(f"Invalid severity '{sev}' in {f.name}")
                rows.append(
                    dict(
                        timestamp=pd.to_datetime(rec["timestamp"], utc=True).to_pydatetime(),
                        equipment_id=str(rec["equipment_id"]),
                        sensor_data=(
                            rec["sensor_data"]
                            if isinstance(rec["sensor_data"], dict)
                            else json.loads(rec["sensor_data"])
                        ),
                        failure_mode=str(rec["failure_mode"]),
                        severity=Severity(sev),
                        actions_taken=str(rec["actions_taken"]),
                        outcome=str(rec["outcome"]),
                        resolution_time_hours=float(rec["resolution_time_hours"]),
                    )
                )

            if rows:
                conn.execute(insert(Incident), rows)
                inserted += len(rows)

    return inserted


if __name__ == "__main__":
    n = ingest_incidents()
    print(f"Inserted {n} incident rows.")
