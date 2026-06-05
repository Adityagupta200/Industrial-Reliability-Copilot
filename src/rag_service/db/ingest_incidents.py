from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from rag_service.core.config import settings
from rag_service.db.session import engine
from rag_service.db.models import Incident, Severity

VALID_SEVERITIES = {s.value for s in Severity}
INCIDENT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "industrial-reliability-copilot/incident-seed/v1",
)


def incident_record_id(record: dict) -> uuid.UUID:
    """Stable ID for idempotent seed-data ingestion across local and EKS runs."""
    canonical = {
        "timestamp": str(record["timestamp"]),
        "equipment_id": str(record["equipment_id"]),
        "failure_mode": str(record["failure_mode"]),
        "severity": str(record["severity"]).lower().strip(),
        "actions_taken": str(record["actions_taken"]),
        "outcome": str(record["outcome"]),
        "resolution_time_hours": str(record["resolution_time_hours"]),
        "sensor_data": (
            record["sensor_data"]
            if isinstance(record["sensor_data"], dict)
            else json.loads(record["sensor_data"])
        ),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return uuid.uuid5(INCIDENT_NAMESPACE, payload)


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
                        id=incident_record_id(rec),
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
                stmt = (
                    insert(Incident)
                    .values(rows)
                    .on_conflict_do_nothing(index_elements=[Incident.id])
                )
                result = conn.execute(stmt)
                inserted += max(int(result.rowcount or 0), 0)

    return inserted


if __name__ == "__main__":
    n = ingest_incidents()
    print(f"Inserted {n} incident rows.")
