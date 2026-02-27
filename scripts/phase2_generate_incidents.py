from __future__ import annotations

import random
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


@dataclass
class IncidentRecord:
    timestamp: str
    equipment_id: str
    sensor_data: dict
    failure_mode: str
    severity: str
    actions_taken: str
    outcome: str
    resolution_time_hours: float


FAILURE_MODES = [
    "bearing_failure",
    "sensor_malfunction",
    "lubrication_issue",
    "overheating",
    "cavitation",
    "scheduled_maintenance",
]

SEVERITIES = ["low", "medium", "high", "critical"]


def _rand_ts(days_back: int = 365) -> datetime:
    now = datetime.now(timezone.utc)
    return now - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def _sensor_bundle(mode: str) -> dict:
    base = {
        "vibration_rms": round(random.uniform(0.1, 6.0), 3),
        "temp_c": round(random.uniform(20, 120), 2),
        "pressure_bar": round(random.uniform(0.5, 15.0), 2),
        "flow_rate_lpm": round(random.uniform(10, 400), 1),
    }
    if mode == "bearing_failure":
        base["vibration_rms"] = round(random.uniform(3.5, 8.5), 3)
    if mode == "overheating":
        base["temp_c"] = round(random.uniform(85, 140), 2)
    if mode == "cavitation":
        base["pressure_bar"] = round(random.uniform(0.5, 3.0), 2)
        base["flow_rate_lpm"] = round(random.uniform(50, 120), 1)
    return base


def generate(n: int = 150) -> list[IncidentRecord]:
    records: list[IncidentRecord] = []
    for _ in range(n):
        mode = random.choice(FAILURE_MODES)
        equip = random.choice(
            ["pump_P-23", "pump_P-07", "motor_M-12", "compressor_C-02", "turbofan_TF-01"]
        )
        sev = random.choices(SEVERITIES, weights=[0.35, 0.35, 0.2, 0.1], k=1)[0]
        ts = _rand_ts().isoformat()

        actions = {
            "bearing_failure": "Inspected bearing housing; replaced bearing; verified alignment; relubricated.",
            "sensor_malfunction": "Re-seated sensor; checked wiring; replaced faulty transducer; recalibrated.",
            "lubrication_issue": "Checked oil level; replaced filter; flushed line; refilled per spec.",
            "overheating": "Checked cooling; cleaned vents; reduced load; verified thermal sensors.",
            "cavitation": "Adjusted inlet conditions; inspected suction line; removed blockage; re-tested.",
            "scheduled_maintenance": "Performed scheduled PM; replaced consumables; updated CMMS notes.",
        }[mode]

        outcome = random.choice(
            [
                "Resolved; equipment returned to service.",
                "Mitigated; monitor for 48 hours.",
                "Escalated; further inspection required.",
            ]
        )

        rt = round(
            random.uniform(0.5, 2.5) if sev in {"low", "medium"} else random.uniform(2.0, 12.0),
            2,
        )

        records.append(
            IncidentRecord(
                timestamp=ts,
                equipment_id=equip,
                sensor_data=_sensor_bundle(mode),
                failure_mode=mode,
                severity=sev,
                actions_taken=actions,
                outcome=outcome,
                resolution_time_hours=rt,
            )
        )
    return records


def main() -> None:
    out_dir = Path("data/raw/incidents")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = generate(180)

    # CSV
    df = pd.DataFrame(
        [
            {
                **asdict(r),
                "sensor_data": json.dumps(r.sensor_data, ensure_ascii=False),
            }
            for r in records
        ]
    )
    df.to_csv(out_dir / "synthetic_incidents.csv", index=False)

    # JSON
    (out_dir / "synthetic_incidents.json").write_text(
        json.dumps({"records": [asdict(r) for r in records]}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} records to {out_dir}")


if __name__ == "__main__":
    main()
