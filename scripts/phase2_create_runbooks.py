from __future__ import annotations

from pathlib import Path


RUNBOOKS = [
    (
        "bearing_replacement_pump_P-23.md",
        """# Procedure: Bearing Replacement for Pump P-23

## Safety
- Lockout/tagout (LOTO) before any work.
- Verify zero pressure in suction/discharge lines.
- Wear PPE: gloves, goggles, hearing protection.

## Tools
- Bearing puller, torque wrench, dial indicator, grease gun.

## Steps
1. Isolate pump and drain as required.
2. Remove coupling guard and decouple drive.
3. Open bearing housing; inspect for scoring and contamination.
4. Remove old bearing using puller; clean shaft and seating surface.
5. Install new bearing; verify fit and alignment.
6. Relubricate per spec; reassemble housing.
7. Perform alignment check; re-couple and reinstall guards.
8. Run at low load for 15 minutes; monitor vibration and temperature.

## Verification
- Vibration RMS returns to baseline range.
- No abnormal temperature rise after 30 minutes.
""",
    ),
    (
        "sensor_recalibration_pressure_transducer.md",
        """# Procedure: Pressure Sensor Recalibration

## Safety
- Depressurize the line and confirm with gauge.
- Follow LOTO where applicable.

## Steps
1. Inspect wiring and connector seating.
2. Replace cable if insulation damage is present.
3. Calibrate using reference source at 0%, 50%, 100% points.
4. Record calibration results and update maintenance log.

## Verification
- Sensor error within tolerance across test points.
""",
    ),
    (
        "cavitation_triage_pump.md",
        """# Procedure: Cavitation Triage for Pumps

## Symptoms
- Noise like gravel, fluctuating discharge pressure, reduced flow.

## Steps
1. Check suction strainer for blockage.
2. Verify NPSH conditions and suction valve position.
3. Inspect for air ingress on suction side.
4. Reduce speed/load temporarily and observe changes.

## Verification
- Stable pressure and flow after corrective action.
""",
    ),
    (
        "overheating_motor_basic_checks.md",
        """# Procedure: Motor Overheating Basic Checks

## Steps
1. Check ventilation paths and clean vents.
2. Verify load current is within rated limits.
3. Inspect bearings for friction and misalignment.
4. Check ambient temperature and cooling system.

## Verification
- Temperature trend normalizes within 20 minutes.
""",
    ),
    (
        "scheduled_maintenance_template.md",
        """# Procedure: Scheduled Preventive Maintenance Template

## Steps
1. Visual inspection (leaks, corrosion, loose fasteners).
2. Verify instrumentation readings and alarms.
3. Lubrication check and replacement as required.
4. Functional test and post-maintenance verification.

## Verification
- All checks passed; update CMMS.
""",
    ),
]


def main() -> None:
    out = Path("data/raw/procedures")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in RUNBOOKS:
        (out / name).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(RUNBOOKS)} runbooks to {out}")


if __name__ == "__main__":
    main()
