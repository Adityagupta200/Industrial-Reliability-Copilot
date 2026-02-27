from __future__ import annotations

import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Float, Enum, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Severity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped["datetime"] = mapped_column(DateTime(timezone=True), nullable=False)
    equipment_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sensor_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    failure_mode: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    severity: Mapped[Severity] = mapped_column(Enum(Severity, name="severity"), nullable=False)
    actions_taken: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    resolution_time_hours: Mapped[float] = mapped_column(Float, nullable=False)


Index("ix_incidents_equipment_timestamp", Incident.equipment_id, Incident.timestamp)
Index("ix_incidents_failure_mode_timestamp", Incident.failure_mode, Incident.timestamp)
