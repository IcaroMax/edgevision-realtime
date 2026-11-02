from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from src.db.database import Base


class SecurityZoneEvent(Base):
    """Stores security perimeter violation events detected by the vision engine."""

    __tablename__ = "security_zone_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    zone_name = Column(String(100), nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    foot_point = Column(JSON, nullable=True)  # [x, y]
    snapshot_path = Column(String(255), nullable=True)

    def to_dict(self):
        """Serializes event object to JSON-compatible dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "zone_name": self.zone_name,
            "bbox": self.bbox,
            "foot_point": self.foot_point,
            "snapshot_path": self.snapshot_path,
        }
