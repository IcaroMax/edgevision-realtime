import numpy as np
import pytest
from src.core.zones import ZoneMonitor


def test_zone_point_inside():
    # Square polygon from (100, 100) to (300, 300)
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = ZoneMonitor(polygon, zone_name="Test Zone")

    # Inside point
    assert zone.is_inside_zone([200, 200]) is True

    # Outside points
    assert zone.is_inside_zone([50, 50]) is False
    assert zone.is_inside_zone([400, 200]) is False
    assert zone.is_inside_zone([200, 400]) is False


def test_zone_point_boundary():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = ZoneMonitor(polygon)

    # Boundary edge
    assert zone.is_inside_zone([50, 0]) is True
    # Vertex
    assert zone.is_inside_zone([0, 0]) is True


def test_zone_check_violations():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = ZoneMonitor(polygon, zone_name="Hazard Zone")

    detections = [
        {
            "track_id": 1,
            "class_name": "person",
            "confidence": 0.95,
            "foot_point": [200, 200],  # Inside
            "bbox": [180, 150, 220, 200]
        },
        {
            "track_id": 2,
            "class_name": "forklift",
            "confidence": 0.88,
            "foot_point": [500, 500],  # Outside
            "bbox": [450, 450, 550, 500]
        }
    ]

    violations, has_violation = zone.check_violations(detections)
    assert has_violation is True
    assert len(violations) == 1
    assert violations[0]["track_id"] == 1
    assert violations[0]["violated_zone"] == "Hazard Zone"


def test_zone_drawing():
    polygon = [[10, 10], [100, 10], [100, 100], [10, 100]]
    zone = ZoneMonitor(polygon)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    drawn_frame = zone.draw_zone(frame, has_violation=False)
    assert drawn_frame.shape == (200, 200, 3)
    assert np.any(drawn_frame > 0)
