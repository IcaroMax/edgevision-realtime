import io
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert data["service"] == "edgevision-realtime"


def test_zone_get_and_post():
    # Test GET current zone
    get_res = client.get("/api/zone")
    assert get_res.status_code == 200
    initial_zone = get_res.json()
    assert "polygon" in initial_zone
    assert len(initial_zone["polygon"]) >= 3

    # Test POST new zone coordinates
    new_polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
    post_res = client.post(
        "/api/zone",
        json={"zone_name": "Test Perimeter Beta", "polygon": new_polygon}
    )
    assert post_res.status_code == 200
    updated_data = post_res.json()
    assert updated_data["status"] == "updated"
    assert updated_data["zone_name"] == "Test Perimeter Beta"
    assert updated_data["polygon"] == new_polygon


def test_zone_invalid_polygon():
    # Less than 3 points is an invalid polygon
    invalid_polygon = [[50, 50], [100, 100]]
    res = client.post(
        "/api/zone",
        json={"zone_name": "Invalid Zone", "polygon": invalid_polygon}
    )
    assert res.status_code == 400


def test_events_endpoint():
    response = client.get("/api/events")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_single_frame_inference_endpoint():
    # Create synthetic test frame image in memory
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    response = client.post(
        "/api/inference",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "violations" in data
    assert "has_violation" in data
    assert "latency_ms" in data
