import numpy as np
import pytest
from src.core.detector import VisionDetector


def test_detector_fallback_init():
    # Provide nonexistent engine file; should trigger fallback without crashing
    detector = VisionDetector(model_path="nonexistent_model.engine")
    assert detector is not None


def test_detect_and_track_output_format():
    detector = VisionDetector(model_path="yolov8n.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = detector.detect_and_track(dummy_frame)
    assert isinstance(results, list)

    for item in results:
        assert "track_id" in item
        assert "class_name" in item
        assert "confidence" in item
        assert "bbox" in item
        assert "foot_point" in item
        assert len(item["bbox"]) == 4
        assert len(item["foot_point"]) == 2
