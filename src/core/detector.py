import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("edgevision.detector")


class VisionDetector:
    """Encapsulates YOLO object detection and ByteTrack tracking.
    
    Supports NVIDIA TensorRT compiled engines (.engine) with seamless fallback
    to PyTorch (.pt) weights.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.is_mock = False
        self._load_model()

    def _load_model(self):
        """Loads TensorRT engine if present, or falls back to PyTorch model."""
        target_path = self.model_path

        # If .engine requested but not found, attempt .pt fallback
        if target_path.endswith(".engine") and not os.path.exists(target_path):
            fallback_pt = target_path.replace(".engine", ".pt")
            logger.warning(
                "TensorRT engine '%s' not found. Falling back to PyTorch weights '%s'",
                target_path, fallback_pt
            )
            target_path = fallback_pt

        try:
            from ultralytics import YOLO
            self.model = YOLO(target_path)
            logger.info("Model loaded successfully: %s", target_path)
        except Exception as exc:
            logger.warning(
                "Failed to initialize Ultralytics YOLO (%s). Operating in mock mode.",
                exc
            )
            self.is_mock = True

    def detect_and_track(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Executes inference and temporal tracking across video frames.
        
        Args:
            frame: Raw BGR frame image from camera.
            tracker: Tracker config YAML (e.g. 'bytetrack.yaml').
            conf_threshold: Minimum detection confidence threshold.

        Returns:
            List of detected objects with bounding boxes, confidence, class and track ID.
        """
        if self.is_mock or self.model is None:
            return self._mock_detect_and_track(frame)

        try:
            # persist=True instructs tracker to maintain IDs between consecutive frames
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                conf=conf_threshold,
                verbose=False
            )

            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(len(boxes))
                )

                for box, track_id, cls_idx, conf in zip(xyxy_coords, track_ids, classes, confs):
                    x1, y1, x2, y2 = box
                    class_name = (
                        self.model.names[cls_idx]
                        if hasattr(self.model, "names") and cls_idx in self.model.names
                        else f"class_{cls_idx}"
                    )
                    detections.append({
                        "track_id": int(track_id),
                        "class_name": str(class_name),
                        "confidence": float(round(conf, 2)),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "foot_point": [int((x1 + x2) / 2), int(y2)],  # Ground contact point
                        "center_point": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    })

            return detections
        except Exception as exc:
            logger.error("Error during detect_and_track inference: %s", exc)
            return self._mock_detect_and_track(frame)

    @staticmethod
    def _mock_detect_and_track(frame: np.ndarray) -> List[Dict[str, Any]]:
        """Deterministic mock detection generator for headless testing environments."""
        h, w = frame.shape[:2]
        return [
            {
                "track_id": 1,
                "class_name": "person",
                "confidence": 0.92,
                "bbox": [int(w * 0.2), int(h * 0.2), int(w * 0.4), int(h * 0.8)],
                "foot_point": [int(w * 0.3), int(h * 0.8)],
                "center_point": [int(w * 0.3), int(h * 0.5)],
            }
        ]
