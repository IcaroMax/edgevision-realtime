from typing import List, Tuple
import cv2
import numpy as np


class ZoneMonitor:
    """Monitors restricted polygon regions of interest (ROI) for perimeter security."""

    def __init__(self, polygon_points: List[List[int]], zone_name: str = "Restricted Area"):
        """Initializes the security zone with 2D polygon vertices.
        
        Args:
            polygon_points: List of [x, y] coordinates defining the polygon perimeter.
            zone_name: Human-readable identifier for the monitored area.
        """
        self.zone_name = zone_name
        self.polygon_points = polygon_points
        self.polygon = np.array(polygon_points, dtype=np.int32)

    def is_inside_zone(self, point: List[int]) -> bool:
        """Determines if a 2D coordinate lies within or on the polygon perimeter.
        
        Uses cv2.pointPolygonTest (Ray casting based geometric test).
        Returns True if point is inside or on the edge (distance >= 0).
        """
        if len(self.polygon) < 3:
            return False
        point_tuple = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.polygon, point_tuple, measureDist=False) >= 0

    def check_violations(self, detections: List[dict]) -> Tuple[List[dict], bool]:
        """Evaluates detected objects against the security polygon.
        
        Args:
            detections: List of detection dictionaries containing 'foot_point' or 'center_point'.
            
        Returns:
            Tuple of (list_of_violating_detections, has_any_violation).
        """
        violations = []
        for det in detections:
            # Check foot_point (ground position) or fallback to center_point
            check_pt = det.get("foot_point") or det.get("center_point")
            if check_pt and self.is_inside_zone(check_pt):
                det_copy = det.copy()
                det_copy["violated_zone"] = self.zone_name
                violations.append(det_copy)

        return violations, len(violations) > 0

    def draw_zone(self, frame: np.ndarray, has_violation: bool = False) -> np.ndarray:
        """Draws the polygon overlay onto the video frame."""
        if len(self.polygon) < 3:
            return frame

        # Red if security breach active, cyan if perimeter clear
        color = (0, 0, 255) if has_violation else (255, 255, 0)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=2)

        # Draw semi-transparent overlay fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Label zone
        centroid_x = int(np.mean(self.polygon[:, 0]))
        centroid_y = int(np.mean(self.polygon[:, 1]))
        cv2.putText(
            frame,
            f"ZONE: {self.zone_name}" + (" [BREACH!]" if has_violation else ""),
            (centroid_x - 60, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255) if has_violation else (255, 255, 255),
            2,
        )
        return frame
