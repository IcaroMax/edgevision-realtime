import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VideoStreamReader:
    """Threaded video capture stream reader that prevents buffer bloat.
    
    Decouples frame decoding from heavy AI inference by continuously reading
    frames in a background thread and keeping only the most recent frame in memory.
    """

    def __init__(self, src: Union[int, str] = 0, name: str = "VideoStreamReader"):
        self.src = src
        self.name = name
        self.cap: Optional[cv2.VideoCapture] = None
        self.grabbed: bool = False
        self.frame: Optional[np.ndarray] = None
        self.stopped: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "VideoStreamReader":
        """Starts the background thread to read frames from the video stream."""
        if isinstance(self.src, str) and self.src.isdigit():
            self.src = int(self.src)

        self.cap = cv2.VideoCapture(self.src)
        
        # Optimize buffer size for RTSP / network streams where supported
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            # Generate synthetic frames if physical camera/RTSP is unavailable
            self.grabbed = True
            self.frame = self._generate_synthetic_frame("Camera stream unavailable")
        else:
            self.grabbed, self.frame = self.cap.read()

        self.stopped = False
        self.thread = threading.Thread(target=self._update, name=self.name, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Continuously grab and decode the latest frame."""
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if not grabbed:
                    # Loop if video file or retry if stream dropped
                    if isinstance(self.src, str) and not self.src.startswith("rtsp"):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(0.01)
                    continue

                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                with self.lock:
                    self.frame = self._generate_synthetic_frame("Simulated Video Stream")
                    self.grabbed = True
                time.sleep(0.033)  # ~30 FPS

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame thread-safely."""
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        """Stops reading frames and releases video capture resources."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

    @staticmethod
    def _generate_synthetic_frame(message: str) -> np.ndarray:
        """Generates a synthetic test frame when no camera input is available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, "EdgeVision-RealTime [Simulation]", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, message, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, timestamp, (30, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Draw moving reference rectangle
        offset = int((time.time() * 50) % 400)
        cv2.rectangle(frame, (100 + offset, 200), (200 + offset, 350), (255, 0, 0), 2)
        cv2.putText(frame, "simulated_target", (100 + offset, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame
