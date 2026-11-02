import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import Any, Dict, List
import cv2
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.connection import ConnectionManager
from src.core.camera import VideoStreamReader
from src.core.detector import VisionDetector
from src.core.zones import ZoneMonitor
from src.db.database import get_db, init_db, SessionLocal
from src.db.models import SecurityZoneEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("edgevision.api")

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
DEFAULT_POLYGON = [
    [150, 100],
    [490, 100],
    [490, 380],
    [150, 380]
]

# Pipeline singletons
camera: VideoStreamReader = None
detector: VisionDetector = None
zone_monitor: ZoneMonitor = ZoneMonitor(DEFAULT_POLYGON, zone_name="Restricted Zone Alpha")
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for hardware discovery and background pipelines."""
    global camera, detector
    logger.info("Initializing EdgeVision-RealTime platform...")
    init_db()

    detector = VisionDetector(model_path=MODEL_PATH)
    camera = VideoStreamReader(src=CAMERA_SOURCE).start()
    logger.info("Video pipeline started on source: %s", CAMERA_SOURCE)

    yield

    logger.info("Shutting down EdgeVision-RealTime platform...")
    if camera is not None:
        camera.stop()


app = FastAPI(
    title="EdgeVision-RealTime API",
    description="Real-Time Computer Vision & Edge AI Serving Pipeline with TensorRT and WebSockets",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for decoupled frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ZoneConfigRequest(BaseModel):
    zone_name: str
    polygon: List[List[int]]


def process_frame_pipeline(frame: np.ndarray, persist_violations: bool = True) -> Dict[str, Any]:
    """Synchronous CPU/GPU heavy pipeline to be executed off the asyncio event loop."""
    start_time = time.perf_counter()
    
    # 1. Inference and temporal tracking
    detections = detector.detect_and_track(frame)
    
    # 2. Geometric security zone evaluation
    violations, has_violation = zone_monitor.check_violations(detections)
    
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 3. Persist security breach events to database
    if persist_violations and has_violation:
        try:
            with SessionLocal() as db:
                for v in violations:
                    event = SecurityZoneEvent(
                        track_id=v["track_id"],
                        class_name=v["class_name"],
                        confidence=v["confidence"],
                        zone_name=zone_monitor.zone_name,
                        bbox=v["bbox"],
                        foot_point=v.get("foot_point"),
                    )
                    db.add(event)
                db.commit()
        except Exception as exc:
            logger.warning("Failed to persist security event to DB: %s", exc)

    return {
        "detections": detections,
        "violations": violations,
        "has_violation": has_violation,
        "zone_name": zone_monitor.zone_name,
        "polygon": zone_monitor.polygon_points,
        "latency_ms": elapsed_ms,
        "fps": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
        "timestamp": time.time(),
    }


@app.get("/health")
def health_check():
    """Returns system diagnostic health and model status."""
    return {
        "status": "healthy",
        "service": "edgevision-realtime",
        "model_loaded": detector.model is not None if detector else False,
        "is_mock": detector.is_mock if detector else False,
        "camera_active": camera.grabbed if camera else False,
        "camera_source": CAMERA_SOURCE,
    }


@app.get("/api/zone")
def get_zone():
    """Returns current security polygon definition."""
    return {
        "zone_name": zone_monitor.zone_name,
        "polygon": zone_monitor.polygon_points,
    }


@app.post("/api/zone")
def update_zone(req: ZoneConfigRequest):
    """Updates security perimeter zone coordinates at runtime."""
    global zone_monitor
    if len(req.polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 vertices.")
    zone_monitor = ZoneMonitor(polygon_points=req.polygon, zone_name=req.zone_name)
    logger.info("Updated security zone '%s' with %d vertices", req.zone_name, len(req.polygon))
    return {"status": "updated", "zone_name": zone_monitor.zone_name, "polygon": zone_monitor.polygon_points}


@app.get("/api/events")
def list_events(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieves recent security perimeter violation events."""
    events = (
        db.query(SecurityZoneEvent)
        .order_by(SecurityZoneEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"events": [e.to_dict() for e in events]}


@app.post("/api/inference")
async def single_frame_inference(file: UploadFile = File(...)):
    """Synchronous inference endpoint dispatched safely to threadpool."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    # Avoid freezing asyncio loop: dispatch heavy CPU/GPU work to threadpool
    results = await asyncio.to_thread(process_frame_pipeline, frame, False)
    return results


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket streaming endpoint for low-latency video and telemetry."""
    await manager.connect(websocket)
    try:
        while True:
            # Client can either send raw JPEG bytes or send ping request for server camera feed
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Run inference safely on worker thread
                    telemetry = await asyncio.to_thread(process_frame_pipeline, frame, True)
                    await websocket.send_text(json.dumps(telemetry))
            else:
                # Client requested server camera feed
                ret, frame = camera.read() if camera else (False, None)
                if ret and frame is not None:
                    telemetry = await asyncio.to_thread(process_frame_pipeline, frame, True)
                    # Encode frame as JPEG to stream back
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    telemetry["frame_bytes_base64"] = buffer.tobytes().hex()
                    await websocket.send_text(json.dumps(telemetry))
                else:
                    await websocket.send_text(json.dumps({"error": "No camera frame available"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket session ended: %s", exc)
        manager.disconnect(websocket)


# Serve static web dashboard
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}
