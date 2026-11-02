// EdgeVision-RealTime Frontend Engine
const canvas = document.getElementById("vision-canvas");
const ctx = canvas.getContext("2d");
const videoElement = document.getElementById("webcam-video");

const valLatency = document.getElementById("val-latency");
const valFps = document.getElementById("val-fps");
const valTracks = document.getElementById("val-tracks");
const badgeSecurity = document.getElementById("badge-security");
const subZoneName = document.getElementById("sub-zone-name");
const wsStatusPill = document.getElementById("ws-status");
const eventsList = document.getElementById("events-list");
const eventCounter = document.getElementById("event-counter");

const btnServerStream = document.getElementById("btn-server-stream");
const btnWebcamStream = document.getElementById("btn-webcam-stream");

let ws = null;
let currentSource = "server"; // "server" or "webcam"
let webcamStream = null;
let eventHistory = [];
let pingInterval = null;

// Determine WebSocket URL dynamically based on location
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const wsHost = window.location.host || "localhost:8000";
const wsUrl = `${wsProtocol}//${wsHost}/ws/stream`;

function initWebSocket() {
  updateStatus("connecting", "Connecting WebSocket...");
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    updateStatus("connected", "Streaming Active (30 FPS)");
    startStreamingLoop();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleTelemetry(data);
    } catch (err) {
      console.error("Failed to parse telemetry:", err);
    }
  };

  ws.onclose = () => {
    updateStatus("disconnected", "Disconnected. Reconnecting...");
    stopStreamingLoop();
    setTimeout(initWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    ws.close();
  };
}

function updateStatus(state, label) {
  wsStatusPill.innerHTML = `
    <span class="dot ${state}"></span>
    <span class="status-label">${label}</span>
  `;
}

function startStreamingLoop() {
  stopStreamingLoop();
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (currentSource === "server") {
        // Ping for server-side camera frame
        ws.send(JSON.stringify({ action: "get_frame" }));
      } else if (currentSource === "webcam" && webcamStream) {
        captureAndSendWebcamFrame();
      }
    }
  }, 33); // ~30 FPS
}

function stopStreamingLoop() {
  if (pingInterval) {
    clearInterval(pingInterval);
    pingInterval = null;
  }
}

function captureAndSendWebcamFrame() {
  if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 640;
    tempCanvas.height = 480;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(videoElement, 0, 0, 640, 480);
    tempCanvas.toBlob((blob) => {
      if (blob && ws && ws.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then((buffer) => ws.send(buffer));
      }
    }, "image/jpeg", 0.7);
  }
}

function handleTelemetry(data) {
  if (data.error) {
    return;
  }

  // Update Metrics
  if (data.latency_ms !== undefined) valLatency.textContent = data.latency_ms;
  if (data.fps !== undefined) valFps.textContent = data.fps;
  if (data.detections) valTracks.textContent = data.detections.length;

  // Update Security Status
  if (data.has_violation) {
    badgeSecurity.textContent = "BREACH DETECTED";
    badgeSecurity.className = "security-status-badge breach";
    if (data.violations && data.violations.length > 0) {
      recordSecurityEvents(data.violations);
    }
  } else {
    badgeSecurity.textContent = "PERIMETER SECURE";
    badgeSecurity.className = "security-status-badge secure";
  }

  if (data.zone_name) subZoneName.textContent = data.zone_name;

  // Render Frame & Bounding Boxes
  if (data.frame_bytes_base64) {
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      renderOverlay(data);
    };
    img.src = `data:image/jpeg;base64,${hexToBase64(data.frame_bytes_base64)}`;
  } else if (currentSource === "webcam") {
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    renderOverlay(data);
  }
}

function hexToBase64(hexstring) {
  let match = hexstring.match(/\w{2}/g);
  if (!match) return "";
  return btoa(
    match.map((a) => String.fromCharCode(parseInt(a, 16))).join("")
  );
}

function renderOverlay(data) {
  // 1. Draw Monitored Polygon Zone
  if (data.polygon && data.polygon.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(data.polygon[0][0], data.polygon[0][1]);
    for (let i = 1; i < data.polygon.length; i++) {
      ctx.lineTo(data.polygon[i][0], data.polygon[i][1]);
    }
    ctx.closePath();
    ctx.strokeStyle = data.has_violation ? "#ef4444" : "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = data.has_violation ? "rgba(239, 68, 68, 0.2)" : "rgba(6, 182, 212, 0.15)";
    ctx.fill();

    ctx.fillStyle = data.has_violation ? "#ef4444" : "#06b6d4";
    ctx.font = "bold 12px 'JetBrains Mono'";
    ctx.fillText(data.zone_name || "RESTRICTED AREA", data.polygon[0][0] + 8, data.polygon[0][1] + 20);
  }

  // 2. Draw Detections and Bounding Boxes
  if (data.detections) {
    data.detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const isBreach = data.violations && data.violations.some((v) => v.track_id === det.track_id);

      ctx.strokeStyle = isBreach ? "#ef4444" : "#10b981";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label badge
      const label = `ID:${det.track_id} ${det.class_name} (${Math.round(det.confidence * 100)}%)`;
      ctx.font = "600 11px 'JetBrains Mono'";
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = isBreach ? "#ef4444" : "#10b981";
      ctx.fillRect(x1, y1 - 18, textWidth + 8, 18);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 4, y1 - 4);

      // Foot contact point
      if (det.foot_point) {
        ctx.beginPath();
        ctx.arc(det.foot_point[0], det.foot_point[1], 4, 0, 2 * Math.PI);
        ctx.fillStyle = isBreach ? "#ef4444" : "#3b82f6";
        ctx.fill();
      }
    });
  }
}

function recordSecurityEvents(violations) {
  const timestamp = new Date().toLocaleTimeString();
  violations.forEach((v) => {
    // Avoid spamming identical track_id within 3 seconds
    const existsRecent = eventHistory.some(
      (e) => e.track_id === v.track_id && (Date.now() - e.rawTime) < 3000
    );
    if (!existsRecent) {
      eventHistory.unshift({
        track_id: v.track_id,
        class_name: v.class_name,
        confidence: v.confidence,
        zone_name: v.violated_zone,
        time: timestamp,
        rawTime: Date.now(),
      });
    }
  });

  if (eventHistory.length > 20) eventHistory.pop();
  renderEventLog();
}

function renderEventLog() {
  eventCounter.textContent = `${eventHistory.length} events`;
  if (eventHistory.length === 0) {
    eventsList.innerHTML = `<div class="empty-state">No security breaches detected. Monitored perimeter is clear.</div>`;
    return;
  }

  eventsList.innerHTML = eventHistory
    .map(
      (e) => `
    <div class="event-card">
      <div class="top-line">
        <span>TRACK #${e.track_id}</span>
        <span>${e.time}</span>
      </div>
      <div class="detail-line">
        VIOLATION: ${e.class_name.toUpperCase()} in ${e.zone_name}
      </div>
      <div class="top-line">
        <span>Conf: ${(e.confidence * 100).toFixed(0)}%</span>
        <span>Status: Logged</span>
      </div>
    </div>
  `
    )
    .join("");
}

// Stream Source Toggle Handlers
btnServerStream.addEventListener("click", () => {
  if (currentSource === "server") return;
  currentSource = "server";
  btnServerStream.classList.add("active");
  btnWebcamStream.classList.remove("active");
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
});

btnWebcamStream.addEventListener("click", async () => {
  if (currentSource === "webcam") return;
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    videoElement.srcObject = webcamStream;
    videoElement.play();
    currentSource = "webcam";
    btnWebcamStream.classList.add("active");
    btnServerStream.classList.remove("active");
  } catch (err) {
    alert("Unable to access local webcam: " + err.message);
  }
});

// Boot
window.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
});
