import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger("edgevision.connection")


class ConnectionManager:
    """Manages active WebSocket connections for real-time video and telemetry broadcast."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts and registers a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected. Total active connections: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        """Removes a disconnected client from the active registry."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected. Remaining connections: %d", len(self.active_connections))

    async def broadcast_text(self, message: str):
        """Broadcasts a JSON string message to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as exc:
                logger.debug("Failed to send message to client, disconnecting: %s", exc)
                self.disconnect(connection)

    async def broadcast_bytes(self, data: bytes):
        """Broadcasts binary data to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_bytes(data)
            except Exception as exc:
                logger.debug("Failed to send bytes to client, disconnecting: %s", exc)
                self.disconnect(connection)
