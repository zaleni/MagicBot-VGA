from __future__ import annotations

import logging
import time
from typing import Optional

import websockets.sync.client

try:
    from .msgpack_numpy import Packer, unpackb
except ImportError:
    from msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Simple websocket client for the Real_Lift2 inference server."""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> None:
        pass
