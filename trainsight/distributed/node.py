from __future__ import annotations

import json
import socket


def publish_event(host: str, port: int, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(data, (host, int(port)))
    finally:
        sock.close()
