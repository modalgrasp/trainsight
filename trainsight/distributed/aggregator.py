from __future__ import annotations

import json
import socket
from typing import Callable


def run_udp_aggregator(host: str, port: int, handler: Callable[[dict], None]) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, int(port)))
    while True:
        data, _addr = sock.recvfrom(65535)
        try:
            handler(json.loads(data.decode("utf-8")))
        except Exception:
            continue
