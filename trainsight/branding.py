from __future__ import annotations

import hashlib

WATERMARK = "Developed by: Pratham Patel"
# Tamper-evident checksum for runtime integrity checks.
WATERMARK_SHA256 = hashlib.sha256(WATERMARK.encode("utf-8")).hexdigest()


def assert_watermark_integrity() -> None:
    current = hashlib.sha256(WATERMARK.encode("utf-8")).hexdigest()
    if current != WATERMARK_SHA256:
        raise RuntimeError("Watermark integrity check failed")
