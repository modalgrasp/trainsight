from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


@dataclass
class ReleaseVerificationResult:
    ok: bool
    message: str


def _canonical_manifest_payload(manifest: dict[str, Any]) -> bytes:
    payload = {
        "version": manifest.get("version", ""),
        "generated_at": manifest.get("generated_at", ""),
        "files": manifest.get("files", {}),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_signature(manifest: dict[str, Any]) -> tuple[bool, str]:
    sig_b64 = manifest.get("signature", "")
    if not sig_b64:
        return False, "Missing manifest signature"

    try:
        pub_pem = files("trainsight.security").joinpath("public_key.pem").read_bytes()
        public_key = serialization.load_pem_public_key(pub_pem)
        if not isinstance(public_key, Ed25519PublicKey):
            return False, "Unsupported public key type"

        signature = base64.b64decode(sig_b64)
        public_key.verify(signature, _canonical_manifest_payload(manifest))
        return True, "Signature valid"
    except Exception as exc:
        return False, f"Signature verification failed: {exc}"


def _verify_hashes(manifest: dict[str, Any]) -> tuple[bool, str]:
    files_map = manifest.get("files", {})
    if not isinstance(files_map, dict) or not files_map:
        return False, "Manifest file list empty"

    pkg_root = files("trainsight")
    for rel_path, expected_hash in files_map.items():
        try:
            payload = pkg_root.joinpath(rel_path).read_bytes()
        except Exception:
            return False, f"Missing file in install: trainsight/{rel_path}"

        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_hash:
            return False, f"Hash mismatch: trainsight/{rel_path}"

    return True, "File hashes valid"


def verify_release(strict: bool = False, full_check: bool | None = None) -> ReleaseVerificationResult:
    if full_check is None:
        full_check = bool(strict)

    try:
        manifest = json.loads(files("trainsight.security").joinpath("release_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Cannot load release manifest: {exc}"
        return ReleaseVerificationResult(False, msg if strict else f"WARNING: {msg}")

    sig_ok, sig_msg = _verify_signature(manifest)
    if not sig_ok:
        return ReleaseVerificationResult(False, sig_msg if strict else f"WARNING: {sig_msg}")

    if full_check:
        hash_ok, hash_msg = _verify_hashes(manifest)
        if not hash_ok:
            return ReleaseVerificationResult(False, hash_msg if strict else f"WARNING: {hash_msg}")

    try:
        installed_version = version("trainsight")
    except PackageNotFoundError:
        installed_version = manifest.get("version", "")

    if str(installed_version) != str(manifest.get("version", "")):
        msg = f"Version mismatch: installed={installed_version}, signed={manifest.get('version')}"
        return ReleaseVerificationResult(False, msg if strict else f"WARNING: {msg}")

    return ReleaseVerificationResult(True, "Official release verification passed")
