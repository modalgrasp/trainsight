from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _iter_project_files(pkg_root: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.py", "*.yaml", "*.yml"):
        files.extend(pkg_root.rglob(ext))
    return sorted([p for p in files if "__pycache__" not in p.parts])


def _load_or_create_private_key(path: Path) -> Ed25519PrivateKey:
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)

    path.parent.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return priv


def sign_release(project_root: Path, private_key_path: Path) -> None:
    pkg_root = project_root / "trainsight"
    security_root = pkg_root / "security"
    security_root.mkdir(parents=True, exist_ok=True)

    priv = _load_or_create_private_key(private_key_path)
    pub = priv.public_key()

    files_map: dict[str, str] = {}
    for file_path in _iter_project_files(pkg_root):
        rel = file_path.relative_to(pkg_root).as_posix()
        files_map[rel] = hashlib.sha256(file_path.read_bytes()).hexdigest()

    manifest = {
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files_map,
    }

    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = priv.sign(payload)
    manifest["signature"] = base64.b64encode(signature).decode("ascii")

    (security_root / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (security_root / "public_key.pem").write_bytes(
        pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="trainsight-sign", description="Sign TrainSight release manifest")
    p.add_argument("--project-root", default=".", help="Project root containing trainsight package")
    p.add_argument("--private-key", default=str(Path.home() / ".trainsight" / "release_private.pem"), help="Path to Ed25519 private key")
    args = p.parse_args()

    sign_release(Path(args.project_root).resolve(), Path(args.private_key).resolve())
    print("Signed release manifest updated")


if __name__ == "__main__":
    main()
