from __future__ import annotations

import argparse
import base64
import bz2
import hashlib
import json
import lzma
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

INFO = b"bookai-source-vault-envelope-v1"
SEED_LABEL = b"bookai-source-vault-x25519-v1\0"


def _b64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def derive_source_private(secret: str) -> X25519PrivateKey:
    seed = hashlib.sha256(SEED_LABEL + secret.encode("utf-8")).digest()
    return X25519PrivateKey.from_private_bytes(seed)


def _key(private: X25519PrivateKey, peer_raw: bytes, salt: bytes) -> bytes:
    shared = private.exchange(X25519PublicKey.from_public_bytes(peer_raw))
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=INFO).derive(shared)


def decrypt_source(parts_dir: Path, output: Path, expected_sha: str) -> None:
    secret = os.environ.get("GIGACHAT_AUTH_KEY", "")
    if not secret:
        raise SystemExit("GIGACHAT_AUTH_KEY is missing")
    parts = sorted(parts_dir.glob("source.part*"))
    if not parts:
        raise SystemExit(f"no encrypted source parts in {parts_dir}")
    obj = json.loads("".join(p.read_text("utf-8") for p in parts))
    private = derive_source_private(secret)
    key = _key(private, _b64(obj["eph_pub"]), _b64(obj["salt"]))
    compressed = AESGCM(key).decrypt(_b64(obj["nonce"]), _b64(obj["ciphertext"]), INFO)
    compression = str(obj.get("compression") or "").casefold()
    if compression == "xz":
        data = lzma.decompress(compressed)
    elif compression == "bz2":
        data = bz2.decompress(compressed)
    elif compression in {"", "none"}:
        data = compressed
    else:
        raise SystemExit(f"unsupported source compression: {compression}")
    digest = hashlib.sha256(data).hexdigest()
    envelope_sha = str(obj.get("source_sha256") or "")
    if digest != expected_sha or envelope_sha != expected_sha:
        raise SystemExit(f"source sha mismatch expected={expected_sha} envelope={envelope_sha} actual={digest}")
    output.write_bytes(data)
    print(f"[bookai-vault] decrypted source bytes={len(data)} sha256={digest}")


def seal_output(source: Path, recipient_public_b64: str, output: Path) -> None:
    recipient_raw = _b64(recipient_public_b64)
    ephemeral = X25519PrivateKey.generate()
    epk = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    salt, nonce = os.urandom(16), os.urandom(12)
    key = _key(ephemeral, recipient_raw, salt)
    data = source.read_bytes()
    compressed = lzma.compress(data, preset=6)
    ciphertext = AESGCM(key).encrypt(nonce, compressed, INFO)
    obj = {
        "v": 1,
        "alg": "X25519-HKDF-SHA256-AESGCM",
        "compression": "xz",
        "eph_pub": base64.b64encode(epk).decode("ascii"),
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }
    output.write_text(json.dumps(obj, separators=(",", ":")), "utf-8")
    print(f"[bookai-vault] sealed output bytes={len(data)} sha256={obj['output_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    dec = sub.add_parser("decrypt-source")
    dec.add_argument("--parts-dir", type=Path, required=True)
    dec.add_argument("--output", type=Path, required=True)
    dec.add_argument("--sha256", required=True)
    seal = sub.add_parser("seal-output")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--recipient-public", required=True)
    seal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "decrypt-source":
        decrypt_source(args.parts_dir, args.output, args.sha256)
    else:
        seal_output(args.source, args.recipient_public, args.output)


if __name__ == "__main__":
    main()
