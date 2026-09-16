from __future__ import annotations

import argparse
import base64
import bz2
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_SOURCE_DERIVE_CONTEXT = b"bookai-source-vault-x25519-v1\0"
_SOURCE_INFO = b"bookai-source-vault-aead-v1"
_OUTPUT_INFO = b"bookai-output-vault-aead-v1"


def _b64d(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _b64e(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _derive_source_private() -> X25519PrivateKey:
    secret = os.environ.get("GIGACHAT_AUTH_KEY", "").encode("utf-8")
    if not secret:
        raise SystemExit("GIGACHAT_AUTH_KEY is required to decrypt secure source")
    seed = hashlib.sha256(_SOURCE_DERIVE_CONTEXT + secret).digest()
    return X25519PrivateKey.from_private_bytes(seed)


def _hkdf(shared: bytes, *, salt: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(shared)


def _read_envelope(path: Path) -> dict:
    obj = json.loads(path.read_text("utf-8"))
    if obj.get("version") != 1:
        raise SystemExit(f"unsupported vault envelope version: {obj.get('version')!r}")
    if obj.get("compression") != "bz2":
        raise SystemExit(f"unsupported compression: {obj.get('compression')!r}")
    return obj


def decrypt_source(inp: Path, out: Path) -> None:
    env = _read_envelope(inp)
    private = _derive_source_private()
    peer = X25519PublicKey.from_public_bytes(_b64d(env["ephemeral_public_key"]))
    shared = private.exchange(peer)
    key = _hkdf(shared, salt=_b64d(env["salt"]), info=_SOURCE_INFO)
    compressed = AESGCM(key).decrypt(_b64d(env["nonce"]), _b64d(env["ciphertext"]), _b64d(env["aad"]))
    plain = bz2.decompress(compressed)
    digest = hashlib.sha256(plain).hexdigest()
    if digest != env.get("plaintext_sha256"):
        raise SystemExit(f"source sha256 mismatch: expected={env.get('plaintext_sha256')} actual={digest}")
    out.write_bytes(plain)
    print(f"[bookai-source-vault] decrypted bytes={len(plain)} sha256={digest}", flush=True)


def encrypt_output(inp: Path, out: Path, recipient_public_key_b64: str) -> None:
    plain = inp.read_bytes()
    recipient = X25519PublicKey.from_public_bytes(_b64d(recipient_public_key_b64.strip()))
    ephemeral = X25519PrivateKey.generate()
    ephemeral_public = ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    salt = os.urandom(16)
    nonce = os.urandom(12)
    shared = ephemeral.exchange(recipient)
    key = _hkdf(shared, salt=salt, info=_OUTPUT_INFO)
    aad = b"bookai:evil-for-evil:translation-ru:v1"
    compressed = bz2.compress(plain, compresslevel=9)
    ciphertext = AESGCM(key).encrypt(nonce, compressed, aad)
    env = {
        "version": 1,
        "cipher": "X25519+HKDF-SHA256+AES-256-GCM",
        "compression": "bz2",
        "info": _OUTPUT_INFO.decode("ascii"),
        "aad": _b64e(aad),
        "ephemeral_public_key": _b64e(ephemeral_public),
        "salt": _b64e(salt),
        "nonce": _b64e(nonce),
        "ciphertext": _b64e(ciphertext),
        "plaintext_sha256": hashlib.sha256(plain).hexdigest(),
        "plaintext_bytes": len(plain),
        "compressed_bytes": len(compressed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, separators=(",", ":")), "utf-8")
    print(
        f"[bookai-output-vault] encrypted bytes={len(plain)} envelope_bytes={out.stat().st_size} "
        f"sha256={env['plaintext_sha256']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    dec = sub.add_parser("decrypt-source")
    dec.add_argument("--input", required=True, type=Path)
    dec.add_argument("--output", required=True, type=Path)

    enc = sub.add_parser("encrypt-output")
    enc.add_argument("--input", required=True, type=Path)
    enc.add_argument("--output", required=True, type=Path)
    enc.add_argument("--recipient-public-key", required=True)

    args = parser.parse_args()
    if args.command == "decrypt-source":
        decrypt_source(args.input, args.output)
    else:
        encrypt_output(args.input, args.output, args.recipient_public_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
