from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path


FILES = {
    "model.enru.intgemm.alphas.bin.gz": 20_000_000,
    "vocab.enru.spm.gz": 100_000,
    "lex.50.50.enru.s2t.bin.gz": 100_000,
}
# Public mirror of the archived Firefox/Bergamot model files. Pin a concrete
# revision instead of `main` so the benchmark is reproducible.
HF_REVISION = "ffb33a7be7079f5c1a1d8db07f9b5c432f0bcc87"
HF_BASE_URL = (
    "https://huggingface.co/TiberiuCristianLeon/Bergamot/resolve/"
    + HF_REVISION
    + "/base/enru"
)


def _copy_or_download(name: str, destination: Path, source_dir: Path | None) -> None:
    if destination.exists() and destination.stat().st_size > 256:
        return
    if source_dir is not None:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing Bergamot source file: {source}")
        if source.stat().st_size <= 256:
            raise RuntimeError(f"{source} is not a real model object")
        shutil.copy2(source, destination)
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    url = f"{HF_BASE_URL}/{name}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "BookReaderAI-Bergamot-Eval/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    if tmp.stat().st_size <= 256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Bergamot mirror returned an invalid object for {name}")
    tmp.replace(destination)
    print(
        f"[bergamot-download] file={name} bytes={destination.stat().st_size} "
        f"sha256={hashlib.sha256(destination.read_bytes()).hexdigest()[:16]}",
        flush=True,
    )


def _gunzip(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size > 256:
        return
    tmp = target.with_suffix(target.suffix + ".part")
    with gzip.open(source, "rb") as inp, tmp.open("wb") as out:
        shutil.copyfileobj(inp, out)
    tmp.replace(target)


def prepare(output: Path, source_dir: Path | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        compressed = output / name
        _copy_or_download(name, compressed, source_dir)
        _gunzip(compressed, output / name.removesuffix(".gz"))

    model = output / "model.enru.intgemm.alphas.bin"
    vocab = output / "vocab.enru.spm"
    shortlist = output / "lex.50.50.enru.s2t.bin"
    if model.stat().st_size < 20_000_000:
        raise RuntimeError(f"Downloaded Bergamot model looks invalid: {model.stat().st_size} bytes")
    if vocab.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded Bergamot vocabulary looks invalid: {vocab.stat().st_size} bytes")
    if shortlist.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded Bergamot shortlist looks invalid: {shortlist.stat().st_size} bytes")

    config = output / "config.bergamot.yml"
    config.write_text(
        f"""models:
  - {model}
vocabs:
  - {vocab}
  - {vocab}
shortlist:
  - {shortlist}
  - false
beam-size: 1
normalize: 1.0
word-penalty: 0
max-length-break: 128
mini-batch-words: 1024
workspace: 128
max-length-factor: 2.0
skip-cost: true
cpu-threads: 0
quiet: true
quiet-translation: true
gemm-precision: int8shiftAlphaAll
alignment: soft
""",
        encoding="utf-8",
    )
    print(
        f"[bergamot-setup] revision={HF_REVISION} model={model.stat().st_size} "
        f"vocab={vocab.stat().st_size} shortlist={shortlist.stat().st_size} config={config}",
        flush=True,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/bookai-bergamot-enru")
    parser.add_argument("--source-dir", default="")
    args = parser.parse_args()
    prepare(Path(args.output), Path(args.source_dir) if args.source_dir else None)


if __name__ == "__main__":
    main()
