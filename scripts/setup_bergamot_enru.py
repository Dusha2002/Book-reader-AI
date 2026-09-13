from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


FILES = {
    "model.enru.intgemm.alphas.bin.gz": 20_000_000,
    "vocab.enru.spm.gz": 100_000,
    "lex.50.50.enru.s2t.bin.gz": 100_000,
}
RAW_URL = "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/models/base/enru"


def _copy_or_download(name: str, destination: Path, source_dir: Path | None) -> None:
    if destination.exists() and destination.stat().st_size > 128:
        return
    if source_dir is not None:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing Bergamot source file: {source}")
        # A Git-LFS pointer is ~130 bytes. Refuse it explicitly rather than failing
        # later while gunzipping with a confusing message.
        if source.stat().st_size <= 256:
            raise RuntimeError(
                f"{source} is still a Git LFS pointer ({source.stat().st_size} bytes); run git lfs pull first"
            )
        shutil.copy2(source, destination)
        return

    # Fallback is useful outside CI only when raw.githubusercontent happens to serve
    # the LFS object. CI uses a sparse Git-LFS checkout deliberately.
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        f"{RAW_URL}/{name}", headers={"User-Agent": "BookReaderAI-Bergamot-Eval/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    if tmp.stat().st_size <= 256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Raw GitHub returned a Git LFS pointer; provide --source-dir after git lfs pull")
    tmp.replace(destination)


def _gunzip(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size > 128:
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
    print(config)
    print(
        f"[bergamot-setup] model={model.stat().st_size} vocab={vocab.stat().st_size} "
        f"shortlist={shortlist.stat().st_size} config={config}",
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
