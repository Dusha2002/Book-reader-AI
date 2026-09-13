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
BASE_URL = (
    "https://media.githubusercontent.com/media/mozilla/"
    "firefox-translations-models/main/models/base/enru"
)


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 128:
        return
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "BookReaderAI-Bergamot-Eval/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)


def _gunzip(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size > 128:
        return
    tmp = target.with_suffix(target.suffix + ".part")
    with gzip.open(source, "rb") as inp, tmp.open("wb") as out:
        shutil.copyfileobj(inp, out)
    tmp.replace(target)


def prepare(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        compressed = output / name
        _download(f"{BASE_URL}/{name}", compressed)
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
    args = parser.parse_args()
    prepare(Path(args.output))


if __name__ == "__main__":
    main()
