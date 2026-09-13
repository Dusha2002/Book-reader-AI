from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from bergamot import ResponseOptions, Service, ServiceConfig, VectorString

    started = time.perf_counter()
    service = Service(ServiceConfig(numWorkers=max(1, args.workers), logLevel="off"))
    model = service.modelFromConfigPath(args.config)
    options = ResponseOptions(
        qualityScores=False,
        alignment=False,
        HTML=False,
        sentenceMappings=False,
    )
    print(
        f"[bergamot-worker] ready load_seconds={time.perf_counter()-started:.3f}",
        file=sys.stderr,
        flush=True,
    )

    def translate(items):
        texts = [str(item.get("text") or "") for item in items]
        responses = service.translate(model, VectorString(texts), options)
        if len(responses) != len(items):
            raise RuntimeError(f"response mismatch {len(responses)} != {len(items)}")
        return [
            {"id": str(item.get("id") or ""), "text": str(response.target.text or "").strip()}
            for item, response in zip(items, responses)
        ]

    if args.smoke:
        rows = translate([{"id": "smoke", "text": "The shortest way to a man's heart is through his stomach."}])
        print(json.dumps({"items": rows}, ensure_ascii=False), flush=True)
        return

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            items = request.get("items") or []
            if not isinstance(items, list):
                raise ValueError("items must be a list")
            t0 = time.perf_counter()
            rows = translate(items)
            response = {
                "items": rows,
                "elapsed_seconds": round(time.perf_counter() - t0, 4),
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
