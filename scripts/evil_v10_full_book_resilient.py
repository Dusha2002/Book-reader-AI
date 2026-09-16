from __future__ import annotations

import os
import subprocess
import sys
import time


def _bump_int(env: dict[str, str], key: str, minimum: int) -> None:
    try:
        current = int(env.get(key) or "0")
    except ValueError:
        current = 0
    env[key] = str(max(current, minimum))


def main() -> None:
    max_attempts = max(1, min(5, int(os.getenv("BOOKAI_V10_FULL_ATTEMPTS") or "3")))
    last_code = 1

    for attempt in range(1, max_attempts + 1):
        env = dict(os.environ)
        env["BOOKAI_V10_FULL_ATTEMPT"] = str(attempt)

        # A retry is not a quality downgrade. It gives the already strict release
        # pipeline more capacity to repair only the chapter that failed. Valid chapter
        # checkpoints remain in place and are reused by evil_v10_full_book.py.
        if attempt >= 2:
            _bump_int(env, "BOOKAI_V10_RESIDUAL_DEEP_MAX", 48)
            _bump_int(env, "BOOKAI_V10_RELEASE_DEEP_MAX", 36)
            _bump_int(env, "BOOKAI_V10_EDITORIAL_MAX", 14)
            _bump_int(env, "BOOKAI_GIGACHAT_RETRIES", 4)
        if attempt >= 3:
            _bump_int(env, "BOOKAI_V10_RESIDUAL_DEEP_MAX", 64)
            _bump_int(env, "BOOKAI_V10_RELEASE_DEEP_MAX", 48)
            _bump_int(env, "BOOKAI_V10_EDITORIAL_MAX", 18)
            _bump_int(env, "BOOKAI_GIGACHAT_RETRIES", 5)

        print(
            f"[v10-full-self-heal] attempt={attempt}/{max_attempts} "
            "reuse_valid_checkpoints=true strict_gate=HARD0+integrity0",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, "scripts/evil_v10_full_book.py"],
            env=env,
            check=False,
        )
        last_code = int(result.returncode)
        if last_code == 0:
            print(
                f"[v10-full-self-heal] completed attempt={attempt}/{max_attempts}",
                flush=True,
            )
            return

        if attempt < max_attempts:
            delay = 4 * attempt
            print(
                f"[v10-full-self-heal] attempt={attempt} failed exit={last_code}; "
                f"retrying unfinished chapter only after {delay}s",
                flush=True,
            )
            time.sleep(delay)

    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
