from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "DATASET_MANIFEST.json"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = []
    for base in (ROOT / "data" / "raw", ROOT / "data" / "processed"):
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    MANIFEST.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} file hashes to {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
