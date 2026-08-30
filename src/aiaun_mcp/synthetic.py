from __future__ import annotations

import csv
from pathlib import Path

COLORS = {
    "red": (220, 32, 32),
    "green": (32, 180, 48),
    "blue": (32, 64, 220),
}


def _ppm(path: Path, rgb: tuple[int, int, int], w: int = 16, h: int = 16) -> None:
    r, g, b = rgb
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    body = bytes([r, g, b]) * (w * h)
    path.write_bytes(header + body)


def generate_synthetic_color_cls(dest: Path, *, per_class: int = 4) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    for name, rgb in COLORS.items():
        for i in range(per_class):
            fname = f"{name}_{i:02d}.ppm"
            _ppm(dest / fname, rgb)
            rows.append((fname, name))
    with (dest / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows(rows)
    readme = dest / "README.txt"
    readme.write_text(
        "AiAuN synthetic color classification smoke set (tiny PPM files).\n",
        encoding="utf-8",
    )
    return dest


def main() -> None:
    from aiaun_mcp.config import repo_root

    dest = repo_root() / "fixtures" / "synthetic_color_cls"
    generate_synthetic_color_cls(dest)
    print(dest)


if __name__ == "__main__":
    main()
