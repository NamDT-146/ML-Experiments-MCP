from __future__ import annotations

from pathlib import Path

from aiaun_mcp.config import get_settings
from aiaun_mcp.synthetic import generate_synthetic_color_cls


def main() -> None:
    root = get_settings().root
    dest = root / "fixtures" / "synthetic_color_cls"
    generate_synthetic_color_cls(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
