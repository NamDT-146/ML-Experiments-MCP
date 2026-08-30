from aiaun_mcp.synthetic import generate_synthetic_color_cls
from aiaun_mcp.config import repo_root

if __name__ == "__main__":
    dest = repo_root() / "fixtures" / "synthetic_color_cls"
    generate_synthetic_color_cls(dest)
    print(dest)
