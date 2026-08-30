# AiAuN

Local-light **MCP** orchestrator: GitHub for code, Kaggle for data + GPU, W&B for runs, Google Drive for small artifacts. The laptop only wraps APIs; training happens on Kaggle.

## Docs

| Doc | What |
|-----|------|
| [docs/VISION.md](docs/VISION.md) | Product intent |
| [docs/SETUP.md](docs/SETUP.md) | Keys, WSL install (`mcp` package), Cursor MCP, add to agents |
| [docs/USAGE.md](docs/USAGE.md) | Prompts, tools, features |
| [docs/design.md](docs/design.md) | Architecture and API flow |
| [docs/implementation.md](docs/implementation.md) | Package layout and how to extend |

## Quick start

1. Fill `.env` from `.env.example` (see Setup).
2. WSL: `bash scripts/install_wsl.sh` (needs Python ≥3.10 + official `mcp`; tries `ssemi` then `base` then `uv` / conda `aiaun`).
3. Copy [`.cursor/mcp.json.example`](.cursor/mcp.json.example) to `.cursor/mcp.json` and enable the `aiaun` server in Cursor.
4. Ask the agent to run an experiment; it should **ask** for data dir, code version, and config if those are missing.

Phase-1 smoke uses `fixtures/synthetic_color_cls` (tiny PPMs), not COCO re-upload.
