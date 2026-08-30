# Implementation

## Layout

```
src/aiaun_mcp/
  config.py       # .env, owner mode, redact, Settings
  experiment.py   # missing-field resolver
  fsutil.py       # inspect dir, metadata JSON
  kaggle_ops.py   # datasets, kernels, kernel_output_to_drive
  drive_ops.py    # SA (+ optional OAuth token); Shared Drive writes
  templates.py    # smoke script (User Secrets Drive push) + agent prompts
  drive_login.py  # optional SSH tunnel OAuth; not required if Shared Drive
  server.py       # MCPServer stdio tools
scripts/generate_synthetic_color_cls.py
tests/unit|migration|acceptance
fixtures/synthetic_color_cls/   # generated
```

Install: `bash scripts/install_wsl.sh` (writes `.aiaun-python`). Entry: `python -m aiaun_mcp` using that interpreter.

The MCP stdio server uses official **`mcp.server.mcpserver.MCPServer`** (`mcp` ≥ 2.1). Do not drop this dependency.

Kaggle Bearer: set `KAGGLE_API_TOKEN` and patch `KaggleApi.config_values` like `semi-mask2former/scripts/kaggle_upload_flat.py`. Dataset existence uses kagglesdk `GetDatasetStatus` (legacy `dataset_status` 403s with Bearer tokens). Windows `H:\...` Drive JSON paths are rewritten to `/mnt/h/...` under WSL.

## Tests

- **Unit:** redact, missing keys, owner personal, resolver, size cap, metadata, smoke has no secrets, config glob.
- **Migration:** `AIAUN_OWNER_MODE=org` changes owner and W&B entity; `KAGGLE_KEY` fallback; Drive folder id swap.
- **Acceptance:** A1 ask-first; A4 kernel metadata; A6 missing token; live A2–A5/A8 behind `AIAUN_LIVE_TEST=1`.

## Extend

Add a function in `kaggle_ops.py` or `drive_ops.py`, wrap it in `server.py` with `@mcp.tool()`, return JSON strings, never interpolate env secrets into kernel templates.

Frozen `Settings` is constructed only in `get_settings`.
