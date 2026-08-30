# Setup

Work in **WSL** at `/mnt/h/Dev/Lab/AiAuN`. The official **`mcp`** package requires **Python ≥ 3.10** (`ssemi` may be 3.8 — then skip it).

**Cursor MCP is a Windows process.** It cannot execute `/home/.../python`. Drive-letter paths like `C:\Windows\System32\wsl.exe` also fail here with `The system cannot find the path specified` (the host treats them as missing files). Do not launch a `.sh` from `H:` (CRLF → `pipefail: invalid option name`).

[`.cursor/mcp.json`](../.cursor/mcp.json) therefore uses a **repo-relative** Windows venv exe (no extra `pip`) which starts WSL Python:

`.venv/Scripts/python.exe scripts/mcp_wsl_trampoline.py`

Create the venv once: `H:\Dev\Conda\Conda\python.exe -m venv .venv --without-pip`. After changing MCP config: disable **aiaun** → enable.

Relative paths in `.env` (e.g. `.secret/sa.json`) work on both OS.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd /mnt/h/Dev/Lab/AiAuN
bash scripts/install_wsl.sh
# order: conda ssemi → conda base → system 3.10+ → uv .venv → new conda env `aiaun`
python -m aiaun_mcp.synthetic
bash scripts/run_tests.sh   # offline, then live if AIAUN_LIVE_TEST=1 inside the script
```

Live Kaggle/Drive (tiny data only):

```bash
AIAUN_LIVE_TEST=1 "$(tr -d '\r' < .aiaun-python)" -m pytest tests/acceptance -m live -q
```

## Keys (`.env`)

Copy `.env.example` to `.env`. Never commit `.env` or `.secret/`.

| Variable | How to get |
|----------|------------|
| `KAGGLE_USERNAME` | Kaggle profile slug (`namdtgk14`) |
| `KAGGLE_API_TOKEN` | [Kaggle Settings → API](https://www.kaggle.com/settings) Bearer `KGAT_*` |
| `WANDB_API_KEY` | [wandb.ai/authorize](https://wandb.ai/authorize) |
| `WANDB_ENTITY` | First path segment of the project URL |
| `WANDB_PROJECT` | Second path segment (e.g. `semi-mask2former`) |
| `GITHUB_TOKEN` | Fine-grained **Contents: Read-only**, or classic **`repo` only** |
| `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` | Path to SA JSON (this repo uses `.secret/*.json`) |
| `GOOGLE_DRIVE_FOLDER_ID` | ID in `drive.google.com/drive/folders/<ID>` after sharing the folder with the SA **email** as Editor |

Enable **Google Drive API**. Login-less auth is a **service account** (no browser on SSH or Kaggle).

**Kaggle → Drive (preferred):** Add User Secrets (Add-ons → Secrets), never paste into the notebook:

- `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` — entire SA JSON as one string
- `GOOGLE_DRIVE_FOLDER_ID` — folder id

The kernel loads the JSON in memory (`from_service_account_info`) and uploads `/kaggle/working/artifacts`. Do not write the JSON into `/kaggle/working` (it would leak in Output).

**Shared Drive required for writes:** Google gives service accounts **no My Drive quota**. Create a Shared Drive, add the SA email as **Content Manager**, put artifacts there. Sharing a normal My Drive folder is enough to *read* (`drive_folder_info`) but *upload* will fail with `storageQuotaExceeded`.

**SSH MCP backup (still no UI):** after the kernel finishes, `kaggle_kernel_output_to_drive` pulls Kaggle Output with the Kaggle API key and uploads small files with the same SA. Copy `.env` + `.secret/*.json` onto the server; do not run a browser login.

Optional OAuth (`python -m aiaun_mcp.drive_login --mode tunnel`) is only a fallback if you cannot use a Shared Drive.

Org later: `AIAUN_OWNER_MODE=org` plus `KAGGLE_ORG`, `GITHUB_ORG`, `WANDB_TEAM`.

### Kaggle User Secrets (kernel runtime)

In a Kaggle notebook: Add-ons → Secrets → add `GITHUB_TOKEN` and `WANDB_API_KEY`. Kernels must read these at runtime. Do not paste them into `script_content`.

## Add to Cursor agents

1. This repo ships [`.cursor/mcp.json`](../.cursor/mcp.json): `.venv/Scripts/python.exe` → [`scripts/mcp_wsl_trampoline.py`](../scripts/mcp_wsl_trampoline.py) → WSL `python -m aiaun_mcp`.
2. Cursor Settings → **MCP** → confirm server **aiaun** is enabled. Restart Cursor if tools do not appear.
3. Agent instructions: [AGENTS.md](../AGENTS.md) and [`.cursor/rules/aiaun-mcp.mdc`](../.cursor/rules/aiaun-mcp.mdc) (ask-first, no secret embedding).

If stdio fails, run `bash scripts/install_wsl.sh` then `bash scripts/mcp_launch.sh` in WSL. PyPI timeouts: retry with a longer `PIP_DEFAULT_TIMEOUT` (the install script already uses 120s).
