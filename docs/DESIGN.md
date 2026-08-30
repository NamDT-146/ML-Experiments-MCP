# Design

This replaces the credential table in the original DESIGN sketch. Product intent remains [VISION.md](VISION.md).

## Architecture

Client–host–server over **stdio**.

- **Host:** Cursor agent. Asks the user, writes kernel scripts, calls tools.
- **Server:** `python -m aiaun_mcp` on WSL via official PyPI **`mcp`** ≥ 2.1 (`MCPServer` stdio). Requires **Python ≥ 3.10**. Install order: `ssemi` → conda `base` → `uv` / conda env `aiaun` ([scripts/install_wsl.sh](../scripts/install_wsl.sh)).
- **Compute:** Kaggle kernels. Data from Kaggle Datasets; code from GitHub clone or an existing code dataset; metrics to W&B via User Secrets.

Google Drive is optional artifact storage (personal service account). There is no Drive org in this phase.

Personal vs org is **env-only**: `AIAUN_OWNER_MODE=personal|org` selects `KAGGLE_USERNAME` vs `KAGGLE_ORG` and `WANDB_ENTITY` vs `WANDB_TEAM`.

## Credentials

| Key | Where | Role |
|-----|--------|------|
| `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` | Local `.env` | Dataset/kernel API (Bearer `KGAT_*`; `KAGGLE_KEY` fallback) |
| `GITHUB_TOKEN` | Local `.env` and Kaggle User Secrets | Local optional; **kernel clone** must use Secrets |
| `WANDB_API_KEY` + entity/project | Local `.env` and Kaggle User Secrets | Kernel logging |
| `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` + `GOOGLE_DRIVE_FOLDER_ID` | Local `.env` **and** Kaggle User Secrets | Login-less. Kernel uploads `/kaggle/working/artifacts` using `from_service_account_info`. Folder **must** be a Shared Drive (SA has no My Drive quota). Never put the JSON in kernel source. |

The server redacts known secret values from error strings. Kernel `script_content` that looks like it contains tokens is refused.

## MCP surface

**Tools:** … `experiment_tracking_links` (Kaggle + W&B + Drive URLs). Kernel/status/Drive tools include a `tracking` object; the agent must show those links to the user.

**Prompts:** `run_kaggle_experiment`, `generate_experiment_notebook`.

`inspect_local_dir` is a tool (not a custom `local_dir://` URI). Monitor is split so one RPC does not block for hours.

## Use case flow

1. User asks to run a setting on Kaggle.
2. Agent resolves fields; **asks** if data dir, code version, or config is missing. Lists YAML candidates when ambiguous.
3. Inspect local dir; `kaggle_dataset_check`; upload only if absent (Kaggle/Drive APIs enforce their own limits).
4. Generate script: Kaggle Secrets for git/W&B; or smoke script with no clone.
5. `kaggle_kernel_push` (internet; GPU only when requested).
6. Poll status/logs. Kernel should push artifacts to Drive via User Secrets. MCP backup: `kaggle_kernel_output_to_drive` (SSH, no browser).
7. **Known gap:** Kaggle→Drive from the kernel may still hit `ConnectionError` to Google; host backup is the verified path until phase 2 hardens in-kernel upload.
