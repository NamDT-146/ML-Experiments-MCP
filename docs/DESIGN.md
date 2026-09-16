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
| `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` | Local default `.env` (host only) | Dataset/kernel API (Bearer `KGAT_*`; `KAGGLE_KEY` fallback) |
| `GITHUB_TOKEN` | Local dotenv + kernel pack / User Secrets | Private clone needs fine-grained **Contents: Read** (or classic `repo`) |
| `WANDB_API_KEY` + entity/project | Local dotenv + kernel pack / User Secrets | Kernel logging; project may differ per env file |
| `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` + `GOOGLE_DRIVE_FOLDER_ID` | Local dotenv + kernel pack / User Secrets | Login-less. Kernel uploads `/kaggle/working/artifacts` using `from_service_account_info`. Folder **must** be a Shared Drive (SA has no My Drive quota). Never put the JSON in kernel source. |

**Multi-env kernel pack:** Host MCP always boots from default `.env`. For Kaggle API auto-runs, `list_runtime_env_files` + `runtime_env_dataset_push(env_file, overrides)` packs a chosen dotenv (`.env`, `.env.*`, or `envs/*.env`) plus optional JSON key overrides into private dataset `{owner}/aiaun-run-env`. Host-only keys (`KAGGLE_*`, `AIAUN_*`) are never packed. Precedence: overrides > selected file > host defaults.

The server redacts known secret values from error strings. Kernel `script_content` that looks like it contains tokens is refused.

## MCP surface

**Tools:** … `list_runtime_env_files`, `runtime_env_dataset_push`, `experiment_tracking_links` (optional `wandb_project` / `wandb_entity`), `wandb_run_lookup`. Kernel/status/Drive tools include a `tracking` object; the agent must show those links to the user.

**Prompts:** `run_kaggle_experiment`, `generate_experiment_notebook`.

`inspect_local_dir` is a tool (not a custom `local_dir://` URI). Monitor is split so one RPC does not block for hours.

## Use case flow

1. User asks to run a setting on Kaggle.
2. Agent resolves fields; **asks** if data dir, code version, or config is missing. Lists YAML candidates when ambiguous.
3. Inspect local dir; `kaggle_dataset_check`; upload only if absent (Kaggle/Drive APIs enforce their own limits).
4. `list_runtime_env_files`; **ask which env file** when alternatives exist; `runtime_env_dataset_push(env_file, overrides)`; attach slug to the kernel.
5. Generate script: env pack / Kaggle Secrets for git/W&B; or smoke script with no clone.
6. `kaggle_kernel_push` (internet; GPU only when requested; `machine_shape` for T4/A100).
7. Poll status/logs. Kernel should push artifacts to Drive via env-loaded credentials. MCP backup: `kaggle_kernel_output_to_drive` (SSH, no browser).
8. If status=ERROR and failureMessage=null: call `classify_kernel_failure` on log text.
9. After COMPLETE: `wandb_run_lookup` with effective WANDB_* fills `tracking.wandb_run`.
10. **Known gap:** Kaggle→Drive from the kernel may still hit `ConnectionError`; host backup (`kaggle_kernel_output_to_drive`) is the verified path.
