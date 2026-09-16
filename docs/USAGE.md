# Usage

Example user prompt:

> Run the experiment of training the setting semi-mask2former coco teacher-student on Kaggle.

The agent should call `resolve_experiment_request`, then **ask** for anything missing (data dir or existing Kaggle slug, code version / branch, which YAML). It must not pick among several coco teacher-student configs silently (`list_repo_configs`).

## Features supported (phase 2)

- Inspect local data dir (count, bytes, samples)
- Check / push Kaggle datasets (no AiAuN size cap; Kaggle API may still reject)
- **Private runtime-env dataset** (`runtime_env_dataset_push`): packs WANDB/GITHUB/Drive keys from `.env` as a private dataset attached to the kernel — unblocks API auto-runs without UI User Secrets
- **Preflight check** (`preflight_experiment`): validates experiment fields, .env keys, and required file paths in attached datasets before push
- Push Kaggle **script** kernels: `run_mode=auto|draft`, explicit `machine_shape` (T4/A100/etc.), slug resolved after push
- Filtered log fetch: main `*.log` and `artifacts/*_train.log` first; no vendor tree download
- **Error classifier** (`classify_kernel_failure`): maps FATAL log lines to `MISSING_SECRET`, `WRONG_ACCELERATOR`, `MODULE_NOT_FOUND`, `FILE_NOT_FOUND`, etc. with remediation hint
- **W&B run lookup** (`wandb_run_lookup`): fills `tracking.wandb_run` after COMPLETE using urllib (no SDK)
- Per-experiment Drive folder `<root>/<kernel-slug>/<run>/`; `static_discovery=True`, `num_retries=5`
- Host backup: `kaggle_kernel_output_to_drive` (SSH, no browser) — call explicitly when kernel Drive fails
- Dataset content manifest (`aiaun_manifest.json`) written at push time for freshness checks
- Shared Drive folder (SA cannot write My Drive)
- Personal owner mode; org via `AIAUN_OWNER_MODE`
- Synthetic color-class smoke dataset + inlined smoke script

## Not in phase 1

- Auto-launch full COCO Mask2Former without confirming a YAML
- Re-uploading VOC/COCO/Cityscapes packs
- Google Drive org / GitHub write
- Blocking multi-hour monitor in one tool call

## Known gaps (phase 2)

**Kernel-side Drive `ConnectionError`:** Kaggle→Google Drive can still fail with
`ConnectionError` even with `static_discovery=True` and `num_retries=5`.
Artifacts stay on Kaggle Output; the kernel prints `DRIVE_SKIP … repr(e)` with
the full exception. Host backup (`kaggle_kernel_output_to_drive`) is the verified
path. Call it explicitly; the agent does not fall back silently.

**Phase-3 target:** deferred or retry-loop Drive upload from the kernel after
training finishes, so the PC can stay off.

## Tools

| Tool | When |
|------|------|
| `resolve_experiment_request` | First; ask if `missing` is set |
| `preflight_experiment` | Before push; validates fields, keys, dataset paths |
| `runtime_env_dataset_push` | Pack .env secrets into private Kaggle dataset for API auto-runs |
| `inspect_local_dir` | Local data path |
| `list_repo_configs` | Ambiguous setting name |
| `kaggle_dataset_check` | `owner/slug` or slug (owner from `.env`) |
| `kaggle_dataset_push` | Only if missing; no AiAuN size cap |
| `aiaun_smoke_script` | CPU color prototype smoke |
| `aiaun_resnet50_gpu_smoke_script` | GPU ResNet50 smoke (T4) |
| `experiment_tracking_links` | Kaggle + W&B + Drive URLs to show the user |
| `kaggle_kernel_push` | `run_mode`, `machine_shape`, `dataset_slugs` incl. env pack |
| `kaggle_kernel_status` | Poll; returns RUNNING / COMPLETE / ERROR |
| `kaggle_kernel_logs` | Filtered logs (main + artifacts); after COMPLETE preferred |
| `classify_kernel_failure` | Paste log text; returns error class + remediation |
| `wandb_run_lookup` | Fill `tracking.wandb_run` after COMPLETE |
| `kaggle_kernel_output_to_drive` | Host-side backup; call only when user accepts |
| `drive_folder_info` / `drive_upload_file` | Local SA; Shared Drive |

Prompts: `run_kaggle_experiment`, `generate_experiment_notebook`.

## Synthetic smoke

```bash
python scripts/generate_synthetic_color_cls.py
```

Dataset slug: `{KAGGLE_USERNAME}/aiaun-synthetic-color-cls`. Kernel: CPU, inlined classifier, no git clone.

## Self-test: ResNet50 GPU (short)

You can run this yourself in Cursor with MCP **aiaun** enabled.

1. Kaggle notebook secrets: `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON`, `GOOGLE_DRIVE_FOLDER_ID` (Shared Drive experiment folder), optional `WANDB_API_KEY`.
2. Prompt the agent:

> Train ResNet50 on the synthetic color smoke dataset on Kaggle GPU. Short run. After training, run inference with the best checkpoint and upload artifacts to the Drive experiment folder.

After `kaggle_kernel_push` / `kaggle_kernel_status`, the JSON includes a `tracking` object. **Always show these links in chat:**

- `tracking.kaggle_kernel` — running kernel
- `tracking.wandb_project` / `tracking.wandb_run` — W&B
- `tracking.drive_folder` / `tracking.drive_file` — Drive experiment folder / uploaded file

Or call `experiment_tracking_links`.

Local helper: `python -c "from aiaun_mcp.templates import resnet50_smoke_script; print(resnet50_smoke_script(dataset_slug='namdtgk14/aiaun-synthetic-color-cls')[:200])"`

