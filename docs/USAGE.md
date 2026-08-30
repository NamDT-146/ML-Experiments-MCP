# Usage

Example user prompt:

> Run the experiment of training the setting semi-mask2former coco teacher-student on Kaggle.

The agent should call `resolve_experiment_request`, then **ask** for anything missing (data dir or existing Kaggle slug, code version / branch, which YAML). It must not pick among several coco teacher-student configs silently (`list_repo_configs`).

## Features supported (phase 1)

- Inspect local data dir (count, bytes, samples)
- Check / push Kaggle datasets (no AiAuN size cap; Kaggle API may still reject)
- Push Kaggle **script** kernels (CPU smoke or GPU), internet on
- Poll kernel status and fetch logs (logs often appear only after complete)
- List `semi-mask2former/configs/**/*.yaml`
- Ask-first experiment resolver
- Kernel uploads `/kaggle/working/artifacts` to Drive via Kaggle User Secrets (login-less SA), under `<GOOGLE_DRIVE_FOLDER_ID>/<kernel-slug>/<version-or-run>/`
- Host backup: `kaggle_kernel_output_to_drive` (SSH, no browser)
- Shared Drive folder (SA cannot write My Drive)
- Personal owner mode; org via `AIAUN_OWNER_MODE` (tested, not default)
- Synthetic color-class smoke dataset + inlined smoke script

## Not in phase 1

- Auto-launch full COCO Mask2Former without confirming a YAML
- Re-uploading VOC/COCO/Cityscapes packs
- Google Drive org / GitHub write
- Blocking multi-hour monitor in one tool call

## Next phase (known gap)

Kernel-side Drive upload can still fail on Kaggle with
`ConnectionError('Connection error trying to communicate with service.')`
even with `static_discovery=True`, `num_retries=5`, and per-experiment folders.
Training/inference may complete; artifacts stay on Kaggle Output until the host
calls `kaggle_kernel_output_to_drive` (verified: log, `metrics.json`,
`predictions.json`, `best.pt` into `<root>/<kernel-slug>/run/`).

Phase-2 work: make in-kernel Drive reliable without requiring the PC (retry/backoff
after training, alternate transport, or deferred upload), so users can power off
during long GPU jobs and still land artifacts on Drive.

## Tools

| Tool | When |
|------|------|
| `resolve_experiment_request` | First; ask if `missing` is set |
| `inspect_local_dir` | Local data path |
| `list_repo_configs` | Ambiguous setting name |
| `kaggle_dataset_check` | `owner/slug` or slug (owner from `.env`) |
| `kaggle_dataset_push` | Only if missing; no AiAuN size cap |
| `aiaun_smoke_script` | CPU color prototype smoke |
| `experiment_tracking_links` | Kaggle + W&B + Drive URLs to show the user |
| `kaggle_kernel_push` | After script is ready |
| `kaggle_kernel_status` / `kaggle_kernel_logs` | Agent loop |
| `kaggle_kernel_output_to_drive` | After kernel complete; host-side backup |
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

