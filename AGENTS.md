# AiAuN MCP — agent rules

When the user asks to run an experiment on Kaggle:

1. Call `resolve_experiment_request`. If `missing` is non-empty, **ask the user**. Do not invent dataset slugs, branches, or config paths.
2. If the setting is ambiguous (e.g. "coco teacher-student"), call `list_repo_configs` and ask which YAML.
3. Prefer existing Kaggle datasets over upload. Do not invent slugs. There is no AiAuN byte cap; Kaggle/Drive APIs may still reject oversized payloads.
4. **Secrets on API auto-run:** Kaggle cannot attach User Secrets via API. Call `list_runtime_env_files` first. If more than default `.env` exists (or the user mentions another project/codebase), **ask which env file** to pack. Then `runtime_env_dataset_push(env_file=…, overrides=…)` — `overrides` is optional JSON for any KEY=VALUE patches. Attach the returned `dataset_slug` to `kaggle_kernel_push dataset_slugs`. Never put secrets in script source. Alternatively use `run_mode=draft` and have the user attach User Secrets via the Kaggle UI.
5. Call `preflight_experiment` before push: checks experiment fields, .env keys, and required dataset paths. Fix all issues before pushing.
6. `kaggle_kernel_push` params: `run_mode=auto` (default) or `draft`; `machine_shape` (default `NvidiaTeslaT4` when `enable_gpu=True`; override with `NvidiaTeslaA100` etc.). Use the `kernel_slug` from the push response (not the requested one) for status and log polling.
7. Kernel uploads `/kaggle/working/artifacts` to Shared Drive `<root>/<kernel-slug>/<run>/` via env-loaded credentials.
8. Poll `kaggle_kernel_status` / `kaggle_kernel_logs`. If status=ERROR and failureMessage=null, call `classify_kernel_failure` on the log text — it returns a classified error and remediation.
9. After COMPLETE, call `wandb_run_lookup` with the same `wandb_project` / `wandb_entity` as `runtime_env_dataset_push` response `effective` (if present) to fill `tracking.wandb_run`. Always show `tracking.kaggle_kernel`, `tracking.wandb_project`, `tracking.drive_folder`.
10. Drive from kernel may still fail with `ConnectionError`. Call `kaggle_kernel_output_to_drive` only when the user explicitly requests host backup. Do not fall back silently.
11. Keys live in repo dotenv files (`.env`, `.env.*`, `envs/*.env`); do not print secret values.
