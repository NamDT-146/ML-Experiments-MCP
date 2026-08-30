# AiAuN MCP — agent rules

When the user asks to run an experiment on Kaggle:

1. Call `resolve_experiment_request`. If `missing` is non-empty, **ask the user**. Do not invent dataset slugs, branches, or config paths.
2. If the setting is ambiguous (e.g. “coco teacher-student”), call `list_repo_configs` and ask which YAML.
3. Prefer existing Kaggle datasets over upload. Do not invent slugs. There is no AiAuN byte cap; Kaggle/Drive APIs may still reject oversized payloads.
4. Never put `GITHUB_TOKEN`, `WANDB_API_KEY`, `KAGGLE_API_TOKEN`, or the Drive SA JSON in kernel source. Use Kaggle User Secrets. Kernel should upload `/kaggle/working/artifacts` to a Shared Drive subfolder `<root>/<kernel-slug>/<run>`.
5. Poll `kaggle_kernel_status` / `kaggle_kernel_logs`. Always show `tracking.kaggle_kernel`, `tracking.wandb_project`, `tracking.drive_folder`. Optional backup: `kaggle_kernel_output_to_drive`.
6. Keys live in repo `.env`; do not print secret values.
