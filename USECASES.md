The AiAuN MCP covered launching a kernel and checking whether it was alive. It did not cover the rest of training and debugging. A full breakdown is in the canvas beside the chat.

What worked was narrow. `kaggle_kernel_push` can attach datasets and turn GPU on. `kaggle_kernel_status` returns COMPLETE, RUNNING, or ERROR once you already know the real slug. `experiment_tracking_links` returns the notebook URL and the W&B project homepage. That is the reliable subset.

**Unsupported.** You cannot attach `WANDB_API_KEY` or `GITHUB_TOKEN`, or check that a secret exists before a run. A push always auto-starts, so every version from v3 to v13 burned a session that died in about a second. There is no draft mode. `enable_gpu: true` does not request a T4×2. The API run that omitted `acc=NvidiaTeslaT4` landed on a P100×1, and status never reports the assigned device. There is no kernel list, so Kaggle’s rewritten slug (`voc-k-net-maskrcnn-iou-thr100-t4x2`) produced a 404 from the URL the MCP returned. There are no live logs, no cancel, no session clock, no resume, no single-file artifact download, and no W&B metrics. `tracking.wandb_run` stayed empty. `drive_folder_info` failed because `GOOGLE_DRIVE_FOLDER_ID` is unset.

**Partial.** Status works only after the slug is correct, and `failureMessage` was null on every failure. Logs worked for that one-line secret error and timed out (`MCP -32001`) on real training output. Dataset check is existence only, so a pack missing `mask2former`, `splits/voc_hetero_seed42.json`, or a real `thirdparty` directory still looked ready. `resolve_experiment_request` checks string fields and said ready while GPU shape and secrets were wrong. Local CLI auth and MCP auth diverged (401 on `~/.kaggle/kaggle.json` while MCP still saw RUNNING).

**Not an MCP bug, but MCP could not show it until the session ended.** The dangling `thirdparty` symlink, the missing Mask2Former sibling, the `datasets` ignore that broke the import, the unbuilt MSDeformAttn op, the missing manifest, the parallel wipe of `summary.json`, and the 300-vs-1449 validation scoring bug all required a full `kaggle kernels output` download outside MCP.

The highest-value additions are a draft push that does not auto-run, an accelerator plus GPU-count field that status echoes back, the slug Kaggle actually created, a log tail that works while RUNNING, a one-file artifact fetch, and a W&B run lookup that fills `tracking.wandb_run`.

## Phase-2 (implemented)

The items above are now addressed:

- **Secrets on API auto-run:** `runtime_env_dataset_push` packs WANDB/GITHUB/Drive keys from `.env` into a private dataset attached to the kernel. No UI interaction needed for `run_mode=auto`.
- **Draft mode:** `kaggle_kernel_push run_mode=draft` writes files without pushing. Returns the UI URL for manual Save & Run with User Secrets.
- **Accelerator:** `machine_shape` param (default `NvidiaTeslaT4`; override with A100, H100, etc.). Kernel metadata sets the field explicitly.
- **Real slug:** `kernel_push` calls `resolve_kernel_slug` after push and returns the slug Kaggle actually created.
- **Preflight:** `preflight_experiment` validates fields, .env keys, and required file paths in attached datasets (via `dataset_list_files` API).
- **Dataset manifest:** `write_aiaun_manifest` stamps content hash + git SHA at upload time.
- **Filtered logs:** `kaggle_kernel_logs` returns main `*.log` + `artifacts/*_train.log` first, skips vendor trees.
- **Error classify:** `classify_kernel_failure` maps FATAL log lines to classes (MISSING_SECRET, WRONG_ACCELERATOR, MODULE_NOT_FOUND, FILE_NOT_FOUND, NETWORK_ERROR, …) with remediation text.
- **W&B run lookup:** `wandb_run_lookup` queries the W&B REST API with urllib and fills `tracking.wandb_run`.

Still open: kernel Drive `ConnectionError` (host backup remains the verified path).