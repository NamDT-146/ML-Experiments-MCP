"""
Lightweight W&B run lookup using urllib (no wandb SDK required locally).
Queries the public W&B REST API with the local WANDB_API_KEY.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

_WANDB_API_BASE = "https://api.wandb.ai"


def _wandb_get(path: str, api_key: str) -> Any:
    url = f"{_WANDB_API_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"W&B API {exc.code}: {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"W&B request failed: {exc}") from exc


def wandb_run_lookup(
    settings,
    *,
    run_name: str = "",
    run_id: str = "",
    limit: int = 5,
) -> dict:
    """
    Look up recent W&B runs for settings.wandb_entity / settings.wandb_project.

    Optionally filter by run_name or run_id. Returns tracking URLs and summary metrics.
    Uses urllib only — no wandb package needed locally.
    """
    api_key = settings.wandb_api_key
    if not api_key:
        return {
            "ok": False,
            "error": "WANDB_API_KEY not set in .env",
        }

    entity = settings.wandb_entity_resolved
    project = settings.wandb_project
    if not entity or not project:
        return {
            "ok": False,
            "error": "WANDB_ENTITY and WANDB_PROJECT must be set in .env",
        }

    try:
        data = _wandb_get(
            f"/api/v1/runs?project={project}&entity={entity}&per_page={limit}",
            api_key,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    runs_raw = data if isinstance(data, list) else data.get("runs", [])
    results: list[dict] = []
    for run in runs_raw:
        rid = run.get("id") or run.get("name") or ""
        rname = run.get("displayName") or run.get("name") or rid
        state = run.get("state", "")
        summary = run.get("summary", {}) or {}
        config = run.get("config", {}) or {}

        # Filter
        if run_id and rid != run_id:
            continue
        if run_name and run_name.lower() not in rname.lower():
            continue

        run_url = f"https://wandb.ai/{entity}/{project}/runs/{rid}"
        results.append({
            "id": rid,
            "name": rname,
            "state": state,
            "url": run_url,
            "summary_keys": list(summary.keys())[:10],
            "best_metric": _best_metric(summary),
        })

    if not results:
        return {
            "ok": True,
            "found": 0,
            "runs": [],
            "note": (
                f"No runs found in {entity}/{project}"
                + (f" matching name='{run_name}'" if run_name else "")
                + (f" id='{run_id}'" if run_id else "")
                + ". The run may still be starting or the entity/project may differ."
            ),
            "wandb_project_url": f"https://wandb.ai/{entity}/{project}",
        }

    return {
        "ok": True,
        "found": len(results),
        "runs": results,
        "wandb_project_url": f"https://wandb.ai/{entity}/{project}",
        "tracking": settings.tracking_links(wandb_run_id=results[0]["id"]) if results else {},
    }


def _best_metric(summary: dict) -> dict:
    """Extract common metric keys if present."""
    candidates = ["best_val_acc", "infer_acc", "AP", "mAP", "IoU", "val_loss", "train_loss"]
    out = {}
    for k in candidates:
        if k in summary:
            out[k] = summary[k]
    if not out:
        # Return first numeric value
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                out[k] = v
                break
    return out
