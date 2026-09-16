from __future__ import annotations

from aiaun_mcp.runtime_env import KERNEL_ENV_LOADER

# Inlined on Kaggle (no aiaun_mcp import). Placeholders: {kernel_slug}
KERNEL_DRIVE_HELPER = r'''
def get_or_create_experiment_folder(drive_service, parent_id, folder_name):
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(folder_name).strip()) or "run"
    name = name[:120]
    qname = name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        "'" + parent_id + "' in parents and name='" + qname
        + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute(num_retries=5)
    items = results.get("files") or []
    if items:
        return items[0]["id"]
    folder = drive_service.files().create(
        body={
            "name": name,
            "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder",
        },
        fields="id",
        supportsAllDrives=True,
    ).execute(num_retries=5)
    return folder.get("id")


def drive_push(folder_files):
    import json as _djson, os as _dos
    # Prefer os.environ (set by _load_aiaun_env), fall back to User Secrets
    sa_raw = _dos.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "")
    root_id = _dos.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
    if not sa_raw or not root_id:
        try:
            from kaggle_secrets import UserSecretsClient
            _us = UserSecretsClient()
            if not sa_raw:
                sa_raw = _us.get_secret("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
            if not root_id:
                root_id = _us.get_secret("GOOGLE_DRIVE_FOLDER_ID")
        except Exception:
            pass
    if not sa_raw or not root_id:
        print("DRIVE_SKIP no Drive credentials available", flush=True)
        return
    info = _djson.loads(sa_raw)
    del sa_raw
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "google-api-python-client", "google-auth"])
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    info.clear()
    svc = build("drive", "v3", credentials=creds, cache_discovery=False, static_discovery=True)
    slug_name = "{kernel_slug}".replace("/", "_")
    ver = os.environ.get("KAGGLE_KERNEL_VERSION") or "run"
    slug_id = get_or_create_experiment_folder(svc, root_id, slug_name)
    run_id = get_or_create_experiment_folder(svc, slug_id, ver)
    print("DRIVE_FOLDER", run_id, "https://drive.google.com/drive/folders/" + run_id, flush=True)
    for fp in sorted(folder_files.glob("*")):
        if not fp.is_file():
            continue
        media = MediaFileUpload(str(fp), resumable=True)
        created = svc.files().create(
            body={"name": fp.name, "parents": [run_id]},
            media_body=media,
            fields="id,name",
            supportsAllDrives=True,
        ).execute(num_retries=5)
        print("DRIVE_OK", created.get("id"), fp.name, flush=True)
'''


SMOKE_SCRIPT = r'''# AiAuN synthetic color-class smoke. No secrets in this file.
# Attach dataset: {dataset_slug}
# Expects labels.csv with columns: file,label  and PPM images in the same folder.

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

{env_loader}
INPUT = Path("/kaggle/input")
print("inputs:", sorted(p.name for p in INPUT.iterdir()) if INPUT.exists() else "missing", flush=True)

def find_labels() -> Path:
    hits = list(INPUT.glob("**/labels.csv"))
    if not hits:
        raise SystemExit("labels.csv not found under /kaggle/input")
    return hits[0]

def read_ppm(path: Path) -> tuple[int, int, int]:
    data = path.read_bytes()
    if data.startswith(b"P6"):
        header, body = data.split(b"\n255\n", 1)
        w, h = 32, 32
        px = body[: w * h * 3]
        n = max(len(px) // 3, 1)
        r = sum(px[i] for i in range(0, len(px), 3)) // n
        g = sum(px[i] for i in range(1, len(px), 3)) // n
        b = sum(px[i] for i in range(2, len(px), 3)) // n
        return r, g, b
    raise SystemExit(f"unsupported image: {path}")

labels_path = find_labels()
root = labels_path.parent
rows = list(csv.DictReader(labels_path.open()))
print("n_samples", len(rows), "root", root, flush=True)

feats = []
ys = []
for row in rows:
    rgb = read_ppm(root / row["file"])
    feats.append(rgb)
    ys.append(row["label"])

# Nearest mean-color prototype (no extra pip)
sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
for (r, g, b), y in zip(feats, ys):
    s = sums[y]
    s[0] += r
    s[1] += g
    s[2] += b
    s[3] += 1
means = {k: (v[0] / v[3], v[1] / v[3], v[2] / v[3]) for k, v in sums.items()}

def pred(rgb):
    best, best_d = None, 1e18
    for lab, m in means.items():
        d = (rgb[0] - m[0]) ** 2 + (rgb[1] - m[1]) ** 2 + (rgb[2] - m[2]) ** 2
        if d < best_d:
            best, best_d = lab, d
    return best

correct = sum(int(pred(x) == y) for x, y in zip(feats, ys))
acc = correct / max(len(ys), 1)
print(f"SMOKE_ACC={acc:.3f} correct={correct}/{len(ys)}", flush=True)
print("WANDB: set User Secret WANDB_API_KEY to log remotely; this smoke does not embed keys.", flush=True)

art = Path("/kaggle/working/artifacts")
art.mkdir(parents=True, exist_ok=True)
(art / "metrics.json").write_text(
    '{"smoke_acc": ' + str(acc) + '}', encoding="utf-8"
)
print("wrote", art / "metrics.json", flush=True)

import json
{drive_helper}
try:
    drive_push(art)
except Exception as _e:
    print("DRIVE_SKIP", type(_e).__name__, repr(_e), flush=True)

print("DONE", flush=True)
'''


def _fill_script(src: str, *, dataset_slug: str, kernel_slug: str) -> str:
    return (
        src.replace("{env_loader}", KERNEL_ENV_LOADER)
        .replace("{drive_helper}", KERNEL_DRIVE_HELPER)
        .replace("{dataset_slug}", dataset_slug)
        .replace("{kernel_slug}", kernel_slug.replace("/", "_"))
    )


def smoke_script(*, dataset_slug: str, kernel_slug: str = "aiaun-color-smoke") -> str:
    return _fill_script(SMOKE_SCRIPT, dataset_slug=dataset_slug, kernel_slug=kernel_slug)


RESNET50_SMOKE_SCRIPT = r'''# AiAuN ResNet50 GPU smoke on synthetic color-cls. No secrets in this file.
# Attach dataset: {dataset_slug}
# Accelerator: GPU T4 (machine_shape NvidiaTeslaT4). P100 sm_60 cannot run current Kaggle torch.

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

{env_loader}

def _ensure_cuda_torch() -> None:
    """Kaggle default GPU is often P100 (sm_60). Image torch is sm_70+ only."""
    if os.environ.get("AIAUN_TORCH_FIXED") == "1":
        return
    import torch
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    print(
        "cuda",
        torch.cuda.is_available(),
        name,
        "sm",
        "{}.{}".format(*cap),
        "torch",
        torch.__version__,
        flush=True,
    )
    if torch.cuda.is_available() and cap[0] < 7:
        print("INSTALL_TORCH_CU118 for P100/sm_60", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch==2.2.2",
                "torchvision==0.17.2",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
            ]
        )
        os.environ["AIAUN_TORCH_FIXED"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


_ensure_cuda_torch()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

INPUT = Path("/kaggle/input")
ART = Path("/kaggle/working/artifacts")
ART.mkdir(parents=True, exist_ok=True)
print(
    "cuda_ready",
    torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "sm",
    "{}.{}".format(*torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "cpu",
    flush=True,
)


def find_labels() -> Path:
    hits = list(INPUT.glob("**/labels.csv"))
    if not hits:
        raise SystemExit("labels.csv not found under /kaggle/input")
    return hits[0]


def read_ppm_chw(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise SystemExit("not P6 " + str(path))
    rest = data[2:].lstrip()
    if rest.startswith(b"#"):
        rest = rest.split(b"\n", 1)[1]
    header, body = rest.split(b"\n255\n", 1)
    wh = header.replace(b"\n", b" ").split()
    w, h = int(wh[0]), int(wh[1])
    arr = torch.frombuffer(bytearray(body[: w * h * 3]), dtype=torch.uint8).reshape(h, w, 3)
    x = arr.permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return x.squeeze(0)


class ColorSet(Dataset):
    def __init__(self, root: Path, rows: list):
        self.root = root
        self.rows = rows
        self.classes = sorted(set(r["label"] for r in rows))
        self.to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return read_ppm_chw(self.root / r["file"]), self.to_idx[r["label"]], r["file"]


{drive_helper}


labels_path = find_labels()
root = labels_path.parent
rows = list(csv.DictReader(labels_path.open()))
print("n_samples", len(rows), "root", root, flush=True)
by = {}
for r in rows:
    by.setdefault(r["label"], []).append(r)
train_rows, val_rows = [], []
for lab, items in by.items():
    items = sorted(items, key=lambda x: x["file"])
    val_rows.append(items[-1])
    train_rows.extend(items[:-1] or items)
print("split train", len(train_rows), "val", len(val_rows), flush=True)

train_ds = ColorSet(root, train_rows)
val_ds = ColorSet(root, val_rows)
classes = train_ds.classes
print("classes", classes, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.zeros(1, device=device).add_(1)
    torch.cuda.synchronize()
    print("cuda_probe_ok", flush=True)
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)
opt = torch.optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
loader = DataLoader(train_ds, batch_size=4, shuffle=True)
vloader = DataLoader(val_ds, batch_size=4)

best_acc, best_path = -1.0, ART / "best.pt"
EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    n = 0
    for xb, yb, _fn in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        tr_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    model.eval()
    correct, tot = 0, 0
    with torch.no_grad():
        for xb, yb, _fn in vloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            tot += int(yb.numel())
    acc = correct / max(tot, 1)
    print("epoch", epoch, "train_loss", round(tr_loss / max(n, 1), 4), "val_acc", round(acc, 4), flush=True)
    if acc >= best_acc:
        best_acc = acc
        torch.save({"model": model.state_dict(), "classes": classes, "val_acc": best_acc}, best_path)
        print("saved", best_path, "val_acc", best_acc, flush=True)

try:
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
preds = []
correct, tot = 0, 0
with torch.no_grad():
    for xb, yb, fns in vloader:
        xb = xb.to(device)
        logits = model(xb)
        idx = logits.argmax(1).cpu().tolist()
        yb_l = yb.tolist()
        names = list(fns) if not isinstance(fns, str) else [fns]
        for i, name in enumerate(names):
            p = classes[idx[i]]
            t = classes[yb_l[i]]
            preds.append({"file": name, "pred": p, "true": t})
            correct += int(p == t)
            tot += 1
inf_acc = correct / max(tot, 1)
print("INFER_BEST_ACC", round(inf_acc, 4), "n", tot, flush=True)
metrics = {
    "backbone": "resnet50",
    "epochs": EPOCHS,
    "best_val_acc": best_acc,
    "infer_acc": inf_acc,
    "device": str(device),
    "n_train": len(train_ds),
    "n_val": len(val_ds),
}
(ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(ART / "predictions.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")
print("artifacts", list(p.name for p in ART.iterdir()), flush=True)
try:
    drive_push(ART)
except Exception as e:
    print("DRIVE_SKIP", type(e).__name__, repr(e), flush=True)
print("DONE", flush=True)
'''


def resnet50_smoke_script(
    *, dataset_slug: str, kernel_slug: str = "aiaun-resnet50-gpu-smoke"
) -> str:
    return _fill_script(RESNET50_SMOKE_SCRIPT, dataset_slug=dataset_slug, kernel_slug=kernel_slug)


def kernel_source_has_secrets(source: str) -> bool:
    lowered = source.lower()
    needles = (
        "kgat_",
        "ghp_",
        "github_pat_",
        "wandb_v1_",
        "wandb_api_key=",
        "begin private key",
        '"private_key": "-----',
    )
    return any(n in lowered for n in needles)


RUN_KAGGLE_EXPERIMENT_PROMPT = """You orchestrate experiments via the aiaun MCP tools.

User intent: run a training setting on Kaggle (example: semi-mask2former coco teacher-student).

Rules:
1. Ask for anything missing. Required: data_dir_or_kaggle_slug, code_version, config_path.
   Call resolve_experiment_request first. If missing is non-empty, ask the user; do not invent values.
2. If several YAML configs match the setting, call list_repo_configs and list candidates; ask which file.
3. inspect_local_dir for a local data path. Prefer an existing Kaggle slug over upload.
4. kaggle_dataset_check. Upload with kaggle_dataset_push only if missing and the user confirmed. No AiAuN size cap.
5. Never paste GITHUB_TOKEN, WANDB_API_KEY, or the Drive service-account JSON into kernel source.
   Kernels must read Kaggle User Secrets: GITHUB_TOKEN, WANDB_API_KEY, GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON,
   GOOGLE_DRIVE_FOLDER_ID. After training, upload /kaggle/working/artifacts to Drive (Shared Drive folder).
6. kaggle_kernel_push then poll kaggle_kernel_status / kaggle_kernel_logs. One tool call must not block for hours.
   After push/status, ALWAYS paste tracking.kaggle_kernel, tracking.wandb_project, and tracking.drive_folder for the user.
   Call experiment_tracking_links if those fields are missing.
7. Optional backup on the MCP host (SSH, no browser): kaggle_kernel_output_to_drive after the kernel finishes.
8. Personal vs org: tools use AIAUN_OWNER_MODE from .env. Do not switch silently.

For GPU ResNet50 smoke on the synthetic color dataset, use aiaun_resnet50_gpu_smoke_script and kaggle_kernel_push with enable_gpu=true (T4 via machine_shape). Still not a full COCO Mask2Former train.
"""


GENERATE_NOTEBOOK_PROMPT = """Write a Kaggle *script* (not a huge notebook) that:
- Reads GITHUB_TOKEN, WANDB_API_KEY, GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON, GOOGLE_DRIVE_FOLDER_ID
  from Kaggle User Secrets (kaggle_secrets.UserSecretsClient). Never hardcode them. Never print them.
- git clone https://x-access-token:${GITHUB_TOKEN}@github.com/{org}/{repo}.git -b {branch}
- pip install only what is missing (keep it small).
- wandb.init(entity=..., project=...) using env from secrets.
- Runs the user-confirmed config file from the cloned repo.
- Writes small survivable artifacts under /kaggle/working/artifacts/.
- After the run, parse the SA JSON from the secret with json.loads +
  google.oauth2.service_account.Credentials.from_service_account_info (no JSON file in Output).
  Upload artifacts to GOOGLE_DRIVE_FOLDER_ID with supportsAllDrives=True.
  That folder MUST be on a Shared Drive (service accounts have no My Drive quota). Login-less.
- Prints stage logs flush=True so kaggle_kernel_logs can be tailed. Print DRIVE_OK or DRIVE_SKIP only.
"""
