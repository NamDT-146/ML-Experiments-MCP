from __future__ import annotations

import pytest

from aiaun_mcp.kaggle_ops import classify_kernel_failure


_E1_LOG = """
Installing packages...
FATAL: secret WANDB_API_KEY unavailable (Connection error trying to communicate with service.).
Add it in the kernel UI → Secrets, then Save & Run (API auto-run has no secrets).
"""

_E3_LOG = """
cuda devices=1 names=['Tesla P100-PCIE-16GB']
FATAL: need T4 x2, got n=1 ['Tesla P100-PCIE-16GB']
"""

_E4_LOG = """
cuda devices=2 names=['Tesla T4', 'Tesla T4']
ModuleNotFoundError: No module named 'mask2former'
  File ".../losses/m2f_criterion.py", line 81, in __init__
    from mask2former.modeling.matcher import HungarianMatcher
"""

_E7_LOG = """
Parallel: knet→gpu0, mask_rcnn→gpu1
FileNotFoundError: [Errno 2] No such file or directory: 'splits/voc_hetero_seed42.json'
"""

_NETWORK_LOG = """
Training finished.
DRIVE_SKIP ConnectionError ConnectionError('Connection error trying to communicate with service.')
DONE
"""

_CLEAN_LOG = """
cuda_ready True Tesla T4 sm 7.5
epoch 0 train_loss 1.0 val_acc 0.9
DONE
"""


def test_classify_e1_missing_secret():
    result = classify_kernel_failure(_E1_LOG)
    assert result["error_class"] == "MISSING_SECRET"
    assert len(result["matched_lines"]) >= 1
    assert "runtime_env" in result["remediation"].lower() or "User Secrets" in result["remediation"]


def test_classify_e3_wrong_accelerator():
    result = classify_kernel_failure(_E3_LOG)
    assert result["error_class"] == "WRONG_ACCELERATOR"


def test_classify_e4_module_not_found():
    result = classify_kernel_failure(_E4_LOG)
    assert result["error_class"] == "MODULE_NOT_FOUND"


def test_classify_e7_file_not_found():
    result = classify_kernel_failure(_E7_LOG)
    assert result["error_class"] == "FILE_NOT_FOUND"
    assert "preflight" in result["remediation"].lower()


def test_classify_network_error():
    result = classify_kernel_failure(_NETWORK_LOG)
    assert result["error_class"] == "NETWORK_ERROR"


def test_classify_clean_unknown():
    result = classify_kernel_failure(_CLEAN_LOG)
    assert result["error_class"] == "UNKNOWN"
    assert result["matched_lines"] == []


def test_classify_multiple_classes():
    combined = _E1_LOG + _E4_LOG
    result = classify_kernel_failure(combined)
    # First match wins for primary
    assert "all_classes" in result
    assert len(result["all_classes"]) >= 2
