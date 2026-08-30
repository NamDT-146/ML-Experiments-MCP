from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("AIAUN_LIVE_TEST", "0") in {"1", "true", "yes"}:
        return
    skip = pytest.mark.skip(reason="set AIAUN_LIVE_TEST=1 to hit Kaggle/Drive")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
