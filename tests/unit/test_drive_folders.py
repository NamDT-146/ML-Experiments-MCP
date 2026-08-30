from aiaun_mcp.drive_ops import run_folder_name, sanitize_folder_name


def test_sanitize_folder_name():
    assert sanitize_folder_name("namdtgk14/aiaun-resnet50-gpu-smoke") == "namdtgk14_aiaun-resnet50-gpu-smoke"
    assert sanitize_folder_name("") == "run"


def test_run_folder_name():
    assert run_folder_name("") == "run"
    assert run_folder_name("3") == "v3"
    assert run_folder_name("v3") == "v3"
