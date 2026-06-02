import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = BASE_DIR / "settings.json"


def _resolve_path(value, settings_path):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((settings_path.parent / path).resolve())


def load_settings(settings_path=DEFAULT_SETTINGS_PATH):
    settings_path = Path(settings_path)
    with settings_path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    settings["paths"] = {
        key: _resolve_path(val, settings_path)
        for key, val in settings["paths"].items()
    }
    return settings
