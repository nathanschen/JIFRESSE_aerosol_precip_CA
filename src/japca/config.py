from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data)!r}")
    return data


def _expand_placeholders(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_placeholders(item, context) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in context:
            return str(context[name])
        if name in os.environ:
            return os.environ[name]
        return match.group(0)

    return PLACEHOLDER_RE.sub(replace, value)


def _config_path(name: str) -> Path:
    return CONFIG_DIR / name


def load_paths_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else _config_path("paths.yaml")
    raw = load_yaml(config_path)
    repo_root = raw.get("repo_root", str(REPO_ROOT))
    context = {
        "repo_root": repo_root,
        "jifresse_root": raw.get("jifresse_root"),
    }
    expanded = _expand_placeholders(raw, context)
    return expanded


def load_variables_config(path: str | Path | None = None) -> dict[str, Any]:
    return _expand_placeholders(
        load_yaml(Path(path) if path is not None else _config_path("variables.yaml")),
        load_paths_config(),
    )


def load_feature_groups_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(Path(path) if path is not None else _config_path("feature_groups.yaml"))


def load_metrics_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(Path(path) if path is not None else _config_path("metrics.yaml"))


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
