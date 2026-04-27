from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VARIANT = "fastgs_baseline"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --variant/--config support. "
            "Install it with `pip install pyyaml` or update the conda environment."
        ) from exc

    if not path.exists():
        raise FileNotFoundError("Config file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping: {}".format(path))
    return data


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _variant_path(variant: str) -> Path:
    return PROJECT_ROOT / "configs" / "variants" / "{}.yaml".format(variant)


def load_train_config(variant: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    selected_variant = variant or DEFAULT_VARIANT
    config = _read_yaml(_variant_path(selected_variant))
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        config = _merge_dict(config, _read_yaml(path))
    config.setdefault("name", selected_variant)
    config.setdefault("scorer", "fastgs_photometric")
    config.setdefault("training_args", {})
    return config


def apply_argparse_defaults(parser, config: Dict[str, Any], sections: Iterable[str] = ("training_args",)) -> None:
    defaults = {"scorer": config.get("scorer", "fastgs_photometric")}
    for section in sections:
        values = config.get(section, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError("Config section {!r} must be a mapping.".format(section))
        defaults.update(values)
    parser.set_defaults(**defaults)
