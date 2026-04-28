import importlib
from typing import Callable, Dict, Tuple


ScorerFn = Callable[..., object]
_SCORERS: Dict[str, ScorerFn] = {}
_LAZY_SCORERS: Dict[str, Tuple[str, str]] = {
    "fastgs_photometric": ("vfm_gs.scorers.fastgs_photometric", "compute_gaussian_score_fastgs"),
    "vfm_topology_scorer": ("vfm_gs.scorers.vfm_topology", "compute_gaussian_score_fastgs_with_vfm"),
}


def register_scorer(name: str, scorer: ScorerFn) -> ScorerFn:
    if not name:
        raise ValueError("Scorer name must be non-empty.")
    if name in _SCORERS:
        raise ValueError("Scorer {!r} is already registered.".format(name))
    _SCORERS[name] = scorer
    return scorer


def get_scorer(name: str) -> ScorerFn:
    if name in _SCORERS:
        return _SCORERS[name]
    if name in _LAZY_SCORERS:
        module_name, attr_name = _LAZY_SCORERS[name]
        module = importlib.import_module(module_name)
        scorer = getattr(module, attr_name)
        _SCORERS[name] = scorer
        return scorer
    available = ", ".join(list_scorers()) or "<none>"
    raise KeyError("Unknown scorer {!r}. Available scorers: {}".format(name, available))


def list_scorers():
    return sorted(set(_SCORERS).union(_LAZY_SCORERS))
