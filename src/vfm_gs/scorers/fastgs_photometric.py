from vfm_gs.utils.fast_utils import compute_gaussian_score_fastgs

from .registry import register_scorer


register_scorer("fastgs_photometric", compute_gaussian_score_fastgs)
