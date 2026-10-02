"""Data handling, dataset loaders, open-set protocols, and low-variance curation."""

from .dataset import Market1501Dataset, SyntheticUniformDatasetGenerator, ReIDSample
from .protocol import OpenSetProtocol, OpenSetSplit
from .low_variance import extract_hsv_color_histogram, curate_low_variance_subset

__all__ = [
    "Market1501Dataset",
    "SyntheticUniformDatasetGenerator",
    "ReIDSample",
    "OpenSetProtocol",
    "OpenSetSplit",
    "extract_hsv_color_histogram",
    "curate_low_variance_subset",
]
