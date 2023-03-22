"""Enums to be used in the project."""
from enum import Enum


class ModelType(Enum):
    """Enum for the different model types."""

    SPARSE = "sparse"
    DENSE = "dense"


class Retrieval(Enum):
    """Enum for the different retrieval types."""

    EXHAUSTIVE = "exhaustive"
    NETVLAD = "netvlad"


class Features(Enum):
    """Enum for the different feature types."""

    SIFT = "sift"
    SUPERPOINT = "superpoint_aachen"


class Matcher(Enum):
    """Enum for the different matcher types."""

    SUPERGLUE = "superglue"
    NN_RATIO = "NN-ratio"
    NN_MUTUAL = "NN-mutual"
