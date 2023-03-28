"""Enums to be used in the project."""
from enum import Enum


class ModelType(Enum):
    """Enum for the different model types."""

    SPARSE = "sparse"
    DENSE = "dense"


class Retrieval(Enum):
    """Enum for the different retrieval types."""

    NETVLAD = "netvlad"
    COSPLACE = "cosplace"
    POSES = "poses"
    COVISIBILITY = "covisibility"


class Features(Enum):
    """Enum for the different feature types."""

    SIFT = "sift"
    SUPERPOINT_MAX = "superpoint_max"


class Matcher(Enum):
    """Enum for the different matcher types."""

    SUPERGLUE = "superglue"
    NN_RATIO = "NN-ratio"
    NN_MUTUAL = "NN-mutual"


class PairsFrom(Enum):
    """Enum for the different pairs from types."""

    POSES = "poses"
    RETRIEVAL = "retrieval"
