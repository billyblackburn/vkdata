"""vklearn: scikit-learn / cuML-style Vulkan GPU machine learning on vknumpy & vkpandas."""

from . import cluster, decomposition, linear_model, metrics, model_selection, neighbors, pipeline, preprocessing
from .cluster import KMeans
from .decomposition import PCA
from .linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from .metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from .model_selection import train_test_split
from .neighbors import KNeighborsClassifier, NearestNeighbors
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, StandardScaler

__all__ = [
    "KMeans",
    "KNeighborsClassifier",
    "LinearRegression",
    "LogisticRegression",
    "MinMaxScaler",
    "NearestNeighbors",
    "PCA",
    "Perceptron",
    "Pipeline",
    "Ridge",
    "StandardScaler",
    "accuracy_score",
    "cluster",
    "confusion_matrix",
    "decomposition",
    "linear_model",
    "mean_absolute_error",
    "mean_squared_error",
    "metrics",
    "model_selection",
    "neighbors",
    "pipeline",
    "preprocessing",
    "r2_score",
    "root_mean_squared_error",
    "train_test_split",
]
