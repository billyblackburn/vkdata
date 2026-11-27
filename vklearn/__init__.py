"""vklearn: scikit-learn / cuML-style Vulkan GPU machine learning on vknumpy & vkpandas."""

from . import (
    accel,
    cluster,
    datasets,
    decomposition,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    neighbors,
    pipeline,
    preprocessing,
)
from .cluster import DBSCAN, KMeans
from .datasets import FASHION_MNIST_CLASSES, load_california_housing, load_fashion_mnist
from .decomposition import PCA
from .ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from .metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from .model_selection import KFold, cross_val_score, train_test_split
from .neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler

__all__ = [
    "DBSCAN",
    "FASHION_MNIST_CLASSES",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KFold",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "LinearRegression",
    "LogisticRegression",
    "MinMaxScaler",
    "NearestNeighbors",
    "OneHotEncoder",
    "PCA",
    "Perceptron",
    "Pipeline",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "SimpleImputer",
    "StandardScaler",
    "accel",
    "accuracy_score",
    "cluster",
    "confusion_matrix",
    "cross_val_score",
    "datasets",
    "decomposition",
    "ensemble",
    "linear_model",
    "load_california_housing",
    "load_fashion_mnist",
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
