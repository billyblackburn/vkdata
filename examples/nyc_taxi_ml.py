"""End-to-end Vulkan GPU Machine Learning on Real NYC Taxi Trips (vkpandas + vklearn)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "vkpandas", _ROOT.parent / "delightful-hertz"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import vknumpy as vnp
import vkpandas as vpd
import vklearn as vkl

PARQUET = Path.home() / ".cache" / "hvsampledata" / "nyc_taxi" / "v2" / "nyc_taxi_wide.parq"


def main() -> None:
    print(f"Vulkan device: {vnp.device_info()['name']} ({vnp.device_info()['backend']})")

    t0 = time.perf_counter()
    df = vpd.read_parquet(
        PARQUET,
        columns=["pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "trip_distance", "fare_amount", "tip_amount", "pickup_hour"],
    )
    # Filter to valid Manhattan/NYC trips
    valid = (
        (df["fare_amount"] > 2.5) & (df["fare_amount"] < 100.0) &
        (df["trip_distance"] > 0.1) & (df["trip_distance"] < 50.0) &
        (df["pickup_x"] >= -8242000.0) & (df["pickup_x"] <= -8226000.0) &
        (df["pickup_y"] >= 4968000.0) & (df["pickup_y"] <= 4985000.0)
    )
    clean = df[valid]
    print(f"Loaded & GPU-filtered {len(clean):,} valid trips in {time.perf_counter() - t0:.2f}s")

    # Use a 1,000,000-row subset for fast interactive ML demo (scales to all 11.4M rows too!)
    sub = clean[:1_000_000]
    features = ["trip_distance", "pickup_x", "pickup_y", "dropoff_x", "dropoff_y", "pickup_hour"]
    X = sub[features]
    y_fare = sub["fare_amount"]
    y_tipped = (sub["tip_amount"] > 0.0).astype(vnp.int32)

    X_tr, X_te, yf_tr, yf_te, yt_tr, yt_te = vkl.train_test_split(
        X, y_fare, y_tipped, test_size=0.2, random_state=42
    )

    # 1. GPU Ridge Regression (Fare Prediction)
    t1 = time.perf_counter()
    fare_pipe = vkl.Pipeline([
        ("scaler", vkl.StandardScaler()),
        ("ridge", vkl.Ridge(alpha=1.0)),
    ]).fit(X_tr, yf_tr)
    pred_fare = fare_pipe.predict(X_te)
    dt_reg = time.perf_counter() - t1
    print(
        f"\n[1] GPU Ridge Fare Prediction (800k train / 200k test) in {dt_reg:.3f}s:\n"
        f"    R^2 = {vkl.r2_score(yf_te, pred_fare):.4f} | "
        f"RMSE = ${vkl.root_mean_squared_error(yf_te, pred_fare):.2f} | "
        f"MAE = ${vkl.mean_absolute_error(yf_te, pred_fare):.2f}"
    )

    # 2. GPU Logistic Regression (Tipped vs No-Tip Classification via Newton-IRLS)
    t2 = time.perf_counter()
    tip_pipe = vkl.Pipeline([
        ("scaler", vkl.StandardScaler()),
        ("logreg", vkl.LogisticRegression(C=1.0, max_iter=15)),
    ]).fit(X_tr, yt_tr)
    pred_tip = tip_pipe.predict(X_te)
    dt_clf = time.perf_counter() - t2
    cm = vkl.confusion_matrix(yt_te, pred_tip, num_classes=2).to_numpy()
    print(
        f"\n[2] GPU LogisticRegression Tip Classifier (800k train, {tip_pipe.named_steps['logreg'].n_iter_} IRLS iters) in {dt_clf:.3f}s:\n"
        f"    Accuracy = {vkl.accuracy_score(yt_te, pred_tip) * 100:.2f}% | Confusion Matrix = {cm.tolist()}"
    )

    # 3. GPU K-Means (8 Pickup Hotspot Clusters on 1M Trips)
    t3 = time.perf_counter()
    coords = sub[["pickup_x", "pickup_y"]]
    km = vkl.KMeans(n_clusters=8, max_iter=20, random_state=42).fit(coords)
    dt_km = time.perf_counter() - t3
    counts = vnp.bincount(km.labels_, minlength=8).to_numpy()
    print(
        f"\n[3] GPU KMeans (K=8 on 1,000,000 Manhattan pickups, {km.n_iter_} iters) in {dt_km:.3f}s:\n"
        f"    Cluster sizes: {counts.tolist()}"
    )


if __name__ == "__main__":
    main()
