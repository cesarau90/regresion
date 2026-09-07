import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import RAW_NUMERIC, RATIO_FEATURES, add_ratio_features


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
MODEL_PATH = BASE_DIR / "modelo_vivienda.joblib"
MAP_PATH = BASE_DIR / "mapa.json"

GRID_STEP = 0.2                      # grados (~22 km) de la malla sobre California
OTHER_NUMERIC = [c for c in RAW_NUMERIC if c not in ("longitude", "latitude")]


def construir_pipeline(preprocessor, modelo):
    return Pipeline([
        ("features", FunctionTransformer(add_ratio_features)),
        ("preprocessor", preprocessor),
        ("modelo", modelo),
    ])


def build_map_assets(pipeline, X_train, y_train):
    """Genera mapa.json: la superficie de precios que predice el modelo sobre una
    malla de California, la categoría de proximidad al mar por celda y un puñado
    de distritos reales para el botón de ejemplos.
    """
    labels = sorted(X_train["ocean_proximity"].dropna().unique().tolist())
    label_idx = {name: i for i, name in enumerate(labels)}

    geo = X_train[["latitude", "longitude"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=40).fit(geo)

    lats = np.arange(32.4, 42.2, GRID_STEP)
    lons = np.arange(-124.5, -114.0, GRID_STEP)
    grid = np.array([(lat, lon) for lat in lats for lon in lons])
    dist, idx = nn.kneighbors(grid)

    feature_rows, kept = [], []
    for (lat, lon), d, neigh_idx in zip(grid, dist, idx):
        if d[0] > 0.25:                      # sin distritos cerca -> fuera del mapa habitado
            continue
        neigh = X_train.iloc[neigh_idx]
        row = {c: float(neigh[c].median()) for c in OTHER_NUMERIC}
        row["latitude"] = float(lat)
        row["longitude"] = float(lon)
        row["ocean_proximity"] = neigh["ocean_proximity"].mode().iloc[0]
        feature_rows.append(row)
        kept.append((lat, lon, label_idx[row["ocean_proximity"]]))

    preds = pipeline.predict(pd.DataFrame(feature_rows))
    cells = [
        [round(lat, 3), round(lon, 3), int(round(price, -3)), prox]
        for (lat, lon, prox), price in zip(kept, preds)
    ]
    prices = np.array([c[2] for c in cells])

    order = y_train.sort_values()
    fracs = (0.02, 0.16, 0.32, 0.5, 0.67, 0.82, 0.93, 0.99)
    example_idx = [order.index[int(f * (len(order) - 1))] for f in fracs]
    examples = []
    for i in dict.fromkeys(example_idx):     # sin duplicados, conserva orden
        r = X_train.loc[i]
        examples.append({
            **{c: float(r[c]) for c in RAW_NUMERIC},
            "ocean_proximity": r["ocean_proximity"],
            "real": int(y_train.loc[i]),
        })

    MAP_PATH.write_text(json.dumps({
        "ocean_labels": labels,
        "grid_step": GRID_STEP,
        "price_domain": [int(np.percentile(prices, 5)), int(np.percentile(prices, 95))],
        "cells": cells,
        "examples": examples,
    }), encoding="utf-8")
    print(f"Mapa guardado en {MAP_PATH.name} ({len(cells)} celdas, {len(examples)} ejemplos)")


def main():
    print("Descargando y preparando California Housing...")
    df = pd.read_csv(DATA_URL).drop_duplicates()
    df = df[df["median_house_value"] != 500001].copy()
    X = df.drop(columns="median_house_value")
    y = df["median_house_value"]

    numeric = RAW_NUMERIC + RATIO_FEATURES
    categorical = ["ocean_proximity"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo de punto: Gradient Boosting (fue el mejor en la comparación offline).
    pipeline = construir_pipeline(
        preprocessor, HistGradientBoostingRegressor(random_state=42)
    )
    pipeline.fit(X_train, y_train)

    # Dos modelos cuantílicos para el intervalo orientativo P10-P90.
    pipeline_low = construir_pipeline(
        preprocessor,
        HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=42),
    ).fit(X_train, y_train)
    pipeline_high = construir_pipeline(
        preprocessor,
        HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=42),
    ).fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    counts, edges = np.histogram(y_train, bins=24)
    # Los rangos que la API usa para avisar "fuera de lo observado" solo cubren
    # las variables crudas del formulario, no los ratios derivados.
    ranges = {
        name: [float(X_train[name].min()), float(X_train[name].max())]
        for name in RAW_NUMERIC
    }
    artifact = {
        "pipeline": pipeline,
        "pipeline_low": pipeline_low,
        "pipeline_high": pipeline_high,
        "metadata": {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
            "ranges": ranges,
            "categories": sorted(X_train["ocean_proximity"].dropna().unique().tolist()),
            "histogram": {
                "counts": counts.astype(int).tolist(),
                "edges": edges.astype(float).tolist(),
            },
        },
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"Modelo guardado en {MODEL_PATH.name}")
    print(f"MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | R²: {r2:.3f}")

    build_map_assets(pipeline, X_train, y_train)


if __name__ == "__main__":
    main()
