from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
MODEL_PATH = BASE_DIR / "modelo_vivienda.joblib"


def main():
    print("Descargando y preparando California Housing...")
    df = pd.read_csv(DATA_URL).drop_duplicates()
    df = df[df["median_house_value"] != 500001].copy()
    X = df.drop(columns="median_house_value")
    y = df["median_house_value"]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
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
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("modelo", RandomForestRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    counts, edges = np.histogram(y_train, bins=24)
    ranges = {
        name: [float(X_train[name].min()), float(X_train[name].max())]
        for name in numeric
    }
    artifact = {
        "pipeline": pipeline,
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


if __name__ == "__main__":
    main()
