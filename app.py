from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_vivienda.joblib"
FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity",
]

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
artifact = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/predecir")
def predecir():
    if artifact is None:
        return jsonify(error="El modelo aún no está entrenado. Ejecuta: python entrenar_modelo.py"), 503

    payload = request.get_json(silent=True) or {}
    missing = [name for name in FEATURES if payload.get(name) in (None, "")]
    if missing:
        return jsonify(error=f"Faltan campos: {', '.join(missing)}"), 400

    try:
        row = {name: payload[name] for name in FEATURES}
        for name in FEATURES[:-1]:
            row[name] = float(row[name])
            if not np.isfinite(row[name]):
                raise ValueError(name)
        if row["ocean_proximity"] not in artifact["metadata"]["categories"]:
            raise ValueError("ocean_proximity")
    except (TypeError, ValueError):
        return jsonify(error="Hay valores inválidos en el formulario."), 400

    frame = pd.DataFrame([row], columns=FEATURES)
    estimate = float(artifact["pipeline"].predict(frame)[0])

    # Intervalo orientativo P10-P90 por regresión cuantílica (dos modelos aparte).
    # Es una referencia de dispersión, no un intervalo de confianza calibrado.
    low = float(artifact["pipeline_low"].predict(frame)[0])
    high = float(artifact["pipeline_high"].predict(frame)[0])
    low, high = min(low, high), max(low, high)

    warnings = []
    for name, limits in artifact["metadata"]["ranges"].items():
        if row[name] < limits[0] or row[name] > limits[1]:
            warnings.append(f"{name} está fuera del rango observado en entrenamiento")

    return jsonify(
        estimate=round(estimate, 2),
        reference_interval=[round(float(low), 2), round(float(high), 2)],
        mae=artifact["metadata"]["mae"],
        histogram=artifact["metadata"]["histogram"],
        warnings=warnings,
        model_context="Valor mediano estimado para un distrito censal de California, en dólares de 1990.",
    )


if __name__ == "__main__":
    app.run(debug=True)
