"""Feature engineering compartido entre el proyecto, el entrenamiento y la API.

Se mantiene en un solo archivo para que el pipeline serializado (`modelo_vivienda.joblib`)
pueda reconstruir la función al des-serializarse desde cualquiera de los tres scripts.
"""

import numpy as np

# Las 8 variables numéricas que llegan crudas desde el formulario / dataset.
RAW_NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]

# Ratios por distrito derivados de las anteriores.
RATIO_FEATURES = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]


def add_ratio_features(X):
    """Agrega ratios por distrito a un DataFrame de entrada.

    Son operaciones fila a fila (no usan estadísticos del conjunto), así que
    aplicarlas dentro del pipeline no introduce data leakage. Los infinitos que
    puedan surgir de una división por cero se convierten a NaN para que el
    `SimpleImputer` posterior los resuelva.
    """
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"]
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
    X["population_per_household"] = X["population"] / X["households"]
    return X.replace([np.inf, -np.inf], np.nan)
