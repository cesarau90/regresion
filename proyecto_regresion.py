# %% [markdown]
# # Proyecto de Regresión Supervisada — Predicción de precios de vivienda
# Dataset: California Housing (housing.csv, Aurélien Géron)

# %% Celda 1 — Importar librerías y cargar datos
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # para guardar gráficas a archivo sin necesitar pantalla
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import RAW_NUMERIC, RATIO_FEATURES, add_ratio_features

sns.set_style("whitegrid")

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)
print(df.head())

# %% Celda 2 — Inspección general
print("Shape:", df.shape)
df.info()

# %% Celda 3 — Estadísticas descriptivas
print(df.describe())

# %% Celda 4 — Nulos y duplicados
print("Valores nulos por columna:\n", df.isnull().sum())
print("\nFilas duplicadas:", df.duplicated().sum())

# %% Celda 5 — Distribución de la variable objetivo
plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], bins=40, kde=True)
plt.title("Distribución del precio de vivienda")
plt.xlabel("Precio ($)")
plt.savefig("01_histograma_precio.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Celda 6 — Matriz de correlación (solo numéricas)
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.savefig("02_matriz_correlacion.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Celda 7 — Relación ingreso vs precio, y precio por proximidad al mar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.3, ax=axes[0])
axes[0].set_title("Ingreso mediano vs Precio")
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", ax=axes[1])
axes[1].set_title("Precio según proximidad al mar")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("03_eda_scatter_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Celda 8 — Eliminar duplicados
df = df.drop_duplicates()
print("Shape tras quitar duplicados:", df.shape)

# %% Celda 9 — Outliers: decisión sobre el target censurado
Q1 = df["median_house_value"].quantile(0.25)
Q3 = df["median_house_value"].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
print("Límite superior IQR:", limite_superior)
print("Filas con valor == 500001 (tope censurado):", (df["median_house_value"] == 500001).sum())

df = df[df["median_house_value"] != 500001]
print("Shape final:", df.shape)

# %% Celda 10 — Feature engineering + preprocesamiento dentro del pipeline
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Ratios por distrito: habitaciones y dormitorios por hogar y ocupación media.
# Son operaciones fila a fila, así que van dentro del pipeline sin data leakage.
# (definidos en features.py para compartirlos con la API)
feature_engineer = FunctionTransformer(add_ratio_features)
numeric_features = RAW_NUMERIC + RATIO_FEATURES
categorical_features = ["ocean_proximity"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

def construir_pipeline(modelo):
    """Pipeline completo: ratios -> imputación/escalado/encoding -> modelo."""
    return Pipeline(steps=[
        ("features", feature_engineer),
        ("preprocessor", preprocessor),
        ("modelo", modelo),
    ])

# %% Celda 11 — División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# %% Celda 12 — Definir y entrenar pipelines completos
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}

pipelines = {}
for nombre, modelo in modelos.items():
    pipe = construir_pipeline(modelo)
    pipe.fit(X_train, y_train)
    pipelines[nombre] = pipe
    print(f"{nombre} entrenado")

# %% Celda 13 — Tabla comparativa: validación cruzada (train) + set de prueba
# La CV de 5 folds sobre train da una estimación con incertidumbre y evita
# concluir a partir de un único split. El set de prueba queda como juez final.
resultados = []
for nombre, pipe in pipelines.items():
    cv_rmse = -cross_val_score(
        pipe, X_train, y_train, cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    resultados.append({
        "Modelo": nombre,
        "CV_RMSE_medio": cv_rmse.mean(),
        "CV_RMSE_std": cv_rmse.std(),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

df_resultados = pd.DataFrame(resultados).sort_values("R2", ascending=False)
print(df_resultados.to_string(index=False))

# %% Celda 14 — Gráficas: valores reales vs predichos, y residuales (mejor modelo)
mejor_nombre = df_resultados.iloc[0]["Modelo"]
mejor_pipe = pipelines[mejor_nombre]
y_pred_mejor = mejor_pipe.predict(X_test)
residuales = y_test - y_pred_mejor

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred_mejor, alpha=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_xlabel("Precio real")
axes[0].set_ylabel("Precio predicho")
axes[0].set_title(f"Real vs Predicho ({mejor_nombre})")

axes[1].scatter(y_pred_mejor, residuales, alpha=0.3)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("Precio predicho")
axes[1].set_ylabel("Residual (real - predicho)")
axes[1].set_title("Gráfica de residuales")
plt.tight_layout()
plt.savefig("04_real_vs_predicho_residuales.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Celda 15 — Importancia por permutación del mejor modelo
# Model-agnóstico (sirve igual para Random Forest o Gradient Boosting) y se mide
# sobre el set de prueba: cuánto empeora el R² al barajar cada variable de entrada.
perm = permutation_importance(
    mejor_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
df_importancias = pd.DataFrame({
    "Variable": X_test.columns,
    "Importancia": perm.importances_mean,
    "std": perm.importances_std,
}).sort_values("Importancia", ascending=False).head(10)

orden = df_importancias.iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(orden["Variable"], orden["Importancia"], xerr=orden["std"], color="#1b6b74")
plt.title(f"Importancia por permutación ({mejor_nombre})")
plt.xlabel("Caída media de R² al permutar la variable")
plt.tight_layout()
plt.savefig("05_feature_importances.png", dpi=120, bbox_inches="tight")
plt.close()
print(df_importancias)

# %% Celda 16 — Función para predecir un caso nuevo
def predecir_precio(pipe, **caracteristicas):
    """Recibe características de una vivienda y devuelve el precio estimado."""
    entrada = pd.DataFrame([caracteristicas])
    precio = pipe.predict(entrada)[0]
    return round(precio, 2)

ejemplo = dict(
    longitude=-122.23, latitude=37.88, housing_median_age=41.0,
    total_rooms=880.0, total_bedrooms=129.0, population=322.0,
    households=126.0, median_income=8.3252, ocean_proximity="NEAR BAY",
)
precio_estimado = predecir_precio(mejor_pipe, **ejemplo)
print(f"Precio estimado: ${precio_estimado:,.2f}")

# %% Celda 17 — Resumen final
print(df_resultados.to_string(index=False))

# %% Celda 18 — GridSearchCV para afinar Random Forest
# Búsqueda centrada alrededor de los valores por defecto (que fueron competitivos),
# no en profundidades bajas que solo regularizan de más.
pipe_rf = construir_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))

param_grid = {
    "modelo__n_estimators": [200, 400],
    "modelo__max_depth": [None, 30],
    "modelo__min_samples_leaf": [1, 2, 4],
    "modelo__max_features": [1.0, "sqrt"],
}

grid_search = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)
print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor RMSE en CV:", -grid_search.best_score_)

# %% Celda 19 — Evaluar el modelo afinado en el set de prueba
mejor_rf_tuned = grid_search.best_estimator_
y_pred_tuned = mejor_rf_tuned.predict(X_test)

mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2_tuned = r2_score(y_test, y_pred_tuned)

cv_rmse_tuned = -cross_val_score(
    mejor_rf_tuned, X_train, y_train, cv=5,
    scoring="neg_root_mean_squared_error", n_jobs=-1,
)
df_resultados = pd.concat([
    df_resultados,
    pd.DataFrame([{
        "Modelo": "Random Forest (afinado)",
        "CV_RMSE_medio": cv_rmse_tuned.mean(),
        "CV_RMSE_std": cv_rmse_tuned.std(),
        "MAE": mae_tuned, "RMSE": rmse_tuned, "R2": r2_tuned,
    }]),
], ignore_index=True).sort_values("R2", ascending=False)

print(df_resultados.to_string(index=False))

# %% Celda 20 — Guardar tabla final de resultados a CSV
df_resultados.to_csv("resultados_modelos.csv", index=False)
print("Listo. Gráficas y resultados guardados en la carpeta del proyecto.")
