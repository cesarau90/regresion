# Predicción de precios de vivienda — California Housing

Proyecto de Machine Learning **supervisado de regresión** para una materia de análisis predictivo. Estima el valor mediano de vivienda por distrito censal a partir de ingreso, ubicación y composición del hogar.

## Dataset

[California Housing](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) (censo de EE.UU. de 1990, distribuido públicamente por Aurélien Géron). Se carga directamente por URL, sin credenciales:

```python
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")
```

20,640 distritos censales · 9 variables predictoras (8 numéricas + `ocean_proximity` categórica) · objetivo: `median_house_value`.

A las 8 numéricas se les suman 3 ratios por distrito derivados dentro del pipeline (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`), definidos en `features.py` para compartirlos entre el script, el entrenamiento y la API.

## Cómo correrlo

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python proyecto_regresion.py
```

### Aplicación interactiva

```bash
pip install -r requirements.txt
python entrenar_modelo.py
python app.py
```

Abre `http://127.0.0.1:5000`. El entrenamiento crea `modelo_vivienda.joblib` y `mapa.json`; las consultas posteriores reutilizan esos archivos y no vuelven a entrenar el modelo.

La página incluye un **mapa interactivo de California** (Leaflet, requiere internet para los tiles de OpenStreetMap):

- Clic en el mapa → fija longitud/latitud y autodetecta `ocean_proximity` por la celda más cercana.
- Botón **"Ver superficie de precios del modelo"** → dibuja sobre el mapa la predicción del modelo en una malla de ~0.2° (precalculada en `entrenar_modelo.py`): la silueta de precios de California sale solo de los datos.
- Botón **"Cargar un distrito real"** → llena el formulario con un distrito del censo y compara el valor predicho contra el observado.
- El intervalo P10–P90 sale de dos modelos de regresión cuantílica (`HistGradientBoostingRegressor(loss="quantile")`).

Si el mapa no carga (sin internet), el formulario sigue funcionando igual.

### Deploy en Vercel

Vercel ejecuta `entrenar_modelo.py` durante el build, incluye `modelo_vivienda.joblib` + `mapa.json` + `features.py` en la función Flask y dirige las rutas a `app.py`. No subas `modelo_vivienda.joblib` ni `mapa.json` a Git; después de hacer push de `vercel.json` y `.python-version`, vuelve a desplegar sin reutilizar el caché anterior.

El script (celdas separadas con `# %%`, corribles individualmente en VS Code/Jupyter) genera las gráficas `01`–`05` en `.png`, `resultados_modelos.csv` y las métricas en consola.

### Reporte HTML

`report_template.html` es la única fuente de verdad del texto del reporte. Tras regenerar las gráficas:

```bash
python generar_reporte.py
```

Embebe las 5 gráficas como base64 y produce `index.html` (con el estimador y el mapa, se sirve desde Flask) y `reporte.html` (versión autónoma sin JS, para abrir directo en el navegador).

## Pipeline

Sin data leakage: el feature engineering (ratios fila a fila), la imputación, el escalado (`StandardScaler`) y la codificación (`OneHotEncoder`) van dentro del pipeline, ajustado únicamente con el 80% de entrenamiento (`train_test_split`, `random_state=42`). La validación cruzada re-ajusta todo el pipeline en cada fold.

## Resultados

Cada modelo se evalúa dos veces: validación cruzada de 5 folds sobre el 80% de entrenamiento (media ± desviación del RMSE) y una única medición sobre el 20% de prueba, nunca visto.

| Modelo | CV RMSE (train) | MAE | RMSE | R² |
|---|---|---|---|---|
| **Gradient Boosting** (`HistGradientBoostingRegressor`) | **$42,493 ± 387** | **$29,476** | **$43,074** | **0.814** |
| Random Forest (afinado · GridSearchCV) | $44,825 ± 724 | $31,080 | $46,004 | 0.788 |
| Random Forest | $45,210 ± 338 | $30,680 | $46,165 | 0.786 |
| Regresión Lineal | $59,923 ± 900 | $45,199 | $61,864 | 0.616 |
| Árbol de Decisión | $64,221 ± 768 | $42,098 | $63,421 | 0.597 |

Gradient Boosting gana en las dos mediciones y además con la CV más estable. La importancia por permutación (model-agnóstica, medida sobre el set de prueba) pone la **ubicación geográfica** (`longitude` + `latitude`) y el **ingreso mediano** como las variables que más pesan; `total_bedrooms` y `housing_median_age` casi no aportan.

El ajuste de hiperparámetros con `GridSearchCV` mejoró marginalmente al Random Forest por defecto pero quedó muy lejos de Gradient Boosting — evidencia de que el tipo de algoritmo importó más que su calibración fina.

## Archivos

- `proyecto_regresion.py` — proyecto completo, celda por celda
- `features.py` — ratios por distrito, compartidos entre script, entrenamiento y API
- `entrenar_modelo.py` — entrena el modelo servido + genera `mapa.json`
- `app.py` / `predictor.js` / `predictor.css` — API Flask y front del estimador con mapa
- `report_template.html` + `generar_reporte.py` — fuente y generador del reporte HTML
- `01`–`05_*.png` — gráficas de EDA, evaluación e importancia de variables
- `resultados_modelos.csv` — tabla comparativa de métricas
- `reporte.html` — reporte visual de una página (abrir directo en el navegador)

## Conclusión

Gradient Boosting fue el modelo con mejor desempeño (R² = 0.814, error promedio de ~$29,500), por encima de Random Forest y muy por encima de la regresión lineal base gracias a su capacidad de capturar relaciones no lineales entre ingreso, ubicación y precio. Los ratios por distrito y el uso de validación cruzada endurecieron la comparación. Limitaciones: datos de 1990, faltan variables como estado de la vivienda o metros cuadrados exactos, y el modelo pierde precisión en el extremo superior de precios por un artefacto de censura en el dataset original (valores topados en $500,001).
