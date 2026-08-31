# Predicción de precios de vivienda — California Housing

Proyecto de Machine Learning **supervisado de regresión** para una materia de análisis predictivo. Estima el valor mediano de vivienda por distrito censal a partir de ingreso, ubicación y composición del hogar.

## Dataset

[California Housing](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) (censo de EE.UU. de 1990, distribuido públicamente por Aurélien Géron). Se carga directamente por URL, sin credenciales:

```python
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")
```

20,640 distritos censales · 9 variables predictoras (8 numéricas + `ocean_proximity` categórica) · objetivo: `median_house_value`.

## Cómo correrlo

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python proyecto_regresion.py
```

El script (celdas separadas con `# %%`, corribles individualmente en VS Code/Jupyter) genera las gráficas `01`–`05` en `.png`, `resultados_modelos.csv` y las métricas en consola.

## Pipeline

Sin data leakage: imputación, escalado (`StandardScaler`) y codificación (`OneHotEncoder`) van dentro de un `ColumnTransformer`, ajustado únicamente con el 80% de entrenamiento (`train_test_split`, `random_state=42`).

## Resultados

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | $30,624 | $45,855 | **0.789** |
| Random Forest (afinado · GridSearchCV) | $31,911 | $46,821 | 0.780 |
| Regresión Lineal | $45,917 | $62,562 | 0.607 |
| Árbol de Decisión | $41,843 | $63,065 | 0.601 |

`median_income` concentra el 43% de la importancia del modelo ganador, seguido de `ocean_proximity_INLAND` (16%) y la ubicación geográfica (~12% cada eje).

El ajuste de hiperparámetros con `GridSearchCV` no superó al Random Forest por defecto en el set de prueba — evidencia de que el tipo de algoritmo importó más que su calibración fina.

## Archivos

- `proyecto_regresion.py` — proyecto completo, celda por celda
- `01`–`05_*.png` — gráficas de EDA, evaluación e importancia de variables
- `resultados_modelos.csv` — tabla comparativa de métricas
- `reporte.html` — reporte visual de una página (abrir directo en el navegador)

## Conclusión

Random Forest fue el modelo con mejor desempeño (R² = 0.789, error promedio de ~$30,600), superando con claridad a la regresión lineal base gracias a su capacidad de capturar relaciones no lineales entre ingreso, ubicación y precio. Limitaciones: datos de 1990, faltan variables como estado de la vivienda o metros cuadrados exactos, y el modelo pierde precisión en el extremo superior de precios por un artefacto de censura en el dataset original (valores topados en $500,001).
