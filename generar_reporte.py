"""Construye index.html y reporte.html a partir de report_template.html.

- report_template.html es la única fuente de verdad del texto del reporte.
- Las 5 gráficas (creadas por proyecto_regresion.py) se embeben como base64.
- index.html: se sirve desde Flask, conserva el link a predictor.css y el
  <script> de predictor.js (mapa interactivo + estimador).
- reporte.html: versión autónoma para abrir directo en el navegador, sin JS.
"""

import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE = BASE_DIR / "report_template.html"

IMAGES = {
    "__IMG_HIST__": "01_histograma_precio.png",
    "__IMG_CORR__": "02_matriz_correlacion.png",
    "__IMG_EDA__": "03_eda_scatter_boxplot.png",
    "__IMG_RESID__": "04_real_vs_predicho_residuales.png",
    "__IMG_FEAT__": "05_feature_importances.png",
}

PREDICTOR_CSS_LINK = '<link rel="stylesheet" href="predictor.css">\n'
PREDICTOR_JS_TAG = '<script src="predictor.js"></script>'


def main():
    html = TEMPLATE.read_text(encoding="utf-8")

    missing = [name for name in IMAGES.values() if not (BASE_DIR / name).exists()]
    if missing:
        raise SystemExit(
            "Faltan gráficas: " + ", ".join(missing) + "\nEjecuta primero: python proyecto_regresion.py"
        )
    for token, filename in IMAGES.items():
        encoded = base64.b64encode((BASE_DIR / filename).read_bytes()).decode("ascii")
        html = html.replace(token, encoded)

    (BASE_DIR / "index.html").write_text(html, encoding="utf-8")

    standalone = html.replace(PREDICTOR_CSS_LINK, "").replace(PREDICTOR_JS_TAG + "\n", "").replace(PREDICTOR_JS_TAG, "")
    (BASE_DIR / "reporte.html").write_text(standalone, encoding="utf-8")

    print("Generados index.html y reporte.html desde report_template.html")


if __name__ == "__main__":
    main()
