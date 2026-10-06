# Introducción al Análisis Avanzado de Datos con Python

**Universidad del Valle · Escuela de Estadística · Educación Continua**
12 sesiones · lunes y miércoles 7:00–9:00 p.m. · todo corre en Google Colab

Aquí está el material del curso. No hay que instalar nada: cada notebook se abre en Colab con un clic y descarga solo los datos que necesita.

---

## Cómo trabajar cada sesión

1. Haga clic en el botón **Open in Colab** de la sesión del día.
2. Inicie sesión con su cuenta de Google.
3. **Antes de escribir nada:** menú **Archivo → Guardar una copia en Drive**. Trabaje siempre en esa copia; si no la guarda, lo que escriba se pierde al cerrar la pestaña.
4. Ejecute las celdas **una por una** (Shift + Enter), en orden. Las primeras celdas descargan los datos y las figuras del curso; tardan unos segundos.
5. Cuando vea:
   - 🔍 **Predice** → escriba su predicción *antes* de ejecutar la celda siguiente.
   - ✏️ **Completa** → reemplace los `...` por su código. Si ejecuta todo de una vez, el notebook se detiene ahí hasta que lo complete.
   - ⚡ **Mini-reto** → opcional, para quien termine antes.

> ¿El notebook dice que no pudo bajar los datos? Siga a la celda **Plan B**: le pedirá subir el archivo, que el instructor comparte por el grupo del curso.

---

## 🧪 Laboratorio de algoritmos

Cada algoritmo, para tocarlo: **[nitobest.github.io/curso-aad-python/laboratorio](https://nitobest.github.io/curso-aad-python/laboratorio/)**. Mírelo paso a paso, juegue con sus perillas y supere el reto. Funciona también en el celular.

---

## Sesiones

| Sesión | Módulo | Tema | Notebook |
|---|---|---|---|
| 0 | Antes del curso | Diagnóstico: Python, pandas y estadística básica (20–30 min, sin nota) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S00/S00_diagnostico.ipynb) |
| 1 | M1 · Introducción al ML | Qué es ML y el pipeline completo en una clase | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S01/S01_estudiante.ipynb) |
| 2 | M1 · Introducción al ML | Mirar antes de modelar: datos y preprocesamiento | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S02/S02_estudiante.ipynb) |
| 3 | M1 · Introducción al ML | Entrenar, probar y no hacerse trampa · **Entrega E1** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S03/S03_estudiante.ipynb) |
| 4 | M2 · Supervisado | Regresión lineal: qué hay dentro del `.fit()` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S04/S04_estudiante.ipynb) |
| 5 | M2 · Supervisado | Clasificación: logística, k-NN… y el accuracy que miente | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S05/S05_estudiante.ipynb) |
| 6 | M2 · Supervisado | Árboles y bosques: muchos árboles mejor que uno · **Entrega E2** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S06/S06_estudiante.ipynb) |
| 7 | M3 · No supervisado | K-means y DBSCAN: encontrar grupos sin respuesta | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S07/S07_estudiante.ipynb) |
| 8 | M3 · No supervisado | PCA y t-SNE: ver muchas dimensiones (sin creerle todo al dibujo) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S08/S08_estudiante.ipynb) |
| 9 | M3 · No supervisado | Segmentar municipios de Colombia: PCA + K-means + mapa · **Entrega E3** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S09/S09_estudiante.ipynb) |
| 10 | M4 · Evaluación y optimización | Validación cruzada: ¿es bueno de verdad o tuve suerte con la partición? | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S10/S10_estudiante.ipynb) |
| 11 | M4 · Evaluación y optimización | Afinar sin hacer trampa: gradient boosting, Optuna, SHAP y el modelo final · **Entrega E4** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S11/S11_estudiante.ipynb) |
| 12 | Cierre | Presentaciones del proyecto final y cierre del curso | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S12/S12_estudiante.ipynb) |

Los notebooks se publican a medida que avanza el curso.

---

## Proyecto final

Es el único entregable obligatorio. Usted elige su propio dataset y lo lleva por el mismo pipeline que vemos en clase, en cuatro entregas parciales (E1–E4) y una presentación en la Sesión 12. Detalles y rúbrica: [`proyecto/README.md`](proyecto/README.md).

Para cargar **sus** datos en Colab, súbalos a su Google Drive y use:

```python
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/mi_proyecto/datos.csv')   # o pd.read_excel(...)
```

---

## Datos de clase

Todos los archivos están en la carpeta [`datos/`](datos/): ahí puede verlos, descargarlos o abrirlos en Excel (instrucciones en su README).

| Módulo | Dataset | Fuente |
|---|---|---|
| M1 | Incidentes viales de Medellín 2019–2025 | Secretaría de Movilidad de Medellín; Sánchez Corredor, Arango Uribe y Correa Álvarez, *Data* (MDPI, 2026). [Mendeley Data, DOI 10.17632/r6g5dfnpgh.1](https://data.mendeley.com/datasets/r6g5dfnpgh/1), licencia CC BY 4.0 |
| M2 (regresión) | Insurance (cargos médicos, EE. UU.) | Lantz, *Machine Learning with R*; PyCaret datasets |
| M2 (regresión) | Saber 11, Valle del Cauca, 2022 | ICFES — [datos.gov.co, Resultados únicos Saber 11](https://www.datos.gov.co/d/kgxf-xxbe) |
| M2 (clasificación) | Bank Marketing (telemercadeo bancario) | Moro, Cortez y Rita (2014), [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| M3 | Palmer Penguins (pingüinos de la Antártida) | Gorman, Williams y Fraser (2014); Horst, Hill y Gorman (2020), [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/), CC0 |
| M3 | Dígitos escritos a mano (8×8) | Alpaydin y Kaynak, [UCI ML Repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits); incluido en scikit-learn |
| M3 | Pobreza multidimensional municipal, Censo 2018 | DANE, [Medida de Pobreza Multidimensional municipal de fuente censal](https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-y-desigualdad/medida-de-pobreza-multidimensional-de-fuente-censal); coordenadas: DIVIPOLA ([datos.gov.co](https://www.datos.gov.co/d/gdxc-w37w)) |
