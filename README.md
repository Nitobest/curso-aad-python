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

## Sesiones

| Sesión | Módulo | Tema | Notebook |
|---|---|---|---|
| 0 | Antes del curso | Diagnóstico: Python, pandas y estadística básica (20–30 min, sin nota) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S00/S00_diagnostico.ipynb) |
| 1 | M1 · Introducción al ML | Qué es ML y el pipeline completo en una clase | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S01/S01_estudiante.ipynb) |
| 2 | M1 · Introducción al ML | Mirar antes de modelar: datos y preprocesamiento | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S02/S02_estudiante.ipynb) |
| 3 | M1 · Introducción al ML | Entrenar, probar y no hacerse trampa · **Entrega E1** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitobest/curso-aad-python/blob/main/sesiones/S03/S03_estudiante.ipynb) |
| 4 | M2 · Supervisado | Regresión lineal y regularización | próximamente |
| 5 | M2 · Supervisado | Clasificación: regresión logística y k-NN | próximamente |
| 6 | M2 · Supervisado | Árboles, bosques y métricas · **Entrega E2** | próximamente |
| 7 | M3 · No supervisado | K-means y DBSCAN | próximamente |
| 8 | M3 · No supervisado | PCA y t-SNE | próximamente |
| 9 | M3 · No supervisado | Segmentación y perfilado · **Entrega E3** | próximamente |
| 10 | M4 · Evaluación y optimización | Validación cruzada | próximamente |
| 11 | M4 · Evaluación y optimización | Hiperparámetros y `Pipeline` · **Entrega E4** | próximamente |
| 12 | Cierre | Presentaciones del proyecto final | — |

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

| Módulo | Dataset | Fuente |
|---|---|---|
| M1 | Incidentes viales de Medellín 2019–2025 | Secretaría de Movilidad de Medellín; Sánchez Corredor, Arango Uribe y Correa Álvarez, *Data* (MDPI, 2026). [Mendeley Data, DOI 10.17632/r6g5dfnpgh.1](https://data.mendeley.com/datasets/r6g5dfnpgh/1), licencia CC BY 4.0 |

Los datos de los demás módulos se agregan aquí cuando se publique cada sesión.
