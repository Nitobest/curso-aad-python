# Proyecto final por etapas

Es el único entregable obligatorio del curso. Usted elige **su propio dataset** (datos abiertos de Colombia, UCI o Kaggle; hay una lista de sugerencias en el Notebook 0) y lo lleva por el mismo pipeline que vemos en clase con los datasets del curso. El proyecto es **individual**.

**Requisitos del dataset:** mínimo 1 000 filas, al menos una variable objetivo (continua o categórica) y sin texto libre como variable principal.

## Entregas

| Entrega | Se presenta en | Se entrega antes de | Qué se entrega | Peso |
|---|---|---|---|---|
| **E1** | Sesión 3 | Sesión 4 | Dataset elegido (fuente y qué es una fila), pregunta de ML y tipo de problema, regla de la *y* con su proporción o rango, exploración comentada, la *y* en el tiempo si hay fechas, **columnas sospechosas de fuga** (las que se conocen *después* de la *y*) con su justificación, y decisiones tomadas | 15 % |
| **E2** | Sesión 6 | Sesión 7 | Columnas quitadas (futuro, identificadores) · partición entrenamiento/prueba correcta, con el preprocesamiento ajustado solo con entrenamiento · **línea base** (`Dummy*`) · un modelo supervisado del Módulo 2 con una figura · **métrica justificada** (mínimo dos frases) · lectura honesta | 25 % |
| **E3** | Sesión 9 | Sesión 10 | **PCA y clustering sobre su dataset, con estabilidad e interpretación honesta**: columnas elegidas y por qué (sin la *y*) · escalado y manejo de vacíos · PCA (varianza, biplot, qué es PC1) · K-means o DBSCAN con `k` argumentado (codo, silueta, **estabilidad**, utilidad) · perfil y nombre de cada grupo · validación externa (su *y*, una categoría, el tiempo o el mapa) · mínimo tres frases de lo que **no** se puede concluir | 20 % |
| **E4** | Sesión 11 | Sesión 12 | Modelo final en `Pipeline`: prueba apartada · línea base + **al menos 3 modelos, incluido gradient boosting**, con validación cruzada ± desviación · búsqueda de hiperparámetros justificada (Optuna o `RandomizedSearchCV`) · **una sola** evaluación en prueba, comparada con la validación cruzada y con E2 · SHAP (o importancias) con la advertencia de que no es causalidad · lectura honesta de cuánto ganó afinar · `Pipeline` guardado con `joblib` y una celda que lo carga y predice | 25 % |
| **Presentación** | Sesión 12 | — | Entre 3,5 y 5,5 minutos según el número de inscritos, **4 láminas**: (1) la pregunta y la métrica; (2) los datos, el pipeline y qué trampa evitó; (3) el resultado honesto: línea base → E2 → final, validación cruzada ± y prueba; (4) un error que cometió y qué **no** se puede concluir. Traiga el `.joblib` de E4 para la prueba de humo | 15 % |

Cada notebook de sesión trae, en su sección 📦, la tabla detallada y una plantilla para empezar.

## Cómo entregar

- **Canal:** por el canal que indique el profesor en la primera clase.
- **Formato:** **un notebook de Colab por entrega**, compartido como enlace de Google Drive o de Colab con permiso de **lector** ("Cualquier persona con el enlace"). Si sus datos no son públicos, comparta también el CSV (o un enlace a él) con el mismo permiso.
- **E4** es la excepción: entregue una carpeta de Drive compartida con el notebook, los datos (o su enlace), el archivo `.joblib` del `Pipeline` y un `README` corto (qué pregunta, de dónde salen los datos, cómo correrlo).
- **Nombre sugerido:** `E1_Apellido_Nombre.ipynb` (y así `E2_…`, `E3_…`, `E4_…`).
- **Qué incluir siempre:** al inicio, una celda de texto con su nombre, la pregunta del proyecto y la fuente de los datos. Antes de compartir, use **Entorno de ejecución → Reiniciar y ejecutar todo** para comprobar que corre de principio a fin, y deje las salidas guardadas.

## Rúbrica

La rúbrica sirve para retroalimentar cada entrega. El **certificado depende del 80 % de asistencia**; la rúbrica no cambia ese requisito, salvo que Educación Continua indique otra cosa.

Cada criterio se califica de 1 a 4.

| Criterio | 1 · Insuficiente | 2 · En desarrollo | 3 · Bien | 4 · Excelente |
|---|---|---|---|---|
| **Corrección técnica** | El código no corre o el modelo no corresponde al problema | Corre con errores que hay que corregir a mano, o hay pasos mal aplicados | Corre de principio a fin; algún detalle técnico discutible | Corre de principio a fin; modelo y preprocesamiento adecuados al problema |
| **Ausencia de fuga y partición correcta** (E1, E2, E4) | Preprocesa con todo el dataset, evalúa sobre entrenamiento o usa columnas posteriores a la *y* | Hay partición, pero algo se ajusta con la prueba o queda una columna sospechosa sin discutir | Partición antes de todo y preprocesamiento solo con entrenamiento; alguna decisión sin justificar | Partición antes de todo; preprocesamiento dentro del `Pipeline`; columnas posteriores a la *y* identificadas y quitadas |
| **Escalado y variables justificadas** (en E3, en lugar del anterior) | No escala, o mete la *y* o identificadores para agrupar | Escala, pero las columnas elegidas no se justifican | Escala y justifica las columnas; queda alguna duda sin resolver (resúmenes junto a sus partes, vacíos) | Escala, justifica cada columna, explica qué hizo con los vacíos y **la *y* no se usó para agrupar** (solo para leer los grupos al final) |
| **Interpretación honesta** | Reporta un número sin línea base ni contexto | Compara con la línea base, pero exagera lo que el resultado permite afirmar | Compara con la línea base y menciona límites | Compara con la línea base, discute límites y dice con claridad qué **no** se puede afirmar |
| **Claridad del notebook** | Celdas sueltas, sin texto | Algo de texto, pero hay que adivinar qué se hizo | Secciones claras; a algunos resultados les falta su lectura | Secciones del esqueleto del curso; cada resultado tiene una frase que lo lee |

En E1 todavía no hay modelo: "ausencia de fuga" se califica por las columnas sospechosas que identificó y cómo las justificó.

**Presentación (Sesión 12)**, cada criterio de 1 a 4:

| Criterio | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **Claridad de la pregunta** | No se entiende qué predice o descubre | Se entiende la pregunta, pero no para quién ni con qué métrica | Pregunta y métrica claras | Pregunta, usuario y métrica claros desde la primera lámina |
| **Resultado honesto** | Solo un número, sin comparación | Compara con algo, pero mezcla validación y prueba | Línea base → E2 → final, con validación cruzada y prueba | Lo anterior, y dice cuánto ganó afinar sin exagerar |
| **Límites y error** | No menciona límites | Menciona límites genéricos | Cuenta un error real y un límite concreto | Cuenta un error, cómo lo detectó y qué **no** se puede concluir |
| **Tiempo y 4 láminas** | Se pasa mucho del tiempo o no trae láminas | Se pasa del tiempo o usa muchas más láminas | Cumple el tiempo con pequeños desajustes | Cumple el tiempo con las 4 láminas pedidas |

**Conversión a nota (si se necesita):** nota de la entrega = (promedio de sus criterios ÷ 4) × peso de la entrega; la nota del proyecto es la suma de las cinco. Ejemplo: E2 con criterios 4, 3, 3 y 4 → promedio 3,5 → 3,5 ÷ 4 × 25 % ≈ 21,9 %.
