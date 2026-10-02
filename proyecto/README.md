# Proyecto final por etapas

Único entregable obligatorio del curso. El estudiante elige **su propio dataset** (datos abiertos colombianos, UCI o Kaggle; lista curada en el Notebook 0) y lo lleva por el mismo pipeline en paralelo a los datasets de clase.

**Restricciones del dataset:** mínimo 1 000 filas, al menos una variable objetivo continua o categórica, sin texto libre como feature principal.

**Formato de entrega:** organizado como proyecto aunque viva en Colab/Drive — carpeta con `datos/`, el notebook y un `README.md` corto (qué pregunta, de dónde salen los datos, cómo correrlo).

## Entregas

| Entrega | Sesión | Qué se entrega | Peso sugerido |
|---|---|---|---|
| **E1** | S3 | Notebook con dataset elegido, EDA breve, pregunta de ML formulada y variable objetivo justificada; lista de columnas que se conocen *después* de la *y* (candidatas a leakage) | 15 % |
| **E2** | S6 | Baseline trivial (`Dummy*`) + un modelo supervisado, métrica elegida y justificada, split correcto | 25 % |
| **E3** | S9 | Análisis no supervisado del mismo dataset (clusters o PCA) con interpretación honesta, incluyendo lo que *no* se puede concluir | 20 % |
| **E4** | S11 | Modelo final con CV y tuning dentro de `Pipeline`, comparación contra baseline y contra E2 | 25 % |
| **Presentación** | S12 | 5 minutos: problema, pipeline, resultado, qué aprendió del error | 15 % |

## Rúbrica común (cada criterio 1–4)

| Criterio | 1 | 4 |
|---|---|---|
| Corrección técnica | El código no corre o el modelo no corresponde al problema | Corre de punta a punta, modelo y preprocesamiento adecuados |
| Ausencia de leakage / split correcto | Preprocesa con todo el dataset o evalúa sobre train | Split antes de todo; preprocesamiento dentro del `Pipeline`; sin columnas posteriores a la *y* |
| Interpretación honesta | Reporta la métrica sin baseline ni contexto | Compara con baseline, discute límites y qué no se puede afirmar |
| Claridad del notebook | Celdas sueltas sin texto | Secciones del esqueleto del curso, cada resultado con una frase de lectura |

**[verificar]** si el certificado exige nota o solo asistencia — define si la rúbrica es formal o formativa. **[verificar]** política de entregas tardías.

## Archivos previstos en esta carpeta
- `enunciado.md` — enunciado que se entrega al estudiante en S1 (pendiente).
- `datasets_curados.md` — la lista del Notebook 0, extendida con notas de qué pregunta admite cada uno (pendiente).
- `rubrica.md` — versión imprimible de la rúbrica (pendiente).
