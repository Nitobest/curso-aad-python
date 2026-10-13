# Fichas de referencia

Una página por algoritmo, para leer, imprimir o tener abierta mientras se trabaja en el notebook. Todas siguen la misma estructura: etiquetas (módulo, sesión, supervisado / no supervisado), para qué sirve, la fórmula mínima, la historia en cuatro viñetas, el código en Python con cada parámetro explicado, ventajas, cuidados y la ficha «Dentro del `.fit()`».

Los PDF juntan cada serie en orden de curso.

## Serie 1 · Los algoritmos del curso — [PDF completo](algoritmos/fichas_algoritmos.pdf)

| # | Ficha | Módulo | Sesión | Tipo |
|---|---|---|---|---|
| 01 | [Árbol de decisión](algoritmos/01_arbol.png) | M1 | S1 · S3 · S6 | Supervisado |
| 02 | [Regresión lineal](algoritmos/02_regresion_lineal.png) | M2 | S4 | Supervisado |
| 03 | [Ridge y Lasso](algoritmos/03_ridge_lasso.png) | M2 | S4 | Supervisado |
| 04 | [Regresión logística](algoritmos/04_regresion_logistica.png) | M2 | S5 | Supervisado |
| 05 | [k vecinos más cercanos](algoritmos/05_knn.png) | M2 | S5 | Supervisado |
| 06 | [Bosque aleatorio](algoritmos/06_bosque_aleatorio.png) | M2 | S6 | Supervisado |
| 07 | [K-means](algoritmos/07_kmeans.png) | M3 | S7 · S9 | No supervisado |
| 08 | [DBSCAN](algoritmos/08_dbscan.png) | M3 | S7 | No supervisado |
| 09 | [PCA](algoritmos/09_pca.png) | M3 | S8 | No supervisado |
| 10 | [t-SNE](algoritmos/10_tsne.png) | M3 | S8 | No supervisado |
| 11 | [Gradient boosting](algoritmos/11_gradient_boosting.png) | M4 | S11 | Supervisado |
| 12 | [Validación cruzada](algoritmos/12_validacion_cruzada.png) | M4 | S10 | Evaluación |
| 13 | [Búsqueda de hiperparámetros](algoritmos/13_busqueda_hiperparametros.png) | M4 | S11 | Optimización |

## Serie 2 · Dentro del `.fit()`: qué minimiza cada algoritmo y cómo — [PDF completo](dentro-del-fit/fichas_dentro_del_fit.pdf)

| # | Ficha | Aparece desde | La usan |
|---|---|---|---|
| P1 | [Impureza Gini (y entropía)](dentro-del-fit/P1_gini.png) | S1 | árbol, bosque, boosting |
| P2 | [Suma de cuadrados (MSE, R²)](dentro-del-fit/P2_suma_cuadrados.png) | S4 | regresión lineal, Ridge/Lasso, K-means, boosting |
| P3 | [Penalización L2 y L1](dentro-del-fit/P3_penalizacion.png) | S4 | Ridge, Lasso, logística (C), boosting |
| P4 | [Log-loss](dentro-del-fit/P4_logloss.png) | S5 | logística, boosting |
| P5 | [Distancia e inercia](dentro-del-fit/P5_distancias.png) | S5 | k-NN, K-means, DBSCAN, PCA |
| P6 | [Descenso de gradiente y η](dentro-del-fit/P6_descenso_gradiente.png) | S4 | logística (lbfgs), t-SNE, boosting |
| P7 | [Mapa: qué minimiza cada algoritmo y cómo](dentro-del-fit/P7_mapa_optimizacion.png) | S1 → S11 | los 13 en una tabla |

Las fichas se generan por código, igual que las figuras de los notebooks: misma paleta (azul M1, verde M2, naranja M3, morado M4), mismos datos sintéticos de dos variables, misma ficha de cuatro renglones de la celda A′.
