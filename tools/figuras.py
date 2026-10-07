"""
figuras.py — helpers de figuras para los notebooks del curso
Introducción al Análisis Avanzado de Datos con Python (Univalle).

Regla del curso (documento maestro, sección 6.4): ningún modelo se ajusta con
`.fit()` sin tres figuras —
    (A) intuición del algoritmo ANTES del código, sobre datos sintéticos de 2 variables;
    (B) confirmación con los datos del curso DESPUÉS del `.fit()`;
    (C) "qué pasa cuando cambio X": fila de subplots variando el hiperparámetro clave.
Cada helper devuelve la figura y le escribe abajo una leyenda de una frase ("qué mirar").

Uso en Colab (el notebook baja este archivo del repo junto a los datos):
    import figuras as fg
    fg.pipeline("modelo")
    fg.arbol_paso_a_paso()

Módulo 1 (árbol de decisión): pipeline, arbol_paso_a_paso, arbol_como_elige_el_corte,
arbol_reglas, mapa_hora_dia, arbol_profundidades. `cajas_hojas(ax, modelo, xlim, ylim)` dibuja el borde de
cada hoja sobre cualquier mapa de 2 variables (hojas vecinas con la misma clase no se pierden).
Los helpers de M2–M4 se agregan en su momento, con la misma firma (datos → fig).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------------ paleta del curso
AZUL = "#2E6FB7"     # clase 0 / "solo daños" / entrenamiento
ROJO = "#C4463A"     # clase 1 / "con víctimas" / prueba
GRIS = "#8C8C8C"
NEGRO = "#222222"
MODULO = {1: "#2E6FB7", 2: "#2C8C5B", 3: "#D97C1F", 4: "#7A4DB8"}
CMAP_CLASES = ListedColormap([AZUL, ROJO])
CMAP_PROB = LinearSegmentedColormap.from_list("prob", [AZUL, "#F2F2F2", ROJO])

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
})


def _leyenda(fig, texto):
    """Escribe la frase 'qué mirar' debajo de la figura."""
    fig.text(0.5, -0.01, "Qué mirar: " + texto, ha="center", va="top", fontsize=10,
             style="italic", color=NEGRO, wrap=True)
    return fig


# ================================================================== PIPELINE
ETAPAS = ["DATOS", "PREPROCESAMIENTO", "MODELO", "EVALUACIÓN", "INTERPRETACIÓN"]


def pipeline(etapa_hoy=None, modulo=1, subtitulo=None):
    """Diagrama del pipeline con la etapa de hoy resaltada.

    etapa_hoy: None o "todas" resalta todo; o una etapa ("modelo") o lista de etapas.
    """
    if etapa_hoy in (None, "todas"):
        hoy = set(ETAPAS)
    else:
        hoy = {e.upper() for e in ([etapa_hoy] if isinstance(etapa_hoy, str) else etapa_hoy)}
    color = MODULO.get(modulo, AZUL)
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2); ax.axis("off")
    for i, e in enumerate(ETAPAS):
        x = 0.2 + i * 2.2
        activo = e in hoy
        caja = FancyBboxPatch((x, 0.55), 1.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                              fc=color if activo else "white", ec=color if activo else GRIS,
                              lw=2 if activo else 1.2)
        ax.add_patch(caja)
        ax.text(x + 0.95, 1.0, e, ha="center", va="center", fontsize=9.5,
                color="white" if activo else GRIS, fontweight="bold" if activo else "normal")
        if i < len(ETAPAS) - 1:
            ax.annotate("", xy=(x + 2.2, 1.0), xytext=(x + 1.9, 1.0),
                        arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.5))
    if subtitulo:
        ax.text(5.5, 0.15, subtitulo, ha="center", va="center", fontsize=10, color=NEGRO)
    return fig


# ================================================================== ÁRBOL — (A) intuición
def _datos_sinteticos(n=300, semilla=7):
    """Dos variables 'hora' (0–24) y 'velocidad' (20–100) y una clase 0/1 con una regla
    escondida y algo de ruido. Solo sirven para VER cómo corta un árbol."""
    rng = np.random.default_rng(semilla)
    hora = rng.uniform(0, 24, n)
    vel = rng.uniform(20, 100, n)
    p = 0.15 + 0.6 * ((hora < 6) | (vel > 70)) + 0.15 * (hora < 6) * (vel > 70)
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"hora": hora, "velocidad": vel}), y


def _regiones(ax, modelo, X, y, xcol="hora", ycol="velocidad", titulo=""):
    from sklearn.tree import DecisionTreeClassifier  # noqa
    xx, yy = np.meshgrid(np.linspace(X[xcol].min(), X[xcol].max(), 200),
                         np.linspace(X[ycol].min(), X[ycol].max(), 200))
    malla = pd.DataFrame({xcol: xx.ravel(), ycol: yy.ravel()})
    z = modelo.predict_proba(malla)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap=CMAP_PROB, alpha=0.55)
    cajas_hojas(ax, modelo, (xx.min(), xx.max()), (yy.min(), yy.max()), columnas=(xcol, ycol))
    ax.scatter(X[xcol], X[ycol], c=y, cmap=CMAP_CLASES, s=14, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(titulo)


def cajas_hojas(ax, modelo, xlim, ylim, columnas=None, color=NEGRO, lw=0.7, alpha=0.55):
    """Dibuja el borde de CADA hoja de un árbol de 2 variables (un rectángulo por hoja).

    El fondo pintado por clase o por probabilidad no basta: dos hojas vecinas que predicen la misma clase
    quedan del mismo color y la línea que las separa desaparece ("dice 4 hojas pero veo 3 zonas").
    Los rectángulos se leen directamente de la estructura del árbol (`modelo.tree_`), así que son exactos.
    `columnas` = nombres (x, y) del gráfico; si el modelo se entrenó con las columnas en otro orden, se
    usan sus `feature_names_in_` para saber cuál es cuál. Si el modelo no es un árbol, no hace nada.
    Devuelve el número de hojas dibujadas."""
    t = getattr(modelo, "tree_", None)
    if t is None:
        return 0
    ix, iy = 0, 1
    nombres = getattr(modelo, "feature_names_in_", None)
    if columnas is not None and nombres is not None:
        nombres = list(nombres)
        ix, iy = nombres.index(columnas[0]), nombres.index(columnas[1])
    n_dibujadas = [0]

    def _recorrer(n, x0, x1, y0, y1):
        if x1 <= x0 or y1 <= y0:
            return
        izq, der = t.children_left[n], t.children_right[n]
        if izq == -1:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw,
                                       alpha=alpha, zorder=2))
            n_dibujadas[0] += 1
            return
        f, u = t.feature[n], t.threshold[n]
        if f == ix:
            _recorrer(izq, x0, min(x1, u), y0, y1); _recorrer(der, max(x0, u), x1, y0, y1)
        elif f == iy:
            _recorrer(izq, x0, x1, y0, min(y1, u)); _recorrer(der, x0, x1, max(y0, u), y1)
        else:  # una variable que no está en el gráfico: ambas ramas ocupan la misma caja
            _recorrer(izq, x0, x1, y0, y1); _recorrer(der, x0, x1, y0, y1)

    _recorrer(0, xlim[0], xlim[1], ylim[0], ylim[1])
    return n_dibujadas[0]


def arbol_paso_a_paso(profundidades=(1, 2, 3, 4), n=300):
    """(A) Cómo un árbol parte el plano: una pregunta, dos, tres, cuatro.
    Datos sintéticos de 2 variables; puntos = clase real; fondo = lo que predice el árbol."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = _datos_sinteticos(n)
    fig, axes = plt.subplots(1, len(profundidades), figsize=(4 * len(profundidades), 3.8), sharey=True)
    for ax, d in zip(axes, profundidades):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        _regiones(ax, m, X, y, titulo=f"profundidad {d}: {'1 pregunta' if d == 1 else f'hasta {d} preguntas'}\n"
                                      f"{m.get_n_leaves()} regiones")
    axes[0].set_ylabel("velocidad")
    plt.tight_layout()
    return _leyenda(fig, "Cada pregunta '¿variable ≤ umbral?' corta el plano con una línea recta "
                         "(horizontal o vertical); el árbol solo puede dibujar rectángulos.")


def arbol_como_elige_el_corte(n=300):
    """(A, detalle) Qué minimiza el árbol: prueba todos los umbrales de una variable y se queda
    con el que deja las dos mitades más 'puras' (impureza de Gini más baja)."""
    X, y = _datos_sinteticos(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col in zip(axes, ["hora", "velocidad"]):
        v = X[col].values
        orden = np.sort(v)
        umbrales = (orden[:-1] + orden[1:]) / 2
        ginis = []
        for u in umbrales:
            izq, der = y[v <= u], y[v > u]
            g = lambda s: 1 - (s.mean() ** 2 + (1 - s.mean()) ** 2) if len(s) else 0
            ginis.append((len(izq) * g(izq) + len(der) * g(der)) / len(y))
        ginis = np.array(ginis)
        mejor = umbrales[ginis.argmin()]
        ax.plot(umbrales, ginis, color=NEGRO, lw=1.5)
        ax.axvline(mejor, color=ROJO, ls="--")
        ax.annotate(f"mejor corte: {col} ≤ {mejor:.1f}\nimpureza {ginis.min():.3f}",
                    xy=(mejor, ginis.min()), xytext=(10, 25), textcoords="offset points",
                    fontsize=9, color=ROJO, arrowprops=dict(arrowstyle="->", color=ROJO))
        ax.set_xlabel(f"umbral probado sobre '{col}'"); ax.set_ylabel("impureza (Gini) tras el corte")
        ax.set_title(f"¿Dónde cortar '{col}'?")
    plt.tight_layout()
    return _leyenda(fig, "El árbol no adivina: prueba todos los cortes de todas las variables y elige el de "
                         "impureza mínima (el valle). Luego repite dentro de cada mitad.")


# ================================================================== ÁRBOL — (B) confirmación con los datos del curso
def _tabla_hora_dia(datos, col_hora="hora_num", col_dia="dia_semana", col_y="con_victimas"):
    t = datos.pivot_table(index=col_dia, columns=col_hora, values=col_y, aggfunc="mean")
    return t.reindex(index=range(7), columns=range(24))


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _mapa(ax, tabla, titulo, vmin=0.4, vmax=0.95):
    im = ax.imshow(tabla.values, aspect="auto", cmap=CMAP_PROB, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_yticks(range(7)); ax.set_yticklabels(DIAS)
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("hora del día")
    ax.set_title(titulo)
    return im


def mapa_hora_dia(datos, profundidad=3, col_hora="hora_num", col_dia="dia_semana", col_y="con_victimas"):
    """(B) Izquierda: % real de incidentes con víctimas por hora × día. Derecha: lo que
    predice un árbol que solo ve esas dos variables. Los rectángulos del árbol 'calcan' el mapa."""
    from sklearn.tree import DecisionTreeClassifier
    real = _tabla_hora_dia(datos, col_hora, col_dia, col_y)
    m = DecisionTreeClassifier(max_depth=profundidad, random_state=42).fit(datos[[col_hora, col_dia]], datos[col_y])
    malla = pd.DataFrame([(h, d) for d in range(7) for h in range(24)], columns=[col_hora, col_dia])
    pred = pd.DataFrame(m.predict_proba(malla)[:, 1].reshape(7, 24))
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    _mapa(axes[0], real, "Datos reales: % de incidentes con víctimas")
    im = _mapa(axes[1], pred, f"Árbol de profundidad {profundidad} (solo hora y día)")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label("proporción con víctimas")
    return _leyenda(fig, "El mapa real es suave (madrugada y domingo, más víctimas); el árbol lo aproxima con "
                         "rectángulos: cada rectángulo es una hoja.")


# ================================================================== ÁRBOL — (C) qué pasa cuando cambio la profundidad
def arbol_profundidades(X_train, y_train, X_test, y_test, profundidades=(1, 2, 4, 8, 16, None),
                        col_hora="hora_num", col_dia="dia_semana"):
    """(C) Fila de arriba: el árbol de 2 variables (hora, día) a distintas profundidades — se ve
    cómo los rectángulos se vuelven cada vez más finos. Abajo: accuracy en entrenamiento y
    prueba del árbol COMPLETO (todas las columnas de X) para cada profundidad."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    etiquetas = [str(d) if d is not None else "sin límite" for d in profundidades]
    fig = plt.figure(figsize=(3.2 * len(profundidades), 7))
    gs = fig.add_gridspec(2, len(profundidades), height_ratios=[1, 1.15], hspace=0.55)
    malla = pd.DataFrame([(h, d) for d in range(7) for h in range(24)], columns=[col_hora, col_dia])
    acc_tr, acc_te, hojas = [], [], []
    for j, d in enumerate(profundidades):
        m2 = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train[[col_hora, col_dia]], y_train)
        ax = fig.add_subplot(gs[0, j])
        _mapa(ax, pd.DataFrame(m2.predict_proba(malla)[:, 1].reshape(7, 24)),
              f"profundidad {etiquetas[j]}\n{m2.get_n_leaves()} hojas")
        if j:
            ax.set_yticklabels([]); ax.set_ylabel("")
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
        acc_tr.append(accuracy_score(y_train, m.predict(X_train)))
        acc_te.append(accuracy_score(y_test, m.predict(X_test)))
        hojas.append(m.get_n_leaves())
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(len(profundidades))
    ax.plot(xs, acc_tr, "o-", color=AZUL, label="entrenamiento")
    ax.plot(xs, acc_te, "s-", color=ROJO, label="prueba (datos nunca vistos)")
    for x, a, b, h in zip(xs, acc_tr, acc_te, hojas):
        ax.annotate(f"{h:,} hojas", (x, max(a, b)), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=GRIS)
    ax.set_xticks(xs); ax.set_xticklabels(etiquetas); ax.set_xlabel("max_depth del árbol completo (todas las columnas de X)")
    ax.set_ylabel("accuracy"); ax.legend(loc="center left"); ax.set_ylim(min(acc_te) - 0.03, 1.005)
    ax.set_title("¿Más profundo = mejor? Solo hasta cierto punto")
    return _leyenda(fig, "La curva azul sube siempre (el árbol memoriza); la roja sube, se aplana y luego cae. "
                         "Lo que importa es la roja.")


# ================================================================== utilidades genéricas
def guardar(fig, ruta):
    """Guarda una figura con la leyenda incluida (para las diapos)."""
    fig.savefig(ruta, bbox_inches="tight", dpi=160)
    return ruta


# ================================================================== S1 — tipos de aprendizaje (diapos)
def supervisado_vs_no_supervisado(n=300):
    """Los mismos puntos dos veces: con etiqueta (supervisado: hay una y que imitar) y sin ella
    (no supervisado: solo estructura). Datos sintéticos de 2 variables."""
    X, y = _datos_sinteticos(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(X["hora"], X["velocidad"], c=y, cmap=CMAP_CLASES, s=22, alpha=0.85, edgecolor="white", lw=0.4)
    axes[0].set_title("Supervisado: cada fila trae su respuesta (y)")
    axes[1].scatter(X["hora"], X["velocidad"], c=GRIS, s=22, alpha=0.85, edgecolor="white", lw=0.4)
    axes[1].set_title("No supervisado: solo hay X; se busca estructura")
    for ax in axes:
        ax.set_xlabel("hora"); ax.set_ylabel("velocidad")
    axes[0].scatter([], [], c=AZUL, label="solo daños (0)"); axes[0].scatter([], [], c=ROJO, label="con víctimas (1)")
    axes[0].legend(loc="upper right")
    return _leyenda(fig, "izquierda: el modelo aprende a imitar el color. Derecha: no hay color que imitar; solo se pueden buscar grupos o resumir.")


def formula(latex, ruta=None, fontsize=30, ancho=8, alto=1.4):
    """Renderiza una fórmula (mathtext de matplotlib) como figura, para pegarla en las diapos."""
    fig = plt.figure(figsize=(ancho, alto))
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize, color=NEGRO)
    if ruta:
        fig.savefig(ruta, bbox_inches="tight", dpi=200, transparent=True)
    return fig


# ================================================================== S2 — mirar antes de modelar (figuras de etapa, sin modelo)
VERDE_OK = "#F2F2F2"


def mapa_vacios(df, col_orden="fecha", disfrazados=("N/D",), muestra=1500, col_anio="anio"):
    """Izquierda: cada fila de la imagen es un incidente (ordenados en el tiempo) y cada columna una
    variable; gris oscuro = vacío real (NaN), rojo = vacío disfrazado de texto ("N/D").
    Derecha: % de vacíos por columna. Sirve para ver CUÁNTO falta y también DÓNDE (¿en qué años?)."""
    d = df.sort_values(col_orden).reset_index(drop=True)
    idx = np.linspace(0, len(d) - 1, min(muestra, len(d))).astype(int)
    d = d.iloc[idx]
    M = d.isna().astype(int).values
    for j, c in enumerate(d.columns):
        if d[c].dtype == object:
            M[d[c].astype(str).isin(disfrazados).values, j] = 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    axes[0].imshow(M, aspect="auto", interpolation="nearest",
                   cmap=ListedColormap([VERDE_OK, "#4D4D4D", ROJO]), vmin=0, vmax=2)
    axes[0].set_xticks(range(len(d.columns))); axes[0].set_xticklabels(d.columns, rotation=60, ha="left")
    axes[0].xaxis.tick_top()
    if col_anio in d:
        anios = d[col_anio].values
        cambios = [0] + [i for i in range(1, len(anios)) if anios[i] != anios[i - 1]]
        axes[0].set_yticks(cambios); axes[0].set_yticklabels([str(anios[i]) for i in cambios])
    axes[0].set_xlabel("cada fila de la imagen es un incidente, ordenados en el tiempo (gris claro = dato presente)")
    axes[0].spines[:].set_visible(False)
    real = df.isna().mean() * 100
    disf = pd.Series({c: (df[c].astype(str).isin(disfrazados).mean() * 100 if df[c].dtype == object else 0)
                      for c in df.columns})
    orden = (real + disf).sort_values().index
    axes[1].barh(orden, real[orden], color="#4D4D4D", label="vacío real (NaN)")
    axes[1].barh(orden, disf[orden], left=real[orden], color=ROJO, label=f"disfrazado ({', '.join(disfrazados)})")
    for i, c in enumerate(orden):
        t = real[c] + disf[c]
        if t > 0:
            axes[1].text(t + 0.1, i, f"{t:.1f} %", va="center", fontsize=8)
    axes[1].set_xlabel("% de filas"); axes[1].legend(loc="lower right"); axes[1].set_title("% de vacíos por columna")
    return _leyenda(fig, "no basta con contar vacíos: mire si se concentran en ciertos años. "
                         "Y los rojos no aparecen en df.isna() — están escritos como texto.")


def distribucion_hora(df, col_hora="hora_num", col_y="con_victimas"):
    """Barras: cuántos incidentes hay a cada hora. Línea roja: qué % de ellos tiene víctimas.
    Las dos curvas cuentan historias distintas."""
    n = df[col_hora].value_counts().sort_index()
    p = df.groupby(col_hora)[col_y].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(n.index, n.values, color=AZUL, alpha=0.75, label="número de incidentes")
    ax.set_xlabel("hora del día"); ax.set_ylabel("incidentes"); ax.set_xticks(range(24))
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(p.index, p.values * 100, "o-", color=ROJO, label="% con víctimas")
    ax2.set_ylabel("% con víctimas", color=ROJO); ax2.set_ylim(50, 100)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title("¿Cuándo hay más incidentes? ¿Cuándo son más graves?")
    return _leyenda(fig, "las barras tienen picos en las horas pico (7 a.m. y 5 p.m.); la línea roja tiene su pico "
                         "de madrugada: hay menos incidentes, pero son más graves.")


def serie_mensual(df, col_fecha="fecha", col_y="con_victimas", marcas=None):
    """Incidentes por mes, apilados: con víctimas (rojo) y solo daños (azul). Las marcas señalan
    los quiebres: uno lo explica la realidad (todo cae), el otro el registro (solo cae un color)."""
    s = df.set_index(col_fecha)[col_y]
    vic = s.resample("MS").sum()
    sd = s.resample("MS").size() - vic
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.stackplot(vic.index, vic.values, sd.values, colors=[ROJO, AZUL], alpha=0.8,
                 labels=["con víctimas (heridos o muertos)", "solo daños"])
    ax.set_ylabel("incidentes por mes"); ax.legend(loc="upper right")
    ax.set_title("Incidentes por mes, según gravedad")
    for fecha, texto in (marcas or []):
        f = pd.Timestamp(fecha)
        ax.axvline(f, color=NEGRO, ls="--", lw=1)
        ax.annotate(texto, (f, ax.get_ylim()[1] * 0.97), xytext=(5, 0), textcoords="offset points",
                    color=NEGRO, fontsize=9, va="top")
    return _leyenda(fig, "en abril de 2020 caen los dos colores: cambió la ciudad (cuarentena). En octubre de 2022 "
                         "el rojo sigue igual y el azul desaparece: no cambió la ciudad, cambió el registro.")


def vacio_informa(df, col_y, grupos, col_anio="anio", grupo_anio=None):
    """Izquierda: % con víctimas en todo el dataset y en cada grupo de filas 'con vacío'
    (grupos = {nombre: máscara booleana}). Derecha (opcional): de qué años son las filas del
    grupo `grupo_anio`. Un vacío que 'predice' la y casi siempre está contando otra cosa."""
    base = df[col_y].mean() * 100
    vals = {"todos los incidentes": base} | {k: df.loc[m, col_y].mean() * 100 for k, m in grupos.items()}
    ncols = 2 if grupo_anio else 1
    fig, axes = plt.subplots(1, ncols, figsize=(13 if ncols == 2 else 8, 3.8), squeeze=False)
    ax = axes[0, 0]
    nombres = list(vals)[::-1]
    ax.barh(nombres, [vals[k] for k in nombres], color=[GRIS if k == "todos los incidentes" else ROJO for k in nombres])
    for i, k in enumerate(nombres):
        ax.text(vals[k] + 0.8, i, f"{vals[k]:.0f} %", va="center")
    ax.axvline(base, color=NEGRO, lw=1, ls=":"); ax.set_xlim(0, 105); ax.set_xlabel("% con víctimas")
    ax.set_title("¿Las filas con vacío se parecen al resto?")
    if grupo_anio:
        c = df.loc[grupos[grupo_anio], col_anio].value_counts().sort_index()
        axes[0, 1].bar(c.index.astype(str), c.values, color=ROJO)
        for x, v in zip(c.index.astype(str), c.values):
            axes[0, 1].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        axes[0, 1].set_title(f"¿De qué año son las filas «{grupo_anio}»?")
    return _leyenda(fig, "si las filas con vacío tienen una proporción muy distinta de la y, el vacío "
                         "'informa'. La pregunta siguiente es: ¿informa sobre el incidente, o sobre cuándo se registró?")


def one_hot_ejemplo(df, col, n=6, semilla=3):
    """Una columna de texto con k categorías se vuelve k columnas de 0/1. Dibuja las dos tablas."""
    d = (df[[col]].dropna().sample(frac=1, random_state=semilla)
           .groupby(col, group_keys=False).head(1).head(n).reset_index(drop=True))
    dum = pd.get_dummies(d[col], prefix=col, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * n + 1.0), gridspec_kw={"width_ratios": [1, 3.2]})
    for ax, tabla, titulo in ((axes[0], d, "antes: 1 columna de texto"),
                              (axes[1], dum, f"después: {dum.shape[1]} columnas de 0/1, una por categoría")):
        ax.axis("off"); ax.set_title(titulo)
        t = ax.table(cellText=tabla.values, colLabels=list(tabla.columns), loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
        for (r, cc), cell in t.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            if r == 0:
                cell.set_facecolor("#EAF1FA"); cell.set_text_props(fontweight="bold")
            elif tabla is dum and str(cell.get_text().get_text()) == "1":
                cell.set_facecolor("#F6D5D1")
    return _leyenda(fig, "cada fila tiene exactamente un 1: la categoría a la que pertenece. Ninguna categoría "
                         "queda 'mayor' que otra, que es justo lo que pasaría si las numeráramos 1, 2, 3…")


def escalado_antes_despues(X, columnas):
    """Las mismas variables en sus unidades originales y después de StandardScaler
    (restar la media y dividir por la desviación estándar)."""
    from sklearn.preprocessing import StandardScaler
    d = X[columnas].dropna()
    z = pd.DataFrame(StandardScaler().fit_transform(d), columns=columnas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].boxplot([d[c] for c in columnas], vert=False, showfliers=False)
    axes[0].set_yticks(range(1, len(columnas) + 1)); axes[0].set_yticklabels(columnas)
    axes[0].set_title("Antes: cada variable en su propia unidad")
    axes[0].set_xlabel("valor original (años, horas, grados…)")
    axes[1].boxplot([z[c] for c in columnas], vert=False, showfliers=False)
    axes[1].set_yticks(range(1, len(columnas) + 1)); axes[1].set_yticklabels(columnas)
    axes[1].axvline(0, color=ROJO, ls="--", lw=1)
    axes[1].set_title("Después de StandardScaler: todas en 'desviaciones estándar'")
    axes[1].set_xlabel("z = (x − media) / desviación")
    return _leyenda(fig, "a la izquierda `anio` (≈2 020) aplasta a las coordenadas (≈6 y ≈−75); a la derecha "
                         "todas quedan centradas en 0 con la misma dispersión. La forma de cada caja no cambia.")


def vs_linea_base(resultados, titulo="¿Cuánto le gana el modelo a la línea base?"):
    """resultados = {"nombre del escenario": (accuracy_linea_base, accuracy_modelo), ...}
    Barras agrupadas; la flecha es lo que de verdad aprendió el modelo."""
    nombres = list(resultados)
    base = [resultados[k][0] for k in nombres]; mod = [resultados[k][1] for k in nombres]
    x = np.arange(len(nombres)); w = 0.36
    fig, ax = plt.subplots(figsize=(4 + 2.6 * len(nombres), 4.2))
    ax.bar(x - w / 2, base, w, color=GRIS, label="línea base (siempre la clase mayoritaria)")
    ax.bar(x + w / 2, mod, w, color=MODULO[1], label="árbol de decisión")
    for i in range(len(nombres)):
        ax.text(x[i] - w / 2, base[i] + 0.005, f"{base[i]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, mod[i] + 0.005, f"{mod[i]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.annotate("", xy=(x[i] + w / 2 + 0.2, mod[i]), xytext=(x[i] + w / 2 + 0.2, base[i]),
                    arrowprops=dict(arrowstyle="->", color=ROJO, lw=2))
        ax.text(x[i] + w / 2 + 0.25, (base[i] + mod[i]) / 2, f"+{(mod[i] - base[i]) * 100:.1f}\npuntos",
                color=ROJO, va="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(nombres); ax.set_ylim(min(base) - 0.1, 1)
    ax.set_ylabel("accuracy en prueba"); ax.legend(loc="upper left", fontsize=8); ax.set_title(titulo)
    return _leyenda(fig, "no mire la altura de la barra azul sino la flecha roja: eso es lo que el modelo "
                         "aprendió por encima de adivinar siempre lo mismo.")


# ================================================================== S3 — entrenar, probar y no hacerse trampa
def split_esquema(n_train=0.8, titulo="Separar ANTES de entrenar"):
    """(A) Esquema de train/test: la tabla se parte en dos; el modelo solo ve la parte azul;
    la parte roja es el 'examen' que nunca vio."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 1.6), 9.6 * n_train, 0.9, boxstyle="round,pad=0.02", fc=AZUL, ec="white"))
    ax.add_patch(FancyBboxPatch((0.2 + 9.6 * n_train, 1.6), 9.6 * (1 - n_train), 0.9, boxstyle="round,pad=0.02", fc=ROJO, ec="white"))
    ax.text(0.2 + 4.8 * n_train, 2.05, f"ENTRENAMIENTO ({n_train:.0%})\nel modelo aprende aquí: .fit(X_train, y_train)",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(0.2 + 9.6 * n_train + 4.8 * (1 - n_train), 2.05, f"PRUEBA ({1 - n_train:.0%})\nel 'examen'",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(0.2 + 4.8 * n_train, 1.15, "Todo lo que el modelo 'aprende' —incluidas medias, tasas o escalas—\nse calcula SOLO con esta parte",
            ha="center", va="top", fontsize=9.5, color=AZUL)
    ax.text(0.2 + 9.6 * n_train + 4.8 * (1 - n_train), 1.15, "Se usa una sola vez,\nal final: .score(X_test, y_test)",
            ha="center", va="top", fontsize=9.5, color=ROJO)
    ax.set_title(titulo, fontsize=12)
    return _leyenda(fig, "si algo de la parte roja se filtra a la azul, el examen deja de medir y la nota sale inflada.")


def fuga_target_encoding(datos, col, col_y="con_victimas", max_puntos=4000, semilla=1):
    """(A) Por qué la 'tasa de víctimas por dirección' calculada con TODOS los datos hace trampa:
    en las direcciones con UNA sola fila, la tasa es exactamente la respuesta de esa fila."""
    n = datos[col].map(datos[col].value_counts())
    tasa = datos.groupby(col)[col_y].transform("mean")
    grupos = pd.cut(n, [0, 1, 2, 5, 20, np.inf], labels=["1 fila", "2", "3–5", "6–20", "más de 20"])
    acierto = ((tasa >= 0.5).astype(int) == datos[col_y]).groupby(grupos, observed=False).mean() * 100
    cuantos = grupos.value_counts().reindex(acierto.index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    rng = np.random.default_rng(semilla)
    uno = datos[n == 1]
    m = uno.sample(min(max_puntos, len(uno)), random_state=semilla)
    axes[0].scatter(m[col_y] + rng.normal(0, 0.06, len(m)), tasa.loc[m.index] + rng.normal(0, 0.02, len(m)),
                    s=6, alpha=0.3, c=np.where(m[col_y] == 1, ROJO, AZUL))
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["solo daños (y = 0)", "con víctimas (y = 1)"])
    axes[0].set_ylabel(f"tasa de víctimas de su {col}")
    axes[0].set_title(f"Las {len(uno):,} filas cuya {col} aparece UNA sola vez")
    axes[1].bar(acierto.index.astype(str), acierto.values, color=[ROJO] + [GRIS] * (len(acierto) - 1))
    for i, (a, c) in enumerate(zip(acierto.values, cuantos.values)):
        axes[1].text(i, a + 1, f"{a:.0f} %\n({c:,} filas)", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylim(0, 115); axes[1].set_xlabel(f"¿cuántas veces aparece la {col}?")
    axes[1].set_ylabel("% de filas donde 'tasa ≥ 0,5' acierta la y")
    axes[1].set_title("Entre menos se repite, más 'acierta' la tasa")
    return _leyenda(fig, "izquierda: cuando la dirección aparece una vez, su tasa ES la respuesta (0 o 1). "
                         "La variable no describe la calle: le copia la y al modelo.")


def fuga_resultados(resultados, linea_base=None):
    """(B) resultados = {"escenario": (acc_train, acc_test), ...}. Barras de entrenamiento y prueba
    por escenario; la línea punteada es la línea base."""
    nombres = list(resultados); x = np.arange(len(nombres)); w = 0.36
    tr = [resultados[k][0] for k in nombres]; te = [resultados[k][1] for k in nombres]
    fig, ax = plt.subplots(figsize=(3.2 + 2.9 * len(nombres), 4.4))
    ax.bar(x - w / 2, tr, w, color=AZUL, alpha=0.55, label="entrenamiento")
    ax.bar(x + w / 2, te, w, color=ROJO, label="prueba")
    for i in range(len(nombres)):
        ax.text(x[i] - w / 2, tr[i] + 0.004, f"{tr[i]:.3f}", ha="center", va="bottom", fontsize=9, color=GRIS)
        ax.text(x[i] + w / 2, te[i] + 0.004, f"{te[i]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    if linea_base is not None:
        ax.axhline(linea_base, color=NEGRO, ls=":", lw=1); ax.text(x[-1] + 0.6, linea_base, f"línea base {linea_base:.3f}", va="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(nombres); ax.set_ylim(min(te + [linea_base or 1]) - 0.06, max(tr + te) + 0.04)
    ax.set_ylabel("accuracy"); ax.legend(loc="upper left"); ax.set_title("La misma variable, calculada de dos formas")
    return _leyenda(fig, "con fuga, la prueba sube y se parece al entrenamiento: parece un gran hallazgo. Calculada bien, "
                         "la prueba BAJA y aparece la brecha con el entrenamiento: la variable solo servía para copiar.")


def camino_de_una_fila(modelo, fila, columnas, clases=("solo daños", "con víctimas"), max_pasos=8):
    """(D) El modelo entrenado como FUNCIÓN: una fila real entra, recorre las preguntas del árbol
    y sale con una predicción. Cada caja es una pregunta; en rojo la respuesta de esta fila."""
    X1 = pd.DataFrame([fila.values], columns=list(columnas))
    t = modelo.tree_
    nodos = modelo.decision_path(X1).indices[:max_pasos + 1]
    fig, ax = plt.subplots(figsize=(12, 1.1 * len(nodos) + 1.2))
    ax.set_xlim(0, 12); ax.set_ylim(-0.3, len(nodos) + 0.6); ax.axis("off")
    activos = [f"{c} = {v:g}" for c, v in zip(columnas, fila.values) if v != 0]
    ax.text(0.1, len(nodos) + 0.35, "Entra la fila:  " + " · ".join(activos[:8]), fontsize=9.5, color=NEGRO, va="center")
    for k, nodo in enumerate(nodos):
        y = len(nodos) - 0.6 - k
        if t.children_left[nodo] == -1:
            p = t.value[nodo][0] / t.value[nodo][0].sum()
            texto = f"HOJA → predice «{clases[int(np.argmax(p))]}»   ({p[1]:.0%} con víctimas entre los casos de entrenamiento que llegaron aquí)"
            ax.add_patch(FancyBboxPatch((0.6, y - 0.32), 10.8, 0.64, boxstyle="round,pad=0.02", fc=ROJO, ec=ROJO))
            ax.text(6, y, texto, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        else:
            col = columnas[t.feature[nodo]]; u = t.threshold[nodo]; v = float(fila.iloc[t.feature[nodo]])
            si = v <= u
            ax.add_patch(FancyBboxPatch((0.6, y - 0.32), 10.8, 0.64, boxstyle="round,pad=0.02", fc="white", ec=AZUL, lw=1.5))
            ax.text(1.0, y, f"¿{col} ≤ {u:.2f}?", va="center", fontsize=10, color=NEGRO)
            ax.text(11.0, y, f"esta fila: {v:g} → {'SÍ' if si else 'NO'}", va="center", ha="right", fontsize=10,
                    color=ROJO, fontweight="bold")
            ax.annotate("", xy=(6, y - 0.45), xytext=(6, y - 0.32), arrowprops=dict(arrowstyle="->", color=GRIS))
    return _leyenda(fig, "el árbol entrenado no es una tabla: es una función. Esta fila contestó estas preguntas, en este orden, "
                         "y salió por esta hoja. Así se 'explica' una predicción.")


# ================================================================== M2 · S4 — regresión lineal, descenso de gradiente, regularización
VERDE = MODULO[2]


def _mse(x, y, b0, b1):
    return np.mean((y - (b0 + b1 * x)) ** 2)


def recta_residuales(x, y, malas=((5, 0.0), None), xlabel="x", ylabel="y"):
    """(A) Dos rectas sobre los mismos puntos: una cualquiera y la de mínimos cuadrados. Los
    segmentos son los residuales; el título dice la suma de sus cuadrados (lo que se minimiza)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1, b0 = np.polyfit(x, y, 1)
    rectas = [(malas[0][0] if malas[0] else y.mean(), malas[0][1] if malas[0] else 0.0, "una recta cualquiera"),
              (b0, b1, "la recta de mínimos cuadrados")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    xx = np.linspace(x.min(), x.max(), 50)
    for ax, (c0, c1, t) in zip(axes, rectas):
        pred = c0 + c1 * x
        ax.vlines(x, np.minimum(y, pred), np.maximum(y, pred), color=ROJO, lw=0.8, alpha=0.7)
        ax.scatter(x, y, s=16, color=GRIS, zorder=3)
        ax.plot(xx, c0 + c1 * xx, color=VERDE, lw=2.5)
        ax.set_title(f"{t}\nsuma de residuales² = {np.sum((y - pred) ** 2):,.0f}")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    return _leyenda(fig, "cada línea roja es un error (residual). La regresión lineal elige la recta que hace mínima "
                         "la suma de esos errores AL CUADRADO.")


def superficie_perdida(x, y, b0_rango=None, b1_rango=None, marcar=None):
    """(A) El 'paisaje' de la pérdida: para cada par (b0, b1) el error cuadrático medio. Es un tazón
    con un único fondo: ahí está la mejor recta."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1_opt, b0_opt = np.polyfit(x, y, 1)
    r0 = b0_rango or (b0_opt - 3 * y.std(), b0_opt + 3 * y.std()); r1 = b1_rango or (b1_opt - 3 * y.std(), b1_opt + 3 * y.std())
    B0, B1 = np.meshgrid(np.linspace(*r0, 120), np.linspace(*r1, 120))
    L = np.mean((y[None, None, :] - (B0[..., None] + B1[..., None] * x[None, None, :])) ** 2, axis=2)
    fig = plt.figure(figsize=(13, 4.8))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(B0, B1, L, cmap="Greens_r", alpha=0.85, linewidth=0)
    ax3.set_xlabel("b0 (intercepto)"); ax3.set_ylabel("b1 (pendiente)"); ax3.set_zlabel("pérdida (MSE)")
    ax3.set_title("La pérdida como un tazón")
    ax = fig.add_subplot(1, 2, 2)
    cs = ax.contour(B0, B1, L, levels=18, cmap="Greens_r"); ax.clabel(cs, fontsize=7, fmt="%.0f")
    ax.plot(b0_opt, b1_opt, "*", color=ROJO, ms=16, label=f"mínimo: b0={b0_opt:.1f}, b1={b1_opt:.1f}")
    for (c0, c1, t) in (marcar or []):
        ax.plot(c0, c1, "o", color=AZUL); ax.annotate(t, (c0, c1), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("b0 (intercepto)"); ax.set_ylabel("b1 (pendiente)"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Vista desde arriba: curvas de nivel")
    return _leyenda(fig, "cada punto del plano es una recta posible; su altura es qué tan mal ajusta. Entrenar = encontrar el fondo.")


def _descenso(x, y, eta, pasos, inicio=(0.0, 0.0)):
    b0, b1 = inicio; tray = [(b0, b1, _mse(x, y, b0, b1))]
    for _ in range(pasos):
        r = (b0 + b1 * x) - y
        b0, b1 = b0 - eta * 2 * r.mean(), b1 - eta * 2 * (r * x).mean()
        tray.append((b0, b1, _mse(x, y, b0, b1)))
        if not np.isfinite(tray[-1][2]) or tray[-1][2] > 1e12:
            break
    return np.array(tray)


def descenso_gradiente_paso_a_paso(x, y, eta=0.1, mostrar=(0, 1, 3, 10, 30, 100), inicio=(0.0, 0.0), xlabel="x", ylabel="y"):
    """(A) Descenso de gradiente: arriba, la recta en distintos pasos sobre los datos; abajo, el
    camino sobre las curvas de nivel y la pérdida que baja paso a paso."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    tray = _descenso(x, y, eta, max(mostrar), inicio)
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, len(mostrar), height_ratios=[1, 1.15], hspace=0.45)
    xx = np.linspace(x.min(), x.max(), 30)
    for j, k in enumerate(mostrar):
        ax = fig.add_subplot(gs[0, j]); b0, b1, l = tray[min(k, len(tray) - 1)]
        ax.scatter(x, y, s=6, color=GRIS); ax.plot(xx, b0 + b1 * xx, color=VERDE, lw=2.2)
        ax.set_title(f"paso {k}\nMSE = {l:,.1f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    b1o, b0o = np.polyfit(x, y, 1)
    ax = fig.add_subplot(gs[1, : len(mostrar) // 2])
    B0, B1 = np.meshgrid(np.linspace(min(tray[:, 0].min(), b0o) - 2, max(tray[:, 0].max(), b0o) + 2, 100),
                         np.linspace(min(tray[:, 1].min(), b1o) - 2, max(tray[:, 1].max(), b1o) + 2, 100))
    L = np.mean((y[None, None, :] - (B0[..., None] + B1[..., None] * x[None, None, :])) ** 2, axis=2)
    ax.contour(B0, B1, L, levels=15, cmap="Greens_r", linewidths=0.8)
    ax.plot(tray[:, 0], tray[:, 1], "o-", color=ROJO, ms=3, lw=1); ax.plot(b0o, b1o, "*", color=NEGRO, ms=14)
    ax.set_xlabel("b0"); ax.set_ylabel("b1"); ax.set_title("El camino sobre el tazón (estrella = fondo)")
    ax = fig.add_subplot(gs[1, len(mostrar) // 2:])
    ax.plot(tray[:, 2], color=ROJO); ax.set_yscale("log"); ax.set_xlabel("paso"); ax.set_ylabel("MSE (escala log)")
    ax.set_title(f"La pérdida baja en cada paso (η = {eta})")
    return _leyenda(fig, "en cada paso se calcula hacia dónde sube el tazón (el gradiente) y se da un pasito en contra. "
                         "Arriba se ve el efecto: la recta se acomoda sola a los datos.")


def learning_rate_efecto(x, y, etas=(0.005, 0.1, 0.9, 1.05), pasos=40, inicio=(0.0, 0.0)):
    """(C) La perilla del optimizador: la tasa de aprendizaje η. Muy pequeña no llega; adecuada
    llega; grande zigzaguea; demasiado grande diverge."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1o, b0o = np.polyfit(x, y, 1)
    fig, axes = plt.subplots(1, len(etas), figsize=(3.6 * len(etas), 4))
    for ax, eta in zip(axes, etas):
        tray = _descenso(x, y, eta, pasos, inicio)
        span0 = max(abs(b0o) * 1.6, 3); span1 = max(abs(b1o) * 1.6, 3)
        B0, B1 = np.meshgrid(np.linspace(b0o - span0, b0o + span0, 90), np.linspace(b1o - span1, b1o + span1, 90))
        L = np.mean((y[None, None, :] - (B0[..., None] + B1[..., None] * x[None, None, :])) ** 2, axis=2)
        ax.contour(B0, B1, L, levels=14, cmap="Greens_r", linewidths=0.7)
        t = tray[(np.abs(tray[:, 0] - b0o) < span0 * 1.5) & (np.abs(tray[:, 1] - b1o) < span1 * 1.5)]
        ax.plot(t[:, 0], t[:, 1], "o-", color=ROJO, ms=3, lw=1); ax.plot(b0o, b1o, "*", color=NEGRO, ms=12)
        fin = tray[-1, 2]
        estado = "diverge ✗" if (not np.isfinite(fin) or fin > tray[0, 2]) else ("llega ✓" if fin < _mse(x, y, b0o, b1o) * 1.005 else "no llega aún")
        ax.set_title(f"η = {eta}  →  {estado}"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(b0o - span0, b0o + span0); ax.set_ylim(b1o - span1, b1o + span1)
    return _leyenda(fig, f"mismos datos, mismo punto de partida, {pasos} pasos. Solo cambia el tamaño del paso: "
                         "es la perilla del OPTIMIZADOR, no del modelo.")


def recta_por_grupo(df, x, y, grupo, modelo, columnas, etiquetas=None):
    """(B) El modelo lineal entrenado, dibujado sobre los datos: una recta por grupo (las demás
    variables en su valor típico). Rectas paralelas = el grupo suma una constante."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colores = [AZUL, ROJO, VERDE, MODULO[3]]
    base = pd.DataFrame([{c: (df[c].median() if c in df and pd.api.types.is_numeric_dtype(df[c]) else 0) for c in columnas}])
    xx = np.linspace(df[x].min(), df[x].max(), 50)
    for i, (g, sub) in enumerate(df.groupby(grupo)):
        ax.scatter(sub[x], sub[y], s=10, alpha=0.45, color=colores[i % 4], label=(etiquetas or {}).get(g, g))
        malla = pd.concat([base] * len(xx), ignore_index=True); malla[x] = xx
        for cc in [c for c in columnas if c.startswith(f"{grupo}_")]:
            malla[cc] = int(cc == f"{grupo}_{g}")
        ax.plot(xx, modelo.predict(malla[columnas]), color=colores[i % 4], lw=2.5)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(title=grupo); ax.set_title("Lo que aprendió el modelo, sobre los datos reales")
    return _leyenda(fig, "las dos rectas son paralelas: el modelo lineal dice que ser fumador SUMA una cantidad fija, "
                         "a cualquier edad. La distancia entre rectas es el coeficiente.")


def contribuciones_una_fila(modelo, fila, columnas, real=None, unidad="", top=8):
    """(D) Una fila entra, una predicción sale: intercepto + Σ coeficiente × valor, como una
    cascada. Se muestran los términos que más aportan."""
    contrib = pd.Series(modelo.coef_ * fila.values.astype(float), index=list(columnas))
    contrib = contrib[contrib.abs() > 1e-9]
    principales = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
    resto = contrib.sum() - principales.sum()
    pasos = [("intercepto", modelo.intercept_)] + [(f"{k} = {fila[k]:g}", v) for k, v in principales.items()]
    if abs(resto) > 1e-9:
        pasos.append(("resto de variables", resto))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(pasos) + 1.8))
    acum = 0
    for i, (n, v) in enumerate(pasos):
        ax.barh(i, v, left=acum if i else 0, color=GRIS if i == 0 else (VERDE if v >= 0 else ROJO))
        ax.text((acum + v if i else v) + (0.01 * abs(modelo.intercept_) + 1e-9), i, f"{v:+,.0f}{unidad}", va="center", fontsize=8.5)
        acum = acum + v if i else v
    ax.barh(len(pasos), acum, color=AZUL); ax.text(acum, len(pasos), f"  predicción = {acum:,.0f}{unidad}", va="center", fontweight="bold")
    ylabels = [n for n, _ in pasos] + ["PREDICCIÓN"]
    if real is not None:
        ax.axvline(real, color=NEGRO, ls="--", lw=1); ax.text(real, -0.8, f"real: {real:,.0f}{unidad}", ha="center", fontsize=8.5)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_title("Una fila entra, una predicción sale: la suma de las contribuciones")
    return _leyenda(fig, "cada barra es coeficiente × valor de esa variable para esta persona. Verde suma, rojo resta. "
                         "Un modelo lineal se explica sumando.")


def regularizacion_alpha(X_train, y_train, X_test, y_test, alphas=None, destacar=6):
    """(C) Ridge y Lasso al subir alpha: cómo se encogen los coeficientes (variables escaladas) y
    qué pasa con el R² de prueba. Lasso apaga variables; Ridge solo las encoge."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_train); A, B = sc.transform(X_train), sc.transform(X_test)
    alphas = alphas if alphas is not None else np.logspace(-2, 1.3, 14)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (nombre, M, esc) in zip(axes[:2], [("Ridge", Ridge, 100000), ("Lasso", Lasso, 1)]):
        coefs, r2 = [], []
        for a in alphas:
            m = M(alpha=a * esc, max_iter=20000).fit(A, y_train) if nombre == "Ridge" else M(alpha=a, max_iter=20000).fit(A, y_train)
            coefs.append(m.coef_); r2.append(m.score(B, y_test))
        C = np.array(coefs)
        top = np.argsort(-np.abs(C[0]))[:destacar]
        for j in range(C.shape[1]):
            ax.plot(alphas * esc if nombre == "Ridge" else alphas, C[:, j], color=GRIS if j not in top else None,
                    lw=0.6 if j not in top else 2, alpha=0.5 if j not in top else 1,
                    label=X_train.columns[j][:28] if j in top else None)
        ax.set_xscale("log"); ax.axhline(0, color=NEGRO, lw=0.6)
        ax.set_xlabel("alpha (fuerza de la penalización)"); ax.set_ylabel("coeficiente (variables escaladas)")
        vivos = int((np.abs(C[-1]) > 1e-6).sum())
        ax.set_title(f"{nombre}: con alpha máximo quedan {vivos} de {C.shape[1]} ≠ 0")
        if nombre == "Lasso":
            ax.legend(fontsize=7, loc="upper right")
        axes[2].plot(alphas * esc if nombre == "Ridge" else alphas, r2, "o-", label=nombre, ms=3,
                     color=AZUL if nombre == "Ridge" else ROJO)
    axes[2].set_xscale("log"); axes[2].set_xlabel("alpha"); axes[2].set_ylabel("R² en prueba"); axes[2].legend()
    axes[2].set_title("¿Cuánto se pierde al simplificar?")
    return _leyenda(fig, "al subir alpha los coeficientes se encogen hacia 0. Lasso los APAGA uno a uno (selección de "
                         "variables); Ridge los achica sin apagarlos. A la derecha, el precio en R².")


def pred_vs_real(paneles, unidad=""):
    """paneles = {"título": (y_real, y_pred)}. Diagonal = predicción perfecta."""
    from sklearn.metrics import r2_score, mean_absolute_error
    fig, axes = plt.subplots(1, len(paneles), figsize=(5.2 * len(paneles), 4.6), squeeze=False)
    for ax, (t, (yr, yp)) in zip(axes[0], paneles.items()):
        yr, yp = np.asarray(yr), np.asarray(yp)
        idx = np.random.default_rng(0).choice(len(yr), min(4000, len(yr)), replace=False)
        ax.scatter(yr[idx], yp[idx], s=5, alpha=0.3, color=VERDE)
        lo, hi = min(yr.min(), yp.min()), max(yr.max(), yp.max()); ax.plot([lo, hi], [lo, hi], color=NEGRO, lw=1, ls="--")
        ax.set_title(f"{t}\nR² = {r2_score(yr, yp):.3f} · MAE = {mean_absolute_error(yr, yp):.1f}{unidad}")
        ax.set_xlabel("valor real"); ax.set_ylabel("predicción")
    return _leyenda(fig, "si los puntos caen sobre la diagonal, el modelo 'acierta' todo. Antes de celebrar, "
                         "pregunte de dónde salió esa precisión.")


# ================================================================== M2 · S5 — clasificación: logística y k-NN
def _sigmoide(z):
    return 1 / (1 + np.exp(-z))


def _sinteticos_2c(n=240, semilla=4):
    """Dos clases en 2 variables escaladas, con traslape (para logística y k-NN)."""
    rng = np.random.default_rng(semilla)
    n1 = n // 3
    a = rng.normal([-0.6, -0.4], 0.9, (n - n1, 2)); b = rng.normal([1.0, 0.9], 0.8, (n1, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)
    return pd.DataFrame(X, columns=["x1", "x2"]), y


def sigmoide_vs_recta(n=120, semilla=2):
    """(A) Una y de sí/no contra una variable: la recta se sale de [0, 1]; la sigmoide no."""
    rng = np.random.default_rng(semilla)
    x = rng.uniform(-4, 4, n); y = (rng.uniform(size=n) < _sigmoide(1.6 * x)).astype(int)
    b1, b0 = np.polyfit(x, y, 1)
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression().fit(x.reshape(-1, 1), y)
    xx = np.linspace(-5, 5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, t in zip(axes, ["Con una recta (regresión lineal)", "Con una sigmoide (regresión logística)"]):
        ax.scatter(x, y + rng.normal(0, 0.02, n), s=14, c=np.where(y == 1, ROJO, AZUL), alpha=0.7)
        ax.axhspan(0, 1, color="#F2F2F2", zorder=0); ax.axhline(0.5, color=GRIS, ls=":", lw=1)
        ax.set_title(t); ax.set_xlabel("x (una variable escalada)"); ax.set_ylim(-0.45, 1.45)
    axes[0].plot(xx, b0 + b1 * xx, color=VERDE, lw=2.5)
    axes[0].fill_between(xx, b0 + b1 * xx, 1, where=(b0 + b1 * xx) > 1, color=ROJO, alpha=0.2)
    axes[0].fill_between(xx, b0 + b1 * xx, 0, where=(b0 + b1 * xx) < 0, color=ROJO, alpha=0.2)
    axes[0].text(3.2, 1.3, "¿probabilidad > 1?", color=ROJO, ha="center"); axes[0].text(-3.2, -0.35, "¿probabilidad < 0?", color=ROJO, ha="center")
    axes[1].plot(xx, lg.predict_proba(xx.reshape(-1, 1))[:, 1], color=VERDE, lw=2.5)
    axes[0].set_ylabel("y  (1 = sí, 0 = no)")
    return _leyenda(fig, "la recta promete probabilidades imposibles en los extremos. La sigmoide 'dobla' la recta para que "
                         "siempre quede entre 0 y 1, y cruza 0,5 donde el modelo cambia de opinión.")


def frontera_logistica_2d(n=240):
    """(A) En dos variables, la logística traza una frontera RECTA y pinta la probabilidad a cada lado."""
    from sklearn.linear_model import LogisticRegression
    X, y = _sinteticos_2c(n)
    lg = LogisticRegression().fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-3.5, 3.5, 200))
    P = lg.predict_proba(pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()}))[:, 1].reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), cmap=CMAP_PROB, alpha=0.75)
    ax.contour(xx, yy, P, levels=[0.5], colors=NEGRO, linewidths=2)
    ax.scatter(X["x1"], X["x2"], c=np.where(y == 1, ROJO, AZUL), s=18, edgecolor="white", lw=0.4)
    fig.colorbar(im, ax=ax, label="probabilidad de 'sí'"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("Regresión logística: frontera recta (línea negra = 50 %)")
    return _leyenda(fig, "la línea negra es donde p = 0,5. Lejos de ella el modelo está seguro; cerca, duda. La frontera "
                         "de una logística siempre es recta (en las variables que le damos).")


def matrices_confusion(paneles, etiquetas=("no (0)", "sí (1)")):
    """paneles = {"título": (y_real, y_pred)}. Matriz de confusión con nombres en cada celda y
    las dos métricas que cuentan historias distintas: accuracy y recall."""
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(paneles), figsize=(5.3 * len(paneles), 4.6), squeeze=False)
    nombres = [["verdaderos\nnegativos", "falsos\npositivos"], ["falsos\nnegativos", "verdaderos\npositivos"]]
    for ax, (t, (yr, yp)) in zip(axes[0], paneles.items()):
        M = confusion_matrix(yr, yp, labels=[0, 1])
        ax.imshow([[0, 1], [1, 0]], cmap=ListedColormap(["#EAF4EE", "#F8E1DE"]))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}\n{nombres[i][j]}", ha="center", va="center", fontsize=11,
                        fontweight="bold" if (i, j) == (1, 1) else "normal")
        acc = (M[0, 0] + M[1, 1]) / M.sum(); rec = M[1, 1] / max(M[1].sum(), 1)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"predice {e}" for e in etiquetas])
        ax.set_yticks([0, 1]); ax.set_yticklabels([f"real {e}" for e in etiquetas])
        ax.set_title(f"{t}\naccuracy = {acc:.3f} · recall = {rec:.3f}")
        for s in ax.spines.values():
            s.set_visible(False)
    return _leyenda(fig, "accuracy cuenta los aciertos de las dos filas juntas; recall mira solo la fila de abajo: "
                         "de los que de verdad dijeron sí, ¿a cuántos encontró el modelo?")


def umbral_efecto(y_real, proba, umbrales=None, marcar=(0.5,)):
    """(C) Mover el umbral de decisión: accuracy, recall y precisión para cada umbral."""
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    umbrales = umbrales if umbrales is not None else np.linspace(0.05, 0.8, 31)
    acc, rec, pre = [], [], []
    for t in umbrales:
        p = (proba >= t).astype(int)
        acc.append(accuracy_score(y_real, p)); rec.append(recall_score(y_real, p)); pre.append(precision_score(y_real, p, zero_division=0))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(umbrales, acc, color=NEGRO, lw=2, label="accuracy")
    ax.plot(umbrales, rec, color=ROJO, lw=2.5, label="recall (de los 'sí', cuántos encuentro)")
    ax.plot(umbrales, pre, color=AZUL, lw=2, label="precisión (de los que llamo, cuántos dicen sí)")
    for m in marcar:
        ax.axvline(m, color=GRIS, ls="--", lw=1); ax.text(m, 1.02, f"umbral {m}", ha="center", fontsize=8, color=GRIS)
    ax.set_xlabel("umbral: llamo si la probabilidad de 'sí' es ≥ umbral"); ax.set_ylabel("valor"); ax.set_ylim(0, 1.08)
    ax.legend(loc="center right"); ax.set_title("La misma probabilidad, distintas decisiones")
    return _leyenda(fig, "con el umbral de 0,5 por defecto el recall es bajísimo. Bajar el umbral encuentra más clientes "
                         "que dicen sí a cambio de más llamadas en vano: la accuracy casi no se mueve, el negocio sí.")


def logistica_una_fila(modelo, fila_z, columnas, real=None, top=7):
    """(D) Una fila: contribuciones en la escala de los log-odds (z), y z → probabilidad con la sigmoide."""
    c = pd.Series(modelo.coef_[0] * np.asarray(fila_z, float), index=list(columnas))
    principales = c.reindex(c.abs().sort_values(ascending=False).index).head(top)
    resto = c.sum() - principales.sum(); z = modelo.intercept_[0] + c.sum(); p = _sigmoide(z)
    pasos = [("intercepto", modelo.intercept_[0])] + list(principales.items()) + [("resto de variables", resto)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(pasos) + 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]; acum = 0
    for i, (n, v) in enumerate(pasos):
        ax.barh(i, v, left=acum, color=GRIS if i == 0 else (ROJO if v >= 0 else AZUL)); acum += v
        ax.text(acum, i, f" {v:+.2f}", va="center", fontsize=8)
    ax.barh(len(pasos), z, color=VERDE); ax.text(z, len(pasos), f"  z = {z:.2f}", va="center", fontweight="bold")
    ax.set_yticks(range(len(pasos) + 1)); ax.set_yticklabels([n[:34] for n, _ in pasos] + ["SUMA (z)"]); ax.invert_yaxis()
    ax.axvline(0, color=NEGRO, lw=0.6); ax.set_title("1. Se suman las contribuciones (log-odds)")
    zz = np.linspace(-6, 6, 200); ax = axes[1]
    ax.plot(zz, _sigmoide(zz), color=VERDE, lw=2.5); ax.axhline(0.5, color=GRIS, ls=":", lw=1)
    ax.plot([z, z], [0, p], color=ROJO, ls="--"); ax.plot([zz[0], z], [p, p], color=ROJO, ls="--"); ax.plot(z, p, "o", color=ROJO, ms=9)
    ax.set_title(f"2. La sigmoide convierte z en probabilidad: {p:.0%}" + (f"\n(lo que pasó: {'sí' if real == 1 else 'no'})" if real is not None else ""))
    ax.set_xlabel("z"); ax.set_ylabel("probabilidad de 'sí'")
    return _leyenda(fig, "la logística es una regresión lineal 'doblada': primero suma como en la Sesión 4, después pasa la "
                         "suma por la sigmoide. Rojo empuja hacia 'sí', azul hacia 'no'.")


def knn_vecinos(k=7, n=120, punto=(0.6, 0.2)):
    """(A) k-NN: un caso nuevo mira a sus k vecinos más cercanos y vota."""
    X, y = _sinteticos_2c(n, semilla=11)
    d = np.sqrt(((X.values - np.array(punto)) ** 2).sum(axis=1)); idx = np.argsort(d)[:k]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.scatter(X["x1"], X["x2"], c=np.where(y == 1, ROJO, AZUL), s=26, alpha=0.35, edgecolor="white")
    ax.scatter(X.iloc[idx]["x1"], X.iloc[idx]["x2"], c=np.where(y[idx] == 1, ROJO, AZUL), s=70, edgecolor=NEGRO, lw=1.2, zorder=3)
    for i in idx:
        ax.plot([punto[0], X.iloc[i]["x1"]], [punto[1], X.iloc[i]["x2"]], color=GRIS, lw=0.8)
    ax.add_patch(plt.Circle(punto, d[idx[-1]], fill=False, ls="--", color=NEGRO))
    ax.plot(*punto, "*", ms=22, color=VERDE, markeredgecolor=NEGRO, zorder=4)
    votos = int(y[idx].sum())
    ax.set_title(f"k = {k}: {votos} vecinos dicen 'sí' y {k - votos} dicen 'no' → predice «{'sí' if votos > k / 2 else 'no'}»"
                 f"  (probabilidad {votos / k:.0%})"); ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    return _leyenda(fig, "k-NN no aprende ninguna fórmula: guarda los datos y, para un caso nuevo, pregunta a los k más "
                         "parecidos. 'Parecido' = cerca en distancia euclidiana.")


def knn_k_efecto(ks=(1, 5, 25, 101), n=300):
    """(C) La perilla de k-NN: k pequeño = frontera nerviosa (memoriza); k grande = frontera suave."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    X, y = _sinteticos_2c(n, semilla=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 150), np.linspace(-3.5, 3.5, 150))
    malla = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
    fig, axes = plt.subplots(1, len(ks), figsize=(3.7 * len(ks), 4))
    for ax, k in zip(axes, ks):
        m = KNeighborsClassifier(k).fit(Xtr, ytr)
        ax.contourf(xx, yy, m.predict_proba(malla)[:, 1].reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap=CMAP_PROB, alpha=0.7)
        ax.scatter(Xtr["x1"], Xtr["x2"], c=np.where(ytr == 1, ROJO, AZUL), s=10, edgecolor="white", lw=0.3)
        ax.set_title(f"k = {k}\ntrain {m.score(Xtr, ytr):.2f} · prueba {m.score(Xte, yte):.2f}"); ax.set_xticks([]); ax.set_yticks([])
    return _leyenda(fig, "con k = 1 cada punto de entrenamiento se 'defiende' solo (train perfecto: sobreajuste, como el árbol "
                         "sin límite). Con k grande la frontera se suaviza (y si k se acerca al total de datos, todo se vuelve la clase mayoritaria). k es una perilla del MODELO.")


def knn_escala(df, col_x, col_y, k=15, i_punto=0, semilla=0, n=1500):
    """La misma búsqueda de vecinos con las variables en sus unidades y escaladas. Si una variable
    tiene una escala enorme, 'cercano' solo significa 'cercano en esa variable'."""
    d = df[[col_x, col_y]].dropna().sample(min(n, len(df)), random_state=semilla).reset_index(drop=True)
    Z = (d - d.mean()) / d.std()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (datos, t) in zip(axes, [(d, "Sin escalar"), (Z, "Escalado (z)")]):
        dist = np.sqrt(((datos.values - datos.values[i_punto]) ** 2).sum(axis=1)); idx = np.argsort(dist)[1:k + 1]
        ax.scatter(d[col_x], d[col_y], s=6, color=GRIS, alpha=0.4)
        ax.scatter(d.iloc[idx][col_x], d.iloc[idx][col_y], s=40, color=ROJO, edgecolor=NEGRO, lw=0.5)
        ax.plot(d.iloc[i_punto][col_x], d.iloc[i_punto][col_y], "*", ms=20, color=VERDE, markeredgecolor=NEGRO)
        ax.set_xlabel(col_x); ax.set_ylabel(col_y); ax.set_title(f"{t}: los {k} vecinos del cliente ★")
        ax.set_ylim(d[col_y].quantile(0.01), d[col_y].quantile(0.99))
    return _leyenda(fig, f"sin escalar, {col_y} (miles) aplasta a {col_x} (decenas): los 'vecinos' tienen cualquier edad. "
                         "Escalado, los vecinos se parecen en las dos cosas.")


# ================================================================== M2 · S6 — árboles, bosques, F1 y ROC
def arbol_vs_bosque(n=300, n_arboles=(1, 2, 3), semilla=3):
    """(A) Un árbol profundo es nervioso; cada árbol del bosque ve una muestra distinta (bootstrap)
    y se equivoca distinto; el promedio de muchos es suave."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    X, y = _sinteticos_2c(n, semilla=semilla)
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 150), np.linspace(-3.5, 3.5, 150))
    malla = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
    rf = RandomForestClassifier(200, random_state=0, max_features=1).fit(X, y)
    paneles = [("un árbol sin límite", DecisionTreeClassifier(random_state=0).fit(X, y))]
    paneles += [(f"árbol {k} del bosque\n(muestra bootstrap)", rf.estimators_[k - 1]) for k in n_arboles]
    paneles += [("el bosque: promedio\nde 200 árboles", rf)]
    fig, axes = plt.subplots(1, len(paneles), figsize=(3.4 * len(paneles), 3.9))
    for ax, (t, m) in zip(axes, paneles):
        P = m.predict_proba(malla.values if m is not rf and m is not paneles[0][1] else malla)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), cmap=CMAP_PROB, alpha=0.75)
        ax.scatter(X["x1"], X["x2"], c=np.where(y == 1, ROJO, AZUL), s=7, edgecolor="white", lw=0.2)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    return _leyenda(fig, "cada árbol solo memoriza su propia muestra y se equivoca a su manera; al promediarlos, los "
                         "errores se cancelan y queda una frontera suave. Eso es un bosque aleatorio.")


def complejidad_auc(X_tr, y_tr, X_te, y_te, profundidades=(2, 4, 6, 8, 12, 16, None), n_arboles=150):
    """(C) AUC de entrenamiento y prueba de un árbol y de un bosque, para cada max_depth."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    et = [str(d) if d else "sin límite" for d in profundidades]
    r = {"árbol train": [], "árbol prueba": [], "bosque train": [], "bosque prueba": []}
    for d in profundidades:
        for nombre, m in (("árbol", DecisionTreeClassifier(max_depth=d, random_state=42)),
                          ("bosque", RandomForestClassifier(n_arboles, max_depth=d, random_state=42, n_jobs=-1))):
            m.fit(X_tr, y_tr)
            r[f"{nombre} train"].append(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1]))
            r[f"{nombre} prueba"].append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    fig, ax = plt.subplots(figsize=(11, 4.5)); xs = np.arange(len(profundidades))
    ax.plot(xs, r["árbol train"], "o--", color=AZUL, alpha=0.6, label="un árbol · entrenamiento")
    ax.plot(xs, r["árbol prueba"], "o-", color=AZUL, lw=2.5, label="un árbol · prueba")
    ax.plot(xs, r["bosque train"], "s--", color=VERDE, alpha=0.6, label="bosque · entrenamiento")
    ax.plot(xs, r["bosque prueba"], "s-", color=VERDE, lw=2.5, label="bosque · prueba")
    ax.set_xticks(xs); ax.set_xticklabels(et); ax.set_xlabel("max_depth"); ax.set_ylabel("AUC")
    ax.legend(loc="lower left"); ax.set_title("Más profundo: el árbol se desploma en prueba; el bosque no")
    return _leyenda(fig, "las líneas punteadas (entrenamiento) suben siempre. Lo que importa son las sólidas: el árbol "
                         "solo empeora al crecer; el bosque aguanta porque promedia árboles que se equivocan distinto.")


def roc_explicada(y_real, proba, umbrales=(0.7, 0.5, 0.3, 0.2, 0.1)):
    """La curva ROC sale de mover el umbral (Sesión 5): cada umbral es un punto
    (falsos positivos, verdaderos positivos)."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(y_real, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr, tpr, color=VERDE, alpha=0.12)
    ax.plot(fpr, tpr, color=VERDE, lw=2.5, label=f"modelo (AUC = {roc_auc_score(y_real, proba):.3f})")
    ax.plot([0, 1], [0, 1], color=GRIS, ls="--", label="azar (AUC = 0,5)")
    for u in umbrales:
        p = proba >= u; tp = (p & (np.asarray(y_real) == 1)).sum() / (np.asarray(y_real) == 1).sum()
        fp = (p & (np.asarray(y_real) == 0)).sum() / (np.asarray(y_real) == 0).sum()
        ax.plot(fp, tp, "o", color=ROJO, ms=8); ax.annotate(f"umbral {u}", (fp, tp), xytext=(8, -12), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("tasa de falsos positivos (de los 'no', a cuántos llamo en vano)")
    ax.set_ylabel("tasa de verdaderos positivos = recall"); ax.legend(loc="lower right"); ax.set_aspect("equal")
    ax.set_title("La curva ROC: todos los umbrales a la vez")
    return _leyenda(fig, "cada punto rojo es un umbral de la Sesión 5. El área bajo la curva (AUC) resume todos los umbrales: "
                         "es la probabilidad de que el modelo ponga a un 'sí' por encima de un 'no'.")


def curvas_roc(modelos, y_real):
    """modelos = {"nombre": probabilidades}. Varias curvas ROC en el mismo plano."""
    from sklearn.metrics import roc_curve, roc_auc_score
    colores = [GRIS, AZUL, VERDE, ROJO, MODULO[3], MODULO[4]]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (n, p), c in zip(modelos.items(), colores):
        f, t, _ = roc_curve(y_real, p); ax.plot(f, t, color=c, lw=2.2, label=f"{n} · AUC {roc_auc_score(y_real, p):.3f}")
    ax.plot([0, 1], [0, 1], color=GRIS, ls=":", lw=1)
    ax.set_xlabel("tasa de falsos positivos"); ax.set_ylabel("recall"); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_aspect("equal"); ax.set_title("¿Qué modelo ordena mejor a los clientes?")
    return _leyenda(fig, "la curva más arriba y a la izquierda gana en TODOS los umbrales. El AUC permite comparar modelos "
                         "sin haber elegido todavía el umbral.")


def importancias(modelo, columnas, top=12, titulo="¿Qué variables usa el bosque?"):
    s = pd.Series(modelo.feature_importances_, index=list(columnas)).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(9, 0.38 * top + 1.2))
    ax.barh(s.index, s.values, color=VERDE); ax.set_xlabel("importancia (reducción de Gini, suma 1)"); ax.set_title(titulo)
    return _leyenda(fig, "importancia = cuánto ayudó cada variable a purificar los cortes, sumado en todos los árboles. "
                         "Dice qué USA el modelo, no qué CAUSA el resultado.")


def bosque_una_fila(bosque, fila, real=None):
    """(D) Una fila entra al bosque: cada árbol da su probabilidad; el bosque las promedia."""
    votos = np.array([t.predict_proba(np.asarray(fila, float).reshape(1, -1))[0, 1] for t in bosque.estimators_])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.hist(votos, bins=np.linspace(0, 1, 21), color=VERDE, edgecolor="white")
    ax.axvline(votos.mean(), color=ROJO, lw=2.5); ax.axvline(0.5, color=GRIS, ls="--")
    ax.text(votos.mean(), ax.get_ylim()[1] * 0.92, f"  promedio = {votos.mean():.2f}", color=ROJO, fontweight="bold")
    ax.set_xlabel("probabilidad de 'sí' que da cada árbol"); ax.set_ylabel("número de árboles")
    ax.set_title(f"Los {len(votos)} árboles opinan sobre el mismo cliente" + (f" (lo que pasó: {'sí' if real == 1 else 'no'})" if real is not None else ""))
    return _leyenda(fig, "ningún árbol decide solo: el bosque promedia sus opiniones. Si los árboles están muy divididos, "
                         "la predicción es poco segura aunque el promedio cruce el umbral.")


def distribucion_por_clase(df, col, col_y, etiquetas=("no", "sí"), recorte=0.98, unidad=""):
    """Histograma de una variable separada por la clase de la y (para ver si 'predice demasiado')."""
    lim = df[col].quantile(recorte)
    fig, ax = plt.subplots(figsize=(11, 4))
    for v, c, e in ((0, AZUL, etiquetas[0]), (1, ROJO, etiquetas[1])):
        s = df.loc[df[col_y] == v, col]
        ax.hist(s.clip(upper=lim), bins=50, alpha=0.55, color=c, density=True, label=f"{e} · mediana {s.median():.0f}{unidad}")
    ax.set_xlabel(col); ax.set_ylabel("densidad"); ax.legend(); ax.set_title(f"{col} según la respuesta")
    return _leyenda(fig, "las dos distribuciones se separan mucho más que con cualquier otra variable (la última barra junta todo lo que pasa del percentil 98). "
                         "Antes de usarla, la pregunta de la Sesión 3: ¿la tendría antes de conocer la y?")


# ================================================================== M3 · S7 — K-means y DBSCAN
NARANJA = MODULO[3]
COLORES_CLUSTER = [AZUL, ROJO, "#2C8C5B", NARANJA, "#7A4DB8", "#8C6D46", "#D45E9E", "#4BA3A8"]


def _col_cluster(etiquetas):
    """Color por cluster; el ruido de DBSCAN (-1) va en gris claro."""
    etiquetas = np.asarray(etiquetas)
    return np.array([("#C8C8C8" if e < 0 else COLORES_CLUSTER[int(e) % len(COLORES_CLUSTER)]) for e in etiquetas])


def _xy(X):
    X = pd.DataFrame(X)
    return X.iloc[:, 0].values, X.iloc[:, 1].values, list(X.columns[:2])


def puntos_sin_color(X, titulo="¿Cuántos grupos ve?"):
    """Los datos SIN etiqueta: la pregunta del aprendizaje no supervisado."""
    x, y, (cx, cy) = _xy(X)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(x, y, s=18, color=GRIS, alpha=0.75, edgecolor="white", lw=0.3)
    ax.set_xlabel(cx); ax.set_ylabel(cy); ax.set_title(titulo)
    return _leyenda(fig, "no hay columna y: nadie nos dice a qué grupo pertenece cada punto. Su ojo ya está "
                         "haciendo clustering; el algoritmo tiene que hacerlo con una regla explícita.")


def _lloyd(X, k, semilla, pasos):
    """K-means escrito a mano (algoritmo de Lloyd) guardando cada estado intermedio."""
    rng = np.random.default_rng(semilla)
    C = X[rng.choice(len(X), k, replace=False)].astype(float)
    estados = []
    for _ in range(pasos):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        inercia = d[np.arange(len(X)), lab].sum()
        nuevos = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
        estados.append((C.copy(), lab, inercia, nuevos))
        if np.allclose(nuevos, C):
            break
        C = nuevos
    return estados


def kmeans_iteraciones(X, k=4, semilla=3, mostrar=(0, 1, 2, -1)):
    """(A) K-means paso a paso: centroides al azar → cada punto al más cercano → cada centroide
    al promedio de sus puntos → repetir hasta que nada cambie."""
    Xv = np.asarray(pd.DataFrame(X).iloc[:, :2], float)
    _, _, (cx, cy) = _xy(X)
    est = _lloyd(Xv, k, semilla, 50)
    idx = [i if i >= 0 else len(est) + i for i in mostrar]
    fig, axes = plt.subplots(1, len(idx), figsize=(3.9 * len(idx), 4.1), sharex=True, sharey=True)
    for ax, i in zip(axes, idx):
        C, lab, iner, nuevos = est[i]
        ax.scatter(Xv[:, 0], Xv[:, 1], c=_col_cluster(lab), s=12, alpha=0.75, edgecolor="white", lw=0.2)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=230, c=COLORES_CLUSTER[:k], edgecolor=NEGRO, lw=1.6, zorder=4)
        if i < len(est) - 1:
            for a, b in zip(C, nuevos):
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=NEGRO, lw=1.6))
        etapa = "inicio: centroides al azar" if i == 0 else ("final: ya no se mueven" if i == len(est) - 1 else f"iteración {i}")
        ax.set_title(f"{etapa}\ninercia = {iner:,.0f}", fontsize=10)
        ax.set_xlabel(cx)
    axes[0].set_ylabel(cy)
    plt.tight_layout()
    return _leyenda(fig, "dos pasos que se repiten: (1) cada punto toma el color de su centroide ✖ más cercano; "
                         "(2) cada centroide se muda (flecha) al promedio de sus puntos. La inercia solo puede bajar.")


def kmeans_resultado(X, modelo, titulo=None):
    """(B) Lo que aprendió K-means: k centroides y una región (celda) por centroide."""
    Xd = pd.DataFrame(X)
    x, y, (cx, cy) = _xy(Xd)
    xx, yy = np.meshgrid(np.linspace(x.min() - 0.5, x.max() + 0.5, 300), np.linspace(y.min() - 0.5, y.max() + 0.5, 300))
    malla = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=Xd.columns[:2])
    z = modelo.predict(malla).reshape(xx.shape)
    k = modelo.n_clusters
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.contourf(xx, yy, z, levels=np.arange(-0.5, k), colors=COLORES_CLUSTER[:k], alpha=0.13)
    ax.scatter(x, y, c=_col_cluster(modelo.labels_), s=16, edgecolor="white", lw=0.3)
    C = modelo.cluster_centers_
    ax.scatter(C[:, 0], C[:, 1], marker="X", s=260, c=COLORES_CLUSTER[:k], edgecolor=NEGRO, lw=1.6, zorder=4)
    ax.set_xlabel(cx); ax.set_ylabel(cy)
    ax.set_title(titulo or f"K-means con k = {k} · inercia {modelo.inertia_:,.0f}")
    return _leyenda(fig, "cada región de color es 'todo lo que queda más cerca de ese centroide que de los demás'. "
                         "Las fronteras son rectas: K-means siempre parte el plano en polígonos.")


def kmeans_k_efecto(X, ks=(2, 3, 4, 6), semilla=42):
    """(C) La perilla de K-means: k. El algoritmo entrega exactamente los grupos que se le piden."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    x, y, (cx, cy) = _xy(X)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.8 * len(ks), 3.9), sharex=True, sharey=True)
    for ax, k in zip(axes, ks):
        m = KMeans(k, n_init=10, random_state=semilla).fit(X)
        ax.scatter(x, y, c=_col_cluster(m.labels_), s=10, edgecolor="white", lw=0.2)
        ax.scatter(*m.cluster_centers_[:, :2].T, marker="X", s=150, c=COLORES_CLUSTER[:k], edgecolor=NEGRO, lw=1.2)
        ax.set_title(f"k = {k}\ninercia {m.inertia_:,.0f} · silueta {silhouette_score(X, m.labels_):.2f}", fontsize=10)
        ax.set_xlabel(cx)
    axes[0].set_ylabel(cy)
    plt.tight_layout()
    return _leyenda(fig, "K-means nunca dice 'no hay grupos' ni 'son 4': entrega exactamente k. Con k de más parte grupos "
                         "reales en pedazos; con k de menos junta grupos distintos.")


def codo_silueta(X, ks=range(1, 10), semilla=42, marcar=None):
    """Dos ayudas para elegir k: la inercia (baja siempre) y la silueta (más alta = grupos mejor separados)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    ks = list(ks); iner, sil = [], []
    for k in ks:
        m = KMeans(k, n_init=10, random_state=semilla).fit(X)
        iner.append(m.inertia_); sil.append(silhouette_score(X, m.labels_) if k > 1 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    axes[0].plot(ks, iner, "o-", color=NARANJA, lw=2); axes[0].set_xlabel("k"); axes[0].set_ylabel("inercia")
    axes[0].set_title("El 'codo': la inercia SIEMPRE baja al subir k")
    axes[1].plot(ks, sil, "o-", color=NEGRO, lw=2); axes[1].set_xlabel("k"); axes[1].set_ylabel("silueta promedio")
    axes[1].set_title("La silueta: ¿qué tan bien separados quedan?")
    if marcar:
        for ax in axes:
            ax.axvline(marcar, color=ROJO, ls="--", lw=1.2)
    plt.tight_layout()
    return _leyenda(fig, "no se elige el k de menor inercia (sería k = n, un grupo por punto). Se busca el codo, donde "
                         "agregar un grupo deja de ayudar mucho, y se mira la silueta. Son pistas, no veredictos.")


def kmeans_semillas(X, k=4, semillas=(0, 1, 2, 3)):
    """Con una sola inicialización al azar, K-means puede quedarse en un mínimo local distinto según la semilla."""
    from sklearn.cluster import KMeans
    x, y, (cx, cy) = _xy(X)
    fig, axes = plt.subplots(1, len(semillas), figsize=(3.8 * len(semillas), 3.9), sharex=True, sharey=True)
    for ax, s in zip(axes, semillas):
        m = KMeans(k, init="random", n_init=1, random_state=s).fit(X)
        ax.scatter(x, y, c=_col_cluster(m.labels_), s=10, edgecolor="white", lw=0.2)
        ax.scatter(*m.cluster_centers_[:, :2].T, marker="X", s=150, c=COLORES_CLUSTER[:k], edgecolor=NEGRO, lw=1.2)
        ax.set_title(f"semilla {s} · n_init=1\ninercia {m.inertia_:,.0f}", fontsize=10); ax.set_xlabel(cx)
    axes[0].set_ylabel(cy)
    plt.tight_layout()
    return _leyenda(fig, "mismos datos, mismo k, distinto punto de partida: a veces termina en una solución peor (inercia "
                         "más alta). Por eso n_init=10 corre 10 arranques y se queda con el de menor inercia.")


def dbscan_idea(X, eps=0.5, min_samples=5, ejemplos=3, semilla=0):
    """(A) DBSCAN: un punto es NÚCLEO si tiene al menos min_samples vecinos a distancia ≤ eps; los
    núcleos que se tocan forman un cluster; lo que no alcanza ningún núcleo es RUIDO."""
    from sklearn.cluster import DBSCAN
    Xv = np.asarray(pd.DataFrame(X).iloc[:, :2], float)
    _, _, (cx, cy) = _xy(X)
    m = DBSCAN(eps=eps, min_samples=min_samples).fit(Xv)
    nucleo = np.zeros(len(Xv), bool); nucleo[m.core_sample_indices_] = True
    ruido = m.labels_ == -1; borde = ~nucleo & ~ruido
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    col = _col_cluster(m.labels_)
    ax.scatter(Xv[nucleo, 0], Xv[nucleo, 1], c=col[nucleo], s=34, edgecolor="white", lw=0.4, label="núcleo")
    ax.scatter(Xv[borde, 0], Xv[borde, 1], facecolor="white", edgecolor=col[borde], s=34, lw=1.5, label="borde")
    ax.scatter(Xv[ruido, 0], Xv[ruido, 1], marker="x", color=NEGRO, s=34, lw=1.3, label="ruido")
    rng = np.random.default_rng(semilla)
    for tipo, mask in (("núcleo", nucleo), ("borde", borde), ("ruido", ruido)):
        if mask.any():
            i = rng.choice(np.where(mask)[0])
            ax.add_patch(plt.Circle(Xv[i], eps, fill=False, ls="--", color=NEGRO, lw=1.2))
            n = int((np.sqrt(((Xv - Xv[i]) ** 2).sum(1)) <= eps).sum())
            ax.annotate(f"{tipo}: {n} vecinos en ε", Xv[i], xytext=(12, 12), textcoords="offset points", fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", ec=GRIS))
    ax.set_aspect("equal"); ax.set_xlabel(cx); ax.set_ylabel(cy); ax.legend(loc="lower right")
    k = len(set(m.labels_)) - (1 if ruido.any() else 0)
    ax.set_title(f"DBSCAN (ε = {eps}, min_samples = {min_samples}): {k} clusters y {ruido.sum()} puntos de ruido")
    return _leyenda(fig, "DBSCAN no recibe k: busca zonas densas. Círculo de radio ε alrededor de cada punto; si adentro hay "
                         "al menos min_samples puntos, es núcleo. Los núcleos encadenados son un cluster; lo suelto es ruido.")


def dbscan_eps(X, epsilons=(0.1, 0.2, 0.3, 0.6), min_samples=5):
    """(C) La perilla de DBSCAN: ε. Muy pequeño = todo es ruido; muy grande = todo es un solo grupo."""
    from sklearn.cluster import DBSCAN
    x, y, (cx, cy) = _xy(X)
    fig, axes = plt.subplots(1, len(epsilons), figsize=(3.8 * len(epsilons), 3.9), sharex=True, sharey=True)
    for ax, e in zip(axes, epsilons):
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(X)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        ax.scatter(x, y, c=_col_cluster(lab), s=10, edgecolor="white", lw=0.2)
        ax.set_title(f"ε = {e}\n{k} clusters · {np.sum(lab == -1)} de ruido", fontsize=10); ax.set_xlabel(cx)
    axes[0].set_ylabel(cy)
    plt.tight_layout()
    return _leyenda(fig, "ε es 'qué tan cerca es cerca'. Muy pequeño: casi nadie tiene vecinos y todo es ruido (gris). "
                         "Muy grande: todo se encadena en un solo grupo. Y depende de la escala de las variables.")


def kmeans_vs_dbscan(X, k=2, eps=0.2, min_samples=5):
    """Formas que K-means no puede ver (lunas): K-means corta con rectas; DBSCAN sigue la densidad."""
    from sklearn.cluster import KMeans, DBSCAN
    x, y, (cx, cy) = _xy(X)
    km = KMeans(k, n_init=10, random_state=0).fit(X); db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    axes[0].scatter(x, y, c=_col_cluster(km.labels_), s=14, edgecolor="white", lw=0.3)
    axes[0].scatter(*km.cluster_centers_[:, :2].T, marker="X", s=220, c=COLORES_CLUSTER[:k], edgecolor=NEGRO, lw=1.5)
    axes[0].set_title(f"K-means (k = {k}): parte con una recta")
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    axes[1].scatter(x, y, c=_col_cluster(db.labels_), s=14, edgecolor="white", lw=0.3)
    axes[1].set_title(f"DBSCAN (ε = {eps}): {n_db} grupos que siguen la forma")
    for ax in axes:
        ax.set_xlabel(cx)
    axes[0].set_ylabel(cy)
    return _leyenda(fig, "K-means supone grupos 'redondos' alrededor de un centro, así que corta las lunas por la mitad. "
                         "DBSCAN no supone forma: encadena puntos densos y sigue la curva.")


def tabla_cruzada(real, clusters, titulo="¿Qué especie cayó en cada cluster?", etiqueta_real="especie", leyenda=None):
    """Mapa de calor de la tabla cruzada etiqueta real × cluster (la etiqueta NO se usó para agrupar)."""
    t = pd.crosstab(pd.Series(real, name=etiqueta_real), pd.Series(clusters, name="cluster"))
    fig, ax = plt.subplots(figsize=(1.15 * t.shape[1] + 2.4, 0.5 * t.shape[0] + 1.3))
    ax.imshow(t.values, cmap=LinearSegmentedColormap.from_list("n", ["white", NARANJA]), aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, t.values[i, j], ha="center", va="center", fontsize=12,
                    fontweight="bold" if t.values[i, j] == t.values[:, j].max() and t.values[i, j] > 0 else "normal")
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels([f"cluster {c}" for c in t.columns])
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(t.index)
    for s in ax.spines.values():
        s.set_visible(False)
    pureza = t.max(axis=0).sum() / t.values.sum()
    ax.set_title(f"{titulo}\npureza = {pureza:.0%}", fontsize=11)
    return _leyenda(fig, leyenda or ("las especies NO se usaron para agrupar: se miran después, para ver si los grupos que encontró "
                                     "el algoritmo coinciden con algo real. Pureza = % de puntos que están en el cluster de su mayoría."))


def clusters_por_unidad(df, col_x, col_y, lab_crudo, lab_escalado):
    """💥 El mismo K-means sin escalar y escalado, dibujado en dos variables: sin escalar, los grupos
    son franjas de la variable con números grandes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, lab, t in ((axes[0], lab_crudo, "Sin escalar"), (axes[1], lab_escalado, "Escalado (z)")):
        ax.scatter(df[col_x], df[col_y], c=_col_cluster(lab), s=22, edgecolor="white", lw=0.3)
        ax.set_xlabel(col_x); ax.set_title(f"{t}: K-means con k = {len(set(lab))}")
    axes[0].set_ylabel(col_y)
    return _leyenda(fig, f"sin escalar, los clusters son franjas verticales de {col_x}: K-means mide distancias y una "
                         "diferencia de 500 g pesa muchísimo más que una de 5 mm. Escalado, cada variable pesa parecido.")


def kmeans_una_fila(paneles, columnas):
    """(D) Una fila entra a K-means: distancia² a cada centroide, partida por variable; gana el más corto.
    paneles = [(titulo, centros, fila), ...] — p. ej. el mismo pingüino sin escalar y escalado."""
    colores = [AZUL, ROJO, "#2C8C5B", NARANJA, "#7A4DB8", GRIS]
    n = len(paneles); k = len(paneles[0][1])
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 0.65 * k + 2.0))
    axes = np.atleast_1d(axes)
    for ax, (titulo, centros, fila) in zip(axes, paneles):
        centros = np.asarray(centros, float); fila = np.asarray(fila, float)
        aportes = (centros - fila) ** 2; total = aportes.sum(axis=1); gana = int(total.argmin())
        izq = np.zeros(len(centros))
        for v in range(len(columnas)):
            ax.barh(range(len(centros)), aportes[:, v], left=izq, color=colores[v % len(colores)], label=columnas[v], edgecolor="white")
            izq += aportes[:, v]
        fmt = (lambda t: f"{t:,.0f}") if total.max() > 1000 else (lambda t: f"{t:.2f}")
        for j, t in enumerate(total):
            ax.text(t, j, "  " + fmt(t) + ("  ← gana" if j == gana else ""), va="center", fontweight="bold" if j == gana else "normal", fontsize=9)
        ax.set_yticks(range(len(centros))); ax.set_yticklabels([f"cluster {j}" for j in range(len(centros))]); ax.invert_yaxis()
        ax.set_xlim(0, total.max() * 1.45); ax.set_title(titulo); ax.set_xlabel("distancia² al centroide")
        ax.ticklabel_format(axis="x", style="plain") if total.max() > 1000 else None
    axes[-1].legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    return _leyenda(fig, "K-means asigna midiendo la distancia a cada centroide y escogiendo la más corta. Los colores "
                         "muestran cuánto aporta cada variable: si una sola llena la barra, ella sola decide el grupo.")


# ================================================================== M3 · S8 — PCA y t-SNE
def todas_las_parejas(df, columnas, color_col=None):
    """Las 6 parejas posibles de 4 variables: la forma 'manual' de mirar un dataset de 4 dimensiones."""
    from itertools import combinations
    pares = list(combinations(columnas, 2))
    fig, axes = plt.subplots(2, (len(pares) + 1) // 2, figsize=(4.2 * ((len(pares) + 1) // 2), 7))
    cats = sorted(df[color_col].unique()) if color_col else [None]
    for ax, (a, b) in zip(axes.ravel(), pares):
        for i, c in enumerate(cats):
            s = df if c is None else df[df[color_col] == c]
            ax.scatter(s[a], s[b], s=9, alpha=0.7, color=COLORES_CLUSTER[i], label=c)
        ax.set_xlabel(a, fontsize=9); ax.set_ylabel(b, fontsize=9)
    if color_col:
        axes.ravel()[0].legend(fontsize=8)
    plt.tight_layout()
    return _leyenda(fig, f"con {len(columnas)} variables hay {len(pares)} parejas; con 15 serían 105 y con 64, 2 016. "
                         "Mirar de a dos no escala: necesitamos resumir muchas variables en pocas.")


def pca_proyeccion(n=200, semilla=0):
    """(A) PCA en 2D: buscar la dirección en la que los puntos se esparcen más y proyectar sobre ella."""
    rng = np.random.default_rng(semilla)
    X = rng.multivariate_normal([0, 0], [[3.0, 2.2], [2.2, 2.4]], n)
    X = X - X.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(X.T)); w1 = vecs[:, -1]; w2 = vecs[:, 0]
    malo = np.array([np.cos(np.radians(150)), np.sin(np.radians(150))])
    var_tot = X.var(axis=0, ddof=1).sum()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    lim = 6
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, color=GRIS, alpha=0.7)
    w1 = w1 if w1[0] > 0 else -w1; w2 = w2 if w2[1] > 0 else -w2
    for w, c, t in ((w1, NARANJA, "PC1"), (w2, NEGRO, "PC2")):
        L = max(1.6, 2.2 * np.sqrt(np.var(X @ w)))
        ax.annotate("", xy=w * L, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
        ax.text(*(w * L * 1.12), t, color=c, fontweight="bold", ha="center")
    ax.set_title("Los datos y sus dos ejes naturales")
    for ax, w, c, t in ((axes[1], w1, NARANJA, "Proyectar sobre PC1"), (axes[2], malo, ROJO, "Proyectar sobre otra dirección")):
        p = X @ w; P = np.outer(p, w)
        ax.plot([-lim * w[0], lim * w[0]], [-lim * w[1], lim * w[1]], color=c, lw=2)
        for a, b in zip(X[::3], P[::3]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GRIS, lw=0.5, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], s=8, color=GRIS, alpha=0.5); ax.scatter(P[:, 0], P[:, 1], s=10, color=c)
        ax.set_title(f"{t}\nconserva {p.var(ddof=1) / var_tot:.0%} de la varianza")
    for ax in axes[:3]:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax = axes[3]
    ax.scatter(X @ w1, rng.uniform(-0.15, 0.15, n), s=10, color=NARANJA, alpha=0.7)
    ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xlabel("PC1 (una sola columna)")
    ax.set_title("Resultado: 2 columnas → 1"); ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return _leyenda(fig, "PCA busca la dirección en la que los puntos quedan más esparcidos (PC1) y proyecta sobre ella: "
                         "las sombras sobre la línea naranja conservan casi toda la información; sobre la roja, se amontonan.")


def pca_varianza(pca, marcar=0.9, max_comp=None, titulo=""):
    """(B) Varianza explicada por cada componente y acumulada. ¿Cuántas columnas necesito?"""
    r = pca.explained_variance_ratio_[: max_comp or len(pca.explained_variance_ratio_)]
    acum = np.cumsum(r); xs = np.arange(1, len(r) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xs, r, color=NARANJA, alpha=0.85, label="cada componente")
    ax.plot(xs, acum, "o-", color=NEGRO, ms=4 if len(r) > 12 else 6, label="acumulada")
    if marcar:
        k = int(np.argmax(acum >= marcar) + 1) if (acum >= marcar).any() else None
        ax.axhline(marcar, color=ROJO, ls="--", lw=1)
        if k:
            ax.axvline(k, color=ROJO, ls=":", lw=1); ax.text(k, marcar - 0.1, f"  {k} componentes → {marcar:.0%}", color=ROJO)
    if len(r) <= 12:
        for x, v, a in zip(xs, r, acum):
            ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xlabel("componente principal"); ax.set_ylabel("fracción de la varianza"); ax.set_ylim(0, 1.05)
    ax.set_xticks(xs if len(r) <= 20 else xs[::4]); ax.legend(loc="center right")
    ax.set_title(titulo or "¿Cuánta información guarda cada componente?")
    return _leyenda(fig, "las barras dicen cuánto de la variación total recoge cada componente; la línea, cuánto llevamos "
                         "acumulado. Si los primeros dos acumulan mucho, el dibujo en 2D es fiel.")


def biplot(scores, componentes, columnas, etiquetas=None, var_exp=None, escala=None, top=None):
    """(B) Biplot: cada punto es una fila proyectada en PC1–PC2; cada flecha, cómo pesa una variable original."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if etiquetas is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=14, color=GRIS, alpha=0.6)
    else:
        for i, c in enumerate(pd.Series(etiquetas).unique()):
            m = np.asarray(etiquetas) == c
            ax.scatter(scores[m, 0], scores[m, 1], s=16, alpha=0.65, color=COLORES_CLUSTER[i], label=c)
        ax.legend(loc="lower right", fontsize=9)
    W = np.asarray(componentes)[:2].T
    escala = escala or 0.75 * min(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max()) / np.abs(W).max()
    cols = list(columnas)
    if top:
        keep = np.argsort(-(W ** 2).sum(axis=1))[:top]; W = W[keep]; cols = [cols[i] for i in keep]
    for (a, b), n in zip(W, cols):
        ax.annotate("", xy=(a * escala, b * escala), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=NEGRO, lw=1.8))
        ax.text(a * escala * 1.12, b * escala * 1.12, n, fontsize=9.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.axhline(0, color=GRIS, lw=0.5); ax.axvline(0, color=GRIS, lw=0.5)
    v = var_exp if var_exp is not None else [None, None]
    ax.set_xlabel("PC1" + (f" ({v[0]:.0%} de la varianza)" if v[0] is not None else ""))
    ax.set_ylabel("PC2" + (f" ({v[1]:.0%})" if v[1] is not None else ""))
    ax.set_title("Biplot: filas (puntos) y variables (flechas) en el mismo plano")
    return _leyenda(fig, "una flecha larga hacia la derecha = esa variable empuja a PC1. Flechas en la misma dirección = "
                         "variables correlacionadas. Un punto lejos en la dirección de una flecha tiene mucho de esa variable.")


def pca_una_fila(fila_z, componente, columnas, nombre="PC1", real=None):
    """(D) Una fila entra a PCA: su valor en el componente = suma de (valor z × peso) de cada variable."""
    aportes = np.asarray(fila_z, float) * np.asarray(componente, float)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(columnas) + 1.6))
    cols = [ROJO if a > 0 else AZUL for a in aportes]
    ax.barh(range(len(columnas)), aportes, color=cols)
    for i, (z, w, a) in enumerate(zip(fila_z, componente, aportes)):
        ax.text(a, i, f"  z={z:+.2f} × peso {w:+.2f} = {a:+.2f}  ", va="center", ha="left" if a >= 0 else "right", fontsize=9)
    ax.set_yticks(range(len(columnas))); ax.set_yticklabels(columnas); ax.invert_yaxis()
    ax.axvline(0, color=NEGRO, lw=0.8); lim = max(1.0, np.abs(aportes).max() * 2.2); ax.set_xlim(-lim, lim)
    ax.set_title(f"{nombre} de esta fila = {aportes.sum():+.2f}" + (f"   ({real})" if real else ""))
    ax.set_xlabel(f"aporte a {nombre}")
    return _leyenda(fig, "PCA no es una caja negra: cada componente es una suma ponderada de las variables (escaladas), "
                         "igual que la regresión de la Sesión 4, pero sin y: los pesos se eligen para esparcir los puntos.")


def digitos_muestra(X, y, n=24, semilla=0):
    """Una muestra de las imágenes de 8×8 píxeles: cada una es una fila con 64 columnas."""
    rng = np.random.default_rng(semilla); idx = rng.choice(len(X), n, replace=False)
    cols = 12; filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(1.05 * cols, 1.25 * filas))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(np.asarray(X)[i].reshape(8, 8), cmap="Greys"); ax.set_title(str(y[i]), fontsize=10); ax.axis("off")
    return _leyenda(fig, "cada imagen es una fila de la tabla: 64 columnas, una por píxel (0 = blanco, 16 = negro). "
                         "Un dataset de 64 dimensiones que el ojo entiende al instante.")


def _dispersion_etiquetas(ax, T, y, titulo, s=7):
    y = np.asarray(y)
    cmap = plt.get_cmap("tab10")
    ax.scatter(T[:, 0], T[:, 1], c=[cmap(int(v) % 10) for v in y], s=s, alpha=0.75)
    for v in np.unique(y):
        c = np.median(T[y == v], axis=0)
        ax.text(*c, str(v), fontsize=13, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=cmap(int(v) % 10), alpha=0.85))
    ax.set_title(titulo); ax.set_xticks([]); ax.set_yticks([])


def pca_vs_tsne(X, y, T_pca, T_tsne):
    """PCA (lineal, conserva la varianza global) contra t-SNE (no lineal, conserva vecindades)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    _dispersion_etiquetas(axes[0], T_pca, y, "PCA: las 2 direcciones de más varianza")
    _dispersion_etiquetas(axes[1], T_tsne, y, "t-SNE: cada punto cerca de sus vecinos")
    return _leyenda(fig, "las etiquetas (0–9) NO se usaron: solo colorean. PCA amontona varios dígitos en el centro; t-SNE "
                         "separa casi todos. Pero el t-SNE no es un mapa: sus distancias y tamaños no se pueden leer.")


def tsne_perplexity(X, y, perplexities=(2, 5, 30, 100), semilla=0):
    """(C) La perilla de t-SNE: perplexity (cuántos vecinos 'mira' cada punto). Cambia el dibujo entero."""
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, len(perplexities), figsize=(4.2 * len(perplexities), 4.4))
    for ax, p in zip(axes, perplexities):
        T = TSNE(2, perplexity=p, random_state=semilla, init="pca").fit_transform(X)
        _dispersion_etiquetas(ax, T, y, f"perplexity = {p}", s=4)
    plt.tight_layout()
    return _leyenda(fig, "mismos datos, mismo algoritmo: con perplexity baja aparecen islotes y astillas que no existen; "
                         "con alta, los grupos se acercan. No hay un dibujo 'verdadero': hay que mirar varios.")


def tsne_semillas(X, y, semillas=(0, 1, 2, 3), perplexity=30):
    """t-SNE con distintas semillas: los grupos se mueven, rotan y cambian de vecino."""
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, len(semillas), figsize=(4.2 * len(semillas), 4.4))
    for ax, s in zip(axes, semillas):
        T = TSNE(2, perplexity=perplexity, random_state=s, init="random").fit_transform(X)
        _dispersion_etiquetas(ax, T, y, f"semilla {s}", s=4)
    plt.tight_layout()
    return _leyenda(fig, "en cada semilla el mismo dígito aparece en otro lugar y con otros vecinos. Qué grupos existen "
                         "se repite; dónde quedan y qué tan lejos están unos de otros, no.")


def tsne_ruido(n=500, perplexities=(2, 5, 50), semilla=0):
    """💥 t-SNE sobre ruido puro: puntos al azar uniformes en un cuadrado (NO hay grupos)."""
    from sklearn.manifold import TSNE
    R = np.random.default_rng(semilla).uniform(size=(n, 2))
    paneles = [("los datos: puntos al azar en un cuadrado", R)]
    paneles += [(f"t-SNE · perplexity {p}", TSNE(2, perplexity=p, random_state=semilla).fit_transform(R)) for p in perplexities]
    fig, axes = plt.subplots(1, len(paneles), figsize=(4.3 * len(paneles), 4.4))
    for ax, (t, T) in zip(axes, paneles):
        ax.scatter(T[:, 0], T[:, 1], s=8, color=NARANJA, alpha=0.75); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return _leyenda(fig, f"{n} puntos al azar, sin ningún grupo. Con perplexity baja, t-SNE dibuja grumos y astillas muy "
                         "convincentes. Un dibujo de t-SNE con 'grupos' no prueba que existan grupos.")


def tsne_tamanos(semilla=0, perplexity=30):
    """💥 Tamaños y distancias en t-SNE no significan nada: tres grupos en 10 dimensiones, uno 10 veces
    más apretado y otro muy lejos. PCA los muestra como son; t-SNE los iguala."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(semilla); e = np.eye(10)[0]
    G = [rng.normal(0, 1, (150, 10)), rng.normal(0, 0.1, (150, 10)) + 8 * e, rng.normal(0, 1, (150, 10)) + 60 * e]
    X = np.vstack(G); lab = np.repeat([0, 1, 2], 150)
    nombres = ["A: disperso", "B: 10× más apretado, cerca de A", "C: disperso, muy lejos"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (t, T) in zip(axes, [("PCA: así son los datos", PCA(2).fit_transform(X)),
                                 (f"t-SNE (perplexity {perplexity})", TSNE(2, perplexity=perplexity, random_state=semilla).fit_transform(X))]):
        for k in range(3):
            ax.scatter(T[lab == k, 0], T[lab == k, 1], s=8, color=COLORES_CLUSTER[k], label=nombres[k], alpha=0.75)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8.5, loc="upper center")
    return _leyenda(fig, "en los datos, B es diminuto y está pegado a A, y C está lejísimos. En el t-SNE los tres tienen "
                         "el mismo tamaño y quedan más o menos igual de separados. Tamaño y distancia entre grupos: no se leen.")


# ================================================================== M3 · S9 — segmentación de municipios
CIUDADES = {"BOGOTÁ, D.C.": "Bogotá", "MEDELLÍN": "Medellín", "CALI": "Cali", "BARRANQUILLA": "Barranquilla",
            "QUIBDÓ": "Quibdó", "MITÚ": "Mitú", "RIOHACHA": "Riohacha", "PASTO": "Pasto"}


def mapa_municipios(df, valor=None, etiquetas=None, titulo="", col_lat="latitud", col_lon="longitud", nombres=None):
    """Mapa de puntos: cada municipio en la coordenada de su cabecera, coloreado por un valor continuo
    (p. ej. el IPM) o por la etiqueta de su cluster."""
    d = df.dropna(subset=[col_lat, col_lon])
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    if etiquetas is not None:
        lab = np.asarray(etiquetas)[df[col_lat].notna().values]
        for k in sorted(set(lab)):
            m = lab == k
            n = nombres[k] if nombres else f"cluster {k}"
            ax.scatter(d[col_lon][m], d[col_lat][m], s=13, color=COLORES_CLUSTER[int(k) % 8], label=f"{n} ({m.sum()})", edgecolor="white", lw=0.2)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    else:
        sc = ax.scatter(d[col_lon], d[col_lat], c=d[valor], s=13, cmap=LinearSegmentedColormap.from_list("ipm", ["#F6E7C8", NARANJA, "#7A2E0E"]),
                        edgecolor="white", lw=0.2)
        plt.colorbar(sc, ax=ax, shrink=0.6, label=valor)
    if "municipio" in d:
        for k, v in CIUDADES.items():
            f = d[d["municipio"] == k]
            if len(f):
                ax.annotate(v, (f[col_lon].iloc[0], f[col_lat].iloc[0]), xytext=(5, 3), textcoords="offset points", fontsize=8.5,
                            fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_xlim(-79.6, -66.6); ax.set_ylim(-4.6, 12.9)
    ax.text(-79.4, 12.6, "San Andrés y Providencia:\nfuera del recuadro", fontsize=7.5, color=GRIS, va="top")
    ax.set_aspect("equal"); ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
    ax.set_title(titulo or (f"{valor} por municipio" if valor else "Clusters en el mapa"))
    return _leyenda(fig, "cada punto es la cabecera de un municipio. El algoritmo NO vio las coordenadas: si los colores "
                         "forman regiones, es una pista de que los grupos significan algo.")


def perfil_clusters(df, etiquetas, columnas, nombres=None, titulo="Perfil de cada cluster"):
    """(B) Perfil: promedio de cada variable por cluster. Color = qué tan arriba o abajo del promedio
    nacional está (en desviaciones estándar); número = el valor real."""
    lab = np.asarray(etiquetas); ks = sorted(set(lab))
    med = pd.DataFrame({k: df.loc[lab == k, columnas].mean() for k in ks})
    z = (med.sub(df[columnas].mean(), axis=0)).div(df[columnas].std(), axis=0)
    fig, ax = plt.subplots(figsize=(1.5 * len(ks) + 4.2, 0.42 * len(columnas) + 1.6))
    ax.imshow(z.values, cmap=LinearSegmentedColormap.from_list("z", [AZUL, "#F4F4F4", ROJO]), vmin=-2, vmax=2, aspect="auto")
    for i in range(len(columnas)):
        for j in range(len(ks)):
            ax.text(j, i, f"{med.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
    n = np.bincount(lab)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([(nombres[k] if nombres else f"cluster {k}") + f"\n({n[k]})" for k in ks], fontsize=9)
    ax.set_yticks(range(len(columnas))); ax.set_yticklabels(columnas, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo)
    return _leyenda(fig, "rojo = peor que el promedio de los municipios; azul = mejor. El número es el promedio del cluster (en %). "
                         "Un perfil es un promedio: dentro de cada grupo hay municipios muy distintos.")


def estabilidad_k(Z, ks=range(2, 8), repeticiones=20, fraccion=0.8, semilla=0):
    """¿Los grupos sobreviven si cambio un poco los datos? Para cada k, se re-entrena K-means con el 80 %
    de las filas y se compara con la solución completa (índice de Rand ajustado: 1 = idénticos, 0 = azar)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    rng = np.random.default_rng(semilla); Z = np.asarray(Z)
    res = []
    for k in ks:
        base = KMeans(k, n_init=10, random_state=42).fit(Z)
        a = []
        for r in range(repeticiones):
            idx = rng.choice(len(Z), int(fraccion * len(Z)), replace=False)
            a.append(adjusted_rand_score(base.labels_, KMeans(k, n_init=10, random_state=r).fit(Z[idx]).predict(Z)))
        res.append((k, np.mean(a), np.min(a), silhouette_score(Z, base.labels_)))
    r = pd.DataFrame(res, columns=["k", "estabilidad", "peor", "silueta"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r["k"], r["estabilidad"], "o-", color=NARANJA, lw=2.5, label="estabilidad (promedio de 20 re-muestreos)")
    ax.fill_between(r["k"], r["peor"], r["estabilidad"], color=NARANJA, alpha=0.15, label="peor re-muestreo")
    ax.plot(r["k"], r["silueta"], "s--", color=NEGRO, label="silueta")
    ax.set_xlabel("k"); ax.set_ylim(0, 1.05); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("¿Qué k da grupos que sobreviven a cambiar los datos?")
    return _leyenda(fig, "estabilidad cerca de 1 = los mismos municipios quedan juntos aunque se quite el 20 % de los datos. "
                         "Una silueta baja con estabilidad alta = grupos reales pero sin fronteras nítidas (un continuo con zonas).")


# ================================================================== M4 · S10 — validación cruzada
MORADO = MODULO[4]


def loteria_split(paneles, titulo="La lotería del split: el mismo modelo, 30 particiones distintas"):
    """paneles = {"nombre": lista de scores (uno por random_state del train_test_split)}."""
    fig, axes = plt.subplots(1, len(paneles), figsize=(6.2 * len(paneles), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (n, sc) in zip(axes, paneles.items()):
        sc = np.asarray(sc)
        ax.hist(sc, bins=np.linspace(0.4, 0.85, 31), color=MORADO, alpha=0.85, edgecolor="white")
        ax.axvline(sc.min(), color=ROJO, ls="--"); ax.axvline(sc.max(), color=ROJO, ls="--")
        ax.set_title(f"{n}\nAUC de {sc.min():.3f} a {sc.max():.3f}", fontsize=10.5)
        ax.set_xlabel("AUC en prueba"); ax.set_ylabel("particiones")
    fig.suptitle(titulo, fontsize=11.5, y=1.02); plt.tight_layout()
    return _leyenda(fig, "cada barra cuenta particiones (random_state) distintas del mismo dataset. Con muchos datos la "
                         "lotería casi no importa; con pocos, un solo split puede decir 0,45 o 0,65 del MISMO modelo.")


def cv_folds(k=5, con_prueba=True):
    """(A) Validación cruzada k-fold: cada fila es una ronda; el bloque morado es la validación de esa ronda."""
    fig, ax = plt.subplots(figsize=(11, 0.55 * k + 1.6))
    ancho = 8.0 / k
    for r in range(k):
        y = k - r
        for j in range(k):
            val = j == r
            ax.add_patch(FancyBboxPatch((j * ancho + 0.03, y - 0.38), ancho - 0.06, 0.76, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        fc=MORADO if val else "#E9E3F3", ec="white"))
            ax.text(j * ancho + ancho / 2, y, "validación" if val else "entrenamiento", ha="center", va="center",
                    fontsize=8.5, color="white" if val else NEGRO)
        ax.text(-0.15, y, f"ronda {r + 1}", ha="right", va="center", fontsize=9.5)
        ax.text(8.15, y, f"→ score {r + 1}", ha="left", va="center", fontsize=9.5, color=MORADO)
    if con_prueba:
        ax.add_patch(FancyBboxPatch((9.6, 0.62), 1.3, k - 0.24, boxstyle="round,pad=0.01,rounding_size=0.08", fc=GRIS, ec="white"))
        ax.text(10.25, (k + 1) / 2, "PRUEBA\n(se guarda\nhasta el\nfinal)", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.set_xlim(-1.3, 11.1); ax.set_ylim(0.3, k + 0.9); ax.axis("off")
    ax.set_title(f"Validación cruzada con k = {k}: cada fila de entrenamiento se usa {k - 1} veces para entrenar y 1 para validar")
    return _leyenda(fig, f"en vez de UN examen, {k}: el modelo se entrena {k} veces y cada vez se valida con un pedazo distinto. "
                         "La prueba (gris) no se toca hasta el final.")


def cv_scores(resultados, titulo="Score de cada fold y promedio", metrica="AUC"):
    """(B) resultados = {"modelo": array de scores por fold}. Puntos = folds; barra = promedio ± desviación."""
    fig, ax = plt.subplots(figsize=(1.9 * len(resultados) + 3.5, 4.2))
    for i, (n, sc) in enumerate(resultados.items()):
        sc = np.asarray(sc); c = COLORES_CLUSTER[i % 8]
        ax.bar(i, sc.mean(), color=c, alpha=0.25, width=0.6)
        ax.errorbar(i, sc.mean(), yerr=sc.std(), color=c, capsize=8, lw=2)
        ax.scatter(np.full(len(sc), i) + np.linspace(-0.15, 0.15, len(sc)), sc, color=c, s=28, zorder=3)
        ax.text(i, max(sc.max(), sc.mean() + sc.std()) + 0.006, f"{sc.mean():.3f} ± {sc.std():.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(resultados))); ax.set_xticklabels(list(resultados)); ax.set_ylabel(metrica)
    lo = min(np.min(v) for v in resultados.values()); hi = max(np.max(v) for v in resultados.values())
    ax.set_ylim(lo - 0.03, hi + 0.03); ax.set_title(titulo)
    return _leyenda(fig, "cada punto es un fold. Si las diferencias entre modelos son mucho más grandes que la dispersión "
                         "de sus puntos, la diferencia es real; si se traslapan, no se puede decir cuál es mejor.")


def cv_k_efecto(modelo, X, y, ks=(2, 3, 5, 10, 20), scoring="roc_auc", semilla=42):
    """(C) La perilla de la validación cruzada: k. Más folds = más entrenamiento por ronda, más tiempo."""
    import time
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    filas = []
    for k in ks:
        t = time.time(); sc = cross_val_score(modelo, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=semilla), scoring=scoring)
        filas.append((k, sc, time.time() - t))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for k, sc, _ in filas:
        axes[0].scatter(np.full(len(sc), k), sc, color=MORADO, alpha=0.5, s=18)
    axes[0].plot([f[0] for f in filas], [f[1].mean() for f in filas], "o-", color=NEGRO, lw=2, label="promedio")
    axes[0].set_xlabel("k (número de folds)"); axes[0].set_ylabel(scoring); axes[0].set_xticks(ks); axes[0].legend()
    axes[0].set_title("El promedio casi no cambia; los folds se dispersan")
    axes[1].bar([str(f[0]) for f in filas], [f[2] for f in filas], color=MORADO)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("segundos"); axes[1].set_title("El tiempo crece con k")
    plt.tight_layout()
    return _leyenda(fig, "k = 5 o 10 es el estándar: el promedio ya es estable y el costo es razonable. Con k grande, cada "
                         "fold de validación es pequeño y su score es más ruidoso (puntos más dispersos).")


def duplicados_fuga(n=12, semilla=3):
    """(A) Por qué duplicar filas ANTES de validar es trampa: la misma fila termina en entrenamiento y en validación."""
    rng = np.random.default_rng(semilla)
    ids = list(range(1, n + 1)); si = set(int(v) for v in rng.choice(ids, 3, replace=False))
    filas = ids + [i for i in sorted(si) for _ in range(3)]
    mezcla = [int(v) for v in rng.permutation(filas)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.6))
    for ax, (t, orden, partir) in zip(axes, [("1. Duplicar los 'sí' (rojos) en TODAS las filas", filas, False),
                                              ("2. Después repartir: entrenamiento (izquierda) | validación (derecha)", mezcla, True)]):
        corte = len(orden) - 5
        for j, i in enumerate(orden):
            x = j + (0.8 if partir and j >= corte else 0)
            c = ROJO if i in si else AZUL
            ax.add_patch(FancyBboxPatch((x, 0), 0.86, 0.8, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="white"))
            ax.text(x + 0.43, 0.4, str(i), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if partir:
            ax.axvline(corte + 0.37, color=NEGRO, lw=2, ls="--")
            rep = sorted({i for i in orden[corte:] if i in si and i in orden[:corte]})
            if rep:
                ax.text(len(orden) + 1.0, 0.4, f"← el cliente {', '.join(map(str, rep))} está a los dos lados", va="center", fontsize=9.5, color=ROJO)
        ax.set_xlim(-0.3, len(orden) + 9); ax.set_ylim(-0.2, 1.1); ax.axis("off"); ax.set_title(t, fontsize=10.5, loc="left")
    plt.tight_layout()
    return _leyenda(fig, "si se duplican filas antes de partir, copias del MISMO cliente quedan en entrenamiento y en "
                         "validación. El modelo no predice: reconoce. El score se infla y la prueba de verdad lo desmiente.")


def promesa_vs_realidad(filas, metricas=("AUC", "recall")):
    """filas = {"estrategia": {"CV": {met: v}, "prueba": {met: v}}}. Barras: lo que prometió la validación vs lo que dio la prueba."""
    fig, axes = plt.subplots(1, len(metricas), figsize=(6.2 * len(metricas), 4.2))
    for ax, met in zip(np.atleast_1d(axes), metricas):
        xs = np.arange(len(filas))
        cv = [v["CV"][met] for v in filas.values()]; te = [v["prueba"][met] for v in filas.values()]
        ax.bar(xs - 0.18, cv, 0.36, color=MORADO, label="lo que prometió la validación cruzada")
        ax.bar(xs + 0.18, te, 0.36, color=GRIS, label="lo que dio la prueba (datos nuevos)")
        for x, a, b in zip(xs, cv, te):
            ax.text(x - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=9.5, fontweight="bold")
            ax.text(x + 0.18, b + 0.01, f"{b:.2f}", ha="center", fontsize=9.5)
        ax.set_xticks(xs); ax.set_xticklabels(list(filas), fontsize=9.5); ax.set_ylim(0, 1.15); ax.set_title(met)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    return _leyenda(fig, "una validación honesta promete lo que después cumple (barras parecidas). La que vio copias de "
                         "sus propias filas promete mucho más de lo que entrega.")


# ================================================================== M4 · S11 — boosting, búsqueda de hiperparámetros, SHAP
def boosting_paso_a_paso(etapas=(1, 3, 10, 100), n=120, eta=0.3, semilla=0):
    """(A) Gradient boosting en 1D: cada árbol pequeño corrige lo que los anteriores dejaron sin explicar (los residuales)."""
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(semilla)
    x = np.sort(rng.uniform(0, 10, n)); y = np.sin(x) + 0.3 * x + rng.normal(0, 0.3, n)
    m = GradientBoostingRegressor(n_estimators=max(etapas), learning_rate=eta, max_depth=1, random_state=0).fit(x[:, None], y)
    malla = np.linspace(0, 10, 400)[:, None]
    pred_m = list(m.staged_predict(malla)); pred_x = list(m.staged_predict(x[:, None]))
    fig, axes = plt.subplots(2, len(etapas), figsize=(4.2 * len(etapas), 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for j, k in enumerate(etapas):
        ax = axes[0, j]
        ax.scatter(x, y, s=10, color=GRIS, alpha=0.7)
        ax.plot(malla[:, 0], pred_m[k - 1], color=MORADO, lw=2.5)
        mse = np.mean((y - pred_x[k - 1]) ** 2)
        ax.set_title(f"{k} árbol{'es' if k > 1 else ''} (de 1 corte cada uno)\nMSE = {mse:.2f}", fontsize=10)
        r = y - pred_x[k - 1]; ax2 = axes[1, j]
        ax2.bar(x, r, width=0.07, color=np.where(r > 0, ROJO, AZUL)); ax2.axhline(0, color=NEGRO, lw=0.6)
        ax2.set_ylim(-2.2, 2.2); ax2.set_xlabel("x")
    axes[0, 0].set_ylabel("y"); axes[1, 0].set_ylabel("residuales\n(lo que falta)")
    plt.tight_layout()
    return _leyenda(fig, "cada árbol nuevo se entrena para predecir los residuales que dejaron los anteriores, y se suma con un "
                         f"peso pequeño (learning_rate = {eta}). Los residuales (abajo) se encogen árbol a árbol. Bosque: árboles en paralelo; boosting: en fila.")


def boosting_curva(modelo):
    """(B/C) Curva de entrenamiento de un HistGradientBoosting con early_stopping: pérdida en entrenamiento y
    en validación interna árbol a árbol. Se detiene cuando la validación deja de mejorar."""
    tr = -np.asarray(modelo.train_score_); va = -np.asarray(modelo.validation_score_)
    it = np.arange(len(tr)); mejor = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(it, tr, color=MORADO, lw=2, ls="--", label="entrenamiento")
    ax.plot(it, va, color=MORADO, lw=2.5, label="validación interna (10 %)")
    ax.axvline(mejor, color=ROJO, ls=":", lw=1.5); ax.text(mejor, va.max(), f"  mejor: árbol {mejor}", color=ROJO, va="top")
    ax.set_xlabel("número de árboles"); ax.set_ylabel("log-loss (menor es mejor)"); ax.legend()
    ax.set_title(f"Parada temprana: se detuvo en {len(tr) - 1} árboles")
    return _leyenda(fig, "la pérdida de entrenamiento baja siempre; la de validación baja, se aplana y empieza a subir (sobreajuste, "
                         "Sesión 3). early_stopping corta ahí solo: el número de árboles deja de ser una perilla que hay que buscar.")


def grid_heatmap(cv_results, p1, p2, metrica="AUC"):
    """(C) Resultado de un GridSearchCV de dos hiperparámetros como mapa de calor."""
    r = pd.DataFrame(cv_results)
    t = r.pivot_table(index=f"param_{p1}", columns=f"param_{p2}", values="mean_test_score")
    fig, ax = plt.subplots(figsize=(1.4 * t.shape[1] + 3.2, 0.75 * t.shape[0] + 1.8))
    im = ax.imshow(t.values, cmap=LinearSegmentedColormap.from_list("g", ["white", MORADO]), aspect="auto")
    mx = t.values.max()
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=11, fontweight="bold" if v == mx else "normal",
                    color="white" if v > t.values.min() + 0.7 * (mx - t.values.min()) else NEGRO)
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns); ax.set_xlabel(p2)
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(t.index); ax.set_ylabel(p1)
    plt.colorbar(im, ax=ax, shrink=0.8, label=f"{metrica} (validación cruzada)")
    ax.set_title(f"GridSearchCV: {t.size} combinaciones × folds")
    return _leyenda(fig, "cada casilla es el promedio de la validación cruzada de una combinación. Note cuántas casillas quedan a "
                         "menos de 0,005 de la mejor: muchas configuraciones son prácticamente igual de buenas.")


def grid_vs_random(n=9, semilla=0):
    """(A) Por qué la búsqueda al azar suele ganarle a la grilla (Bergstra y Bengio, 2012): si un hiperparámetro
    importa mucho y otro casi nada, la grilla solo prueba 3 valores del importante; el azar prueba 9."""
    rng = np.random.default_rng(semilla)
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    f = lambda a, b: np.exp(-((a - 0.66) ** 2) / 0.012) + 0.08 * b
    g = int(np.sqrt(n)); gx, gy = np.meshgrid(np.linspace(0.1, 0.9, g), np.linspace(0.1, 0.9, g))
    pts = {"Grilla (9 intentos)": (gx.ravel(), gy.ravel()), "Al azar (9 intentos)": (rng.uniform(0, 1, n), rng.uniform(0, 1, n))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (t, (a, b)) in zip(axes, pts.items()):
        ax.contourf(xx, yy, f(xx, yy), levels=20, cmap=LinearSegmentedColormap.from_list("f", ["white", MORADO]), alpha=0.8)
        ax.scatter(a, b, s=70, color=ROJO, edgecolor=NEGRO, zorder=3)
        for v in a:
            ax.plot([v, v], [-0.06, -0.01], color=ROJO, lw=2, clip_on=False)
        ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1); ax.set_xlabel("hiperparámetro IMPORTANTE"); ax.set_ylabel("hiperparámetro que casi no importa")
        mejor = f(np.asarray(a), np.asarray(b)).max()
        ax.set_title(f"{t}: {len(set(np.round(a, 3)))} valores distintos del importante\nmejor encontrado = {mejor:.2f} (máximo posible ≈ 1,08)")
    plt.tight_layout()
    return _leyenda(fig, "el color es el desempeño (más oscuro = mejor). La grilla gasta 9 intentos en solo 3 valores del hiperparámetro "
                         "que importa (marcas rojas abajo); al azar, prueba 9 valores distintos y es más probable caer cerca del óptimo.")


def optuna_historia(valores, importancias=None, titulo="Optuna: cada intento aprende de los anteriores"):
    """(B) Historia de una búsqueda: valor de cada intento y el mejor hasta ese momento; a la derecha, qué hiperparámetro importó."""
    v = np.asarray(valores, float); mejor = np.maximum.accumulate(v)
    n = 2 if importancias else 1
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.1)); axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.scatter(np.arange(len(v)), v, color=MORADO, alpha=0.6, s=30, label="cada intento")
    ax.step(np.arange(len(v)), mejor, where="post", color=ROJO, lw=2.5, label="mejor hasta ahí")
    ax.set_xlabel("intento"); ax.set_ylabel("AUC (validación cruzada)"); ax.legend(loc="lower right"); ax.set_title(titulo, fontsize=10.5)
    lo = np.percentile(v, 10); ax.set_ylim(lo - 0.01, v.max() + 0.006)
    if importancias:
        s = pd.Series(importancias).sort_values()
        axes[1].barh(s.index, s.values, color=MORADO); axes[1].set_xlabel("importancia (fracción)"); axes[1].set_title("¿Qué hiperparámetro movió el resultado?", fontsize=10.5)
    plt.tight_layout()
    return _leyenda(fig, "los primeros intentos son exploración al azar; después Optuna concentra los intentos donde le ha ido bien. "
                         "La línea roja sube rápido al principio y luego casi se aplana: la mayor parte de la ganancia llega pronto.")


def ganador_busqueda(cv, test, i_mejor, defecto=None):
    """💥 Maldición del ganador: cada punto es una configuración de la búsqueda (CV vs prueba). La ganadora en CV
    no es la ganadora en la prueba, y su ventaja era en parte suerte."""
    cv = np.asarray(cv); test = np.asarray(test)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.scatter(cv, test, s=34, color=MORADO, alpha=0.55, label="cada configuración probada")
    ax.scatter(cv[i_mejor], test[i_mejor], s=170, marker="*", color=ROJO, edgecolor=NEGRO, zorder=4, label="la 'ganadora' de la búsqueda")
    if defecto is not None:
        ax.scatter(*defecto, s=110, marker="s", color=GRIS, edgecolor=NEGRO, zorder=4, label="valores por defecto (sin buscar)")
    lo = min(cv.min(), test.min()) - 0.01; hi = max(cv.max(), test.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=GRIS, ls=":", lw=1); ax.text(hi, hi, " CV = prueba", fontsize=8.5, color=GRIS, va="bottom", ha="right")
    puesto = int((test > test[i_mejor]).sum()) + 1
    ax.set_xlabel("AUC en validación cruzada (lo que vio la búsqueda)"); ax.set_ylabel("AUC en prueba (datos nuevos)")
    ax.set_title(f"La ganadora en validación queda en el puesto {puesto} de {len(cv)} en la prueba")
    ax.legend(loc="upper left", fontsize=8.5)
    return _leyenda(fig, "casi todos los puntos están por debajo de la diagonal y la nube es ancha: con pocos datos, las diferencias "
                         "entre configuraciones son del tamaño del ruido. El mejor de 60 sorteos siempre sale optimista.")


def comparacion_final(filas, metrica="AUC"):
    """filas = {"modelo": (cv_media, cv_desv, prueba)}. Lo que se reporta en E4: validación cruzada ± y una sola prueba."""
    fig, ax = plt.subplots(figsize=(1.8 * len(filas) + 3.5, 4.3))
    xs = np.arange(len(filas))
    cvm = [v[0] for v in filas.values()]; cvs = [v[1] for v in filas.values()]; te = [v[2] for v in filas.values()]
    ax.bar(xs - 0.18, cvm, 0.36, yerr=cvs, capsize=5, color=MORADO, label="validación cruzada (± desv.)")
    ax.bar(xs + 0.18, te, 0.36, color=GRIS, label="prueba (una sola vez)")
    for x, a, b in zip(xs, cvm, te):
        ax.text(x - 0.18, a + 0.012, f"{a:.3f}", ha="center", fontsize=9); ax.text(x + 0.18, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(list(filas), fontsize=9.5); ax.set_ylim(0.45, max(cvm + te) + 0.06); ax.set_ylabel(metrica)
    ax.legend(loc="upper left", fontsize=9); ax.set_title("Comparación honesta: de la línea base al modelo afinado")
    return _leyenda(fig, "lea los saltos de izquierda a derecha: casi toda la ganancia viene de elegir un buen modelo; afinar "
                         "hiperparámetros suma poco. Si validación y prueba coinciden, el número es confiable.")


def shap_una_fila(valores, base, fila, columnas, real=None, top=8):
    """(D) Una fila entra al boosting: SHAP reparte la predicción (en log-odds) entre las variables, partiendo del promedio."""
    v = pd.Series(np.asarray(valores, float), index=list(columnas))
    orden = v.abs().sort_values(ascending=False).index
    prin = list(orden[:top]); resto = v.drop(prin).sum()
    pasos = [(f"{c} = {fila[c]:g}", v[c]) for c in prin] + [(f"otras {len(v) - top} variables", resto)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(pasos) + 2.2), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]; acum = base
    for i, (n, c) in enumerate(pasos):
        ax.barh(i, c, left=acum, color=ROJO if c > 0 else AZUL)
        ax.text(acum + c, i, f" {c:+.2f}", va="center", fontsize=8.5, ha="left" if c >= 0 else "right"); acum += c
    ax.axvline(base, color=GRIS, ls=":"); ax.axvline(acum, color=NEGRO, lw=1.2)
    ax.set_yticks(range(len(pasos))); ax.set_yticklabels([p[0][:38] for p in pasos], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("log-odds de 'sí'"); ax.set_title(f"Desde el promedio ({base:.2f}) hasta esta predicción ({acum:.2f})")
    zz = np.linspace(-6, 3, 200); s = 1 / (1 + np.exp(-zz)); p = 1 / (1 + np.exp(-acum)); p0 = 1 / (1 + np.exp(-base))
    ax2 = axes[1]; ax2.plot(zz, s, color=MORADO, lw=2.2)
    ax2.plot(base, p0, "o", color=GRIS, ms=8); ax2.plot(acum, p, "o", color=ROJO, ms=10)
    ax2.set_title(f"probabilidad: {p0:.0%} (promedio) → {p:.0%}" + (f"\n(lo que pasó: {'sí' if real == 1 else 'no'})" if real is not None else ""), fontsize=10)
    ax2.set_xlabel("log-odds"); ax2.set_ylabel("probabilidad de 'sí'")
    plt.tight_layout()
    return _leyenda(fig, "SHAP reparte la predicción entre las variables: rojo empuja hacia 'sí', azul hacia 'no', y la suma lleva "
                         "del promedio de todos los clientes a la predicción de este. Es la 'una fila' de la logística (S5), para cualquier modelo.")


def shap_global(valores, columnas, top=12):
    """(B) Importancia global con SHAP: promedio del |aporte| de cada variable sobre muchas filas."""
    s = pd.Series(np.abs(np.asarray(valores)).mean(axis=0), index=list(columnas)).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(9, 0.38 * top + 1.3))
    ax.barh(s.index, s.values, color=MORADO); ax.set_xlabel("promedio de |SHAP| (log-odds)")
    ax.set_title("¿Qué variables mueven más las predicciones?")
    return _leyenda(fig, "a diferencia de la importancia de Gini (S6), SHAP mide cuánto mueve cada variable las predicciones, en las "
                         "mismas unidades para todas. Sigue diciendo qué USA el modelo, no qué CAUSA el resultado.")


# ================================================================== S12: cierre del curso
# Foco por sesión en cada etapa del pipeline: 2 = se profundiza, 1 = se usa/retoma, 0 = no aparece.
SESIONES_CURSO = [
    # (sesión, módulo, título corto, datos, foco DATOS, PREP, MODELO, EVAL, INTERP)
    ("S1", 1, "El pipeline completo", "incidentes viales Medellín", (1, 1, 1, 1, 2)),
    ("S2", 1, "Mirar antes de modelar", "incidentes viales Medellín", (2, 2, 1, 1, 1)),
    ("S3", 1, "Entrenar, probar, no hacerse trampa", "incidentes viales Medellín", (1, 2, 1, 2, 1)),
    ("S4", 2, "Regresión lineal", "Saber 11 Valle", (1, 1, 2, 1, 2)),
    ("S5", 2, "Logística, k-NN y accuracy", "Bank Marketing", (1, 1, 2, 2, 1)),
    ("S6", 2, "Árboles y bosques · E2", "Bank Marketing", (1, 0, 2, 2, 1)),
    ("S7", 3, "K-means y DBSCAN", "pingüinos", (1, 2, 2, 1, 2)),
    ("S8", 3, "PCA y t-SNE", "pingüinos, dígitos", (1, 2, 2, 0, 2)),
    ("S9", 3, "Segmentar municipios · E3", "IPM municipal DANE", (2, 1, 1, 1, 2)),
    ("S10", 4, "Validación cruzada", "Bank Marketing", (0, 2, 1, 2, 0)),
    ("S11", 4, "Boosting, Optuna, SHAP · E4", "Bank Marketing", (0, 1, 2, 2, 2)),
    ("S12", 0, "Presentaciones", "su proyecto", (2, 2, 2, 2, 2)),
]


def mapa_del_curso():
    """El curso entero en una figura: 12 sesiones × 5 etapas del pipeline (color = módulo; intensidad = foco)."""
    from matplotlib.colors import to_rgb
    fig, ax = plt.subplots(figsize=(12, 6.4))
    n = len(SESIONES_CURSO)
    for i, (s, mod, tit, datos, foco) in enumerate(SESIONES_CURSO):
        y = n - 1 - i
        base = np.array(to_rgb(MODULO.get(mod, NEGRO)))
        for j, f in enumerate(foco):
            col = (1 - [0, 0.28, 1][f]) * np.ones(3) + [0, 0.28, 1][f] * base if f else (0.96, 0.96, 0.96)
            ax.add_patch(FancyBboxPatch((j + 0.06, y + 0.1), 0.88, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=col, ec="white", lw=1))
        ax.text(-0.15, y + 0.5, f"{s}  {tit}", ha="right", va="center", fontsize=9.5,
                color=MODULO.get(mod, NEGRO), fontweight="bold")
        ax.text(5.15, y + 0.5, datos, ha="left", va="center", fontsize=8.5, color=GRIS)
    for j, e in enumerate(ETAPAS):
        ax.text(j + 0.5, n + 0.2, {"PREPROCESAMIENTO": "PREPROCE-\nSAMIENTO", "INTERPRETACIÓN": "INTERPRE-\nTACIÓN",
                                    "EVALUACIÓN": "EVALUA-\nCIÓN"}.get(e, e),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=NEGRO)
    ax.set_xlim(-3.6, 7.2); ax.set_ylim(-0.3, n + 0.9); ax.axis("off")
    nombres = {1: "M1 · Intro", 2: "M2 · Supervisados", 3: "M3 · No supervisados", 4: "M4 · Evaluación"}
    for k, (m, t) in enumerate(nombres.items()):
        ax.add_patch(plt.Rectangle((-3.4 + k * 2.6, -0.95), 0.3, 0.3, color=MODULO[m], clip_on=False))
        ax.text(-3.0 + k * 2.6, -0.8, t, va="center", fontsize=8.5)
    ax.set_title("El curso en una figura: cada fila es una sesión; el color fuerte es la etapa que se profundizó",
                 fontsize=11, pad=4)
    return _leyenda(fig, "desde S1 se recorrió el pipeline completo; cada sesión profundizó una o dos etapas. "
                         "En S12 todas las columnas son suyas: su proyecto las recorre todas.")


TRAMPAS_CURSO = [
    # (sesión, familia, trampa, lo que prometía, lo que era)
    ("S2", "datos", "El año como predictor", "accuracy 0,831", "la base dejó de registrar 'solo daños' en oct-2022"),
    ("S3", "fuga", "Target encoding con todas las filas", "+4,7 puntos, train = prueba", "prueba 0,744: peor que sin la variable"),
    ("S4", "fuga", "La y hecha de las X", "R² = 1,000", "redescubrió la fórmula del ICFES; lo honesto: 0,18"),
    ("S5", "métrica", "Accuracy sin línea base", "accuracy 0,894", "la línea base que nunca dice 'sí': 0,883"),
    ("S6", "fuga", "duration: consecuencia de la y", "AUC 0,93", "sin ella, el bosque: ~0,80"),
    ("S7", "analista", "Agrupar sin escalar", "68 % de pureza", "agrupó por gramos; escalado: 92 %"),
    ("S8", "analista", "Creerle al t-SNE", "10 islas perfectas", "tamaños y distancias son artefactos"),
    ("S9", "analista", "IPM junto a sus partes", "'más información'", "75 municipios cambian de grupo"),
    ("S10", "fuga", "Duplicar antes de validar", "CV: AUC 0,96", "prueba: 0,80"),
    ("S11", "métrica", "Creerle al best_score_", "CV: 0,770", "prueba: 0,706 (puesto 15 de 60)"),
]
FAMILIAS_TRAMPA = {"fuga": ("fuga de información", ROJO), "métrica": ("métrica mal leída", MODULO[4]),
                   "datos": ("el dato cambió", MODULO[1]), "analista": ("decisión del analista sin y", MODULO[3])}


def trampas_del_curso():
    """Las 10 trampas del curso: lo que prometían contra lo que eran, por familia."""
    n = len(TRAMPAS_CURSO)
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * n + 1.6))
    ax.set_xlim(0, 12.5); ax.set_ylim(-0.6, n + 0.4); ax.axis("off")
    for x, t in [(0.05, "sesión"), (0.85, "trampa"), (4.6, "lo que prometía"), (8.05, "lo que era")]:
        ax.text(x, n + 0.05, t, fontsize=9, color=GRIS, fontweight="bold")
    for i, (s, fam, tr, prom, real) in enumerate(TRAMPAS_CURSO):
        y = n - 1 - i
        col = FAMILIAS_TRAMPA[fam][1]
        ax.add_patch(FancyBboxPatch((0.02, y + 0.12), 0.65, 0.66, boxstyle="round,pad=0,rounding_size=0.1", fc=col, ec="none"))
        ax.text(0.345, y + 0.45, s, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(0.85, y + 0.45, tr, va="center", fontsize=10, color=NEGRO)
        ax.text(4.6, y + 0.45, prom, va="center", fontsize=10, color=ROJO, fontweight="bold")
        ax.annotate("", xy=(7.9, y + 0.45), xytext=(7.35, y + 0.45), arrowprops=dict(arrowstyle="->", color=GRIS))
        ax.text(8.05, y + 0.45, real, va="center", fontsize=10, color=AZUL)
    for k, (nom, col) in enumerate(FAMILIAS_TRAMPA.values()):
        ax.add_patch(plt.Rectangle((0.05 + k * 3.1, -0.5), 0.25, 0.3, color=col))
        ax.text(0.4 + k * 3.1, -0.35, nom, va="center", fontsize=9)
    ax.set_title("Diez veces el número se veía mejor de lo que era", fontsize=12)
    return _leyenda(fig, "ninguna trampa produjo un error de Python: todas produjeron un número bonito. Cuatro familias: "
                         "fugas, métricas mal leídas, datos que cambian y decisiones del analista que ninguna y delata.")


def plan_presentaciones(n, total=120, apertura=10, pausa=5, cierre=15):
    """Reparte el tiempo de S12 entre n presentaciones. Devuelve un dict con los minutos de cada parte."""
    disponible = total - apertura - pausa - cierre
    turno = float(np.floor(2 * disponible / n) / 2)
    if turno < 5:                        # muchos: sin pausa y cierre corto, antes que recortar las charlas
        pausa, cierre = 0, 10
        disponible = total - apertura - pausa - cierre
        turno = float(np.floor(2 * disponible / n) / 2)
    preguntas = 1.5 if turno >= 6 else 1.0
    cambio = 0.5
    charla = min(turno - preguntas - cambio, 6.0)
    sobra = disponible - n * (charla + preguntas + cambio)
    return {"n": n, "charla": charla, "preguntas": preguntas, "cambio": cambio, "apertura": apertura,
            "pausa": pausa, "cierre": cierre, "colchon": round(float(sobra), 1)}


def cronograma_presentaciones(n, inicio="19:00"):
    """Línea de tiempo de los 120 minutos de S12 para n presentaciones (la pausa va a la mitad)."""
    p = plan_presentaciones(n)
    h0, m0 = map(int, inicio.split(":"))
    bloques = [("apertura", p["apertura"], MODULO[1])]
    mitad = int(np.ceil(n / 2))
    for k in range(n):
        if k == mitad and p["pausa"]:
            bloques.append(("pausa", p["pausa"], "#DDDDDD"))
        bloques.append((f"{k + 1}", p["charla"], MORADO))
        bloques.append(("", p["preguntas"] + p["cambio"], "#C9B6E8"))
    if p["colchon"] > 0:
        bloques.append(("colchón", p["colchon"], "#F2F2F2"))
    bloques.append(("cierre del curso", p["cierre"], MODULO[3]))
    fig, ax = plt.subplots(figsize=(13, 2.4))
    t = 0
    for nom, dur, col in bloques:
        ax.barh(0, dur, left=t, color=col, edgecolor="white", height=0.6)
        if nom and dur >= (3 if len(nom) > 3 else 2):
            ax.text(t + dur / 2, 0, nom, ha="center", va="center", fontsize=8 if len(nom) < 4 else 8.5,
                    color="white" if col in (MORADO, MODULO[1], MODULO[3]) else NEGRO, rotation=0)
        t += dur
    marcas = np.arange(0, 121, 15)
    ax.set_xticks(marcas); ax.set_xticklabels([f"{h0 + (m0 + m) // 60}:{(m0 + m) % 60:02d}" for m in marcas])
    ax.set_xlim(0, 120); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    c = lambda v: f"{v:g}".replace(".", ",")
    ax.set_title(f"{n} presentaciones: {c(p['charla'])} min de charla + {c(p['preguntas'])} de preguntas "
                 f"(+{c(p['cambio'])} de cambio) · cierre {p['cierre']} min", fontsize=10.5)
    return _leyenda(fig, "morado fuerte = charla, morado claro = preguntas y cambio de pantalla. El cronómetro es "
                         "del curso, no del estudiante: al minuto se corta con amabilidad y se pasa a preguntas.")


QUE_SIGUE = [
    ("Profundizar en modelos", MODULO[2], [
        "Series de tiempo: validación que respeta el orden (TimeSeriesSplit)",
        "Texto e imágenes: deep learning con PyTorch o Keras",
        "Inferencia causal: cuando la pregunta es '¿qué pasa si…?'",
        "Modelos de lenguaje como herramienta, no como oráculo"]),
    ("Llevar a producción", MODULO[4], [
        "Del notebook al .py: el Pipeline guardado con joblib (S11)",
        "Registrar experimentos: MLflow",
        "Servir el modelo: una API pequeña (FastAPI)",
        "Monitorear: los datos cambian (S2 lo mostró)"]),
    ("Hacerlo bien", MODULO[1], [
        "Sesgo y equidad: ¿el error se reparte igual entre grupos?",
        "Datos personales en Colombia: Ley 1581 de 2012",
        "Comunicar incertidumbre: el ± de S10 en cada informe",
        "Preguntarle a quien produce el dato"]),
]
RECURSOS = ["James, Witten, Hastie, Tibshirani y Taylor (2023). An Introduction to Statistical Learning with Applications in Python. Gratis en statlearning.com",
            "Géron (2022). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3.ª ed. O'Reilly",
            "Guía de usuario de scikit-learn (scikit-learn.org) · Kaggle Learn (cursos cortos gratuitos)",
            "Datos abiertos: datos.gov.co · DANE · UCI ML Repository"]


def que_sigue():
    """Mapa de rutas después del curso: tres direcciones y lecturas recomendadas."""
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0.15, 5.9); ax.axis("off")
    for k, (tit, col, items) in enumerate(QUE_SIGUE):
        x = 0.1 + k * 4.15
        ax.add_patch(FancyBboxPatch((x, 2.1), 3.95, 3.7, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((x, 5.2), 3.95, 0.6, boxstyle="round,pad=0,rounding_size=0.12", fc=col, ec=col))
        ax.text(x + 1.97, 5.5, tit, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        for i, it in enumerate(items):
            import textwrap
            ax.text(x + 0.15, 4.85 - i * 0.72, "• " + textwrap.fill(it, 40), va="top", fontsize=9, color=NEGRO)
    ax.text(0.1, 1.75, "Para seguir leyendo", fontsize=10.5, fontweight="bold", color=NEGRO)
    for i, r in enumerate(RECURSOS):
        ax.text(0.25, 1.38 - i * 0.36, "• " + r, fontsize=8.8, color=NEGRO)
    ax.set_title("¿Y ahora qué? Tres caminos desde aquí", fontsize=12)
    return _leyenda(fig, "no hace falta recorrer los tres. Lo que se lleva de este curso (separar antes de todo, línea base, "
                         "validación honesta, desconfiar del número bonito) sirve en cualquiera de ellos.")


# ================================================================== ÁRBOL — explicado con pelotas (S1, ampliado 6-oct-2026)
def _caja_pelotas(ax, x, y, w, h, n_rojo, n_azul, titulo=None, cols=5, r=None, borde="#BBBBBB", fondo="white", lw=1.2):
    """Una caja con pelotas: rojas = con víctimas, azules = solo daños. Coordenadas de datos del eje."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fondo, ec=borde, lw=lw, zorder=1))
    n = n_rojo + n_azul
    filas = int(np.ceil(n / cols)) if n else 1
    r = r or min(w / (cols * 2.6), h / (filas * 2.6 + (1.2 if titulo else 0.4)))
    colores = [ROJO] * n_rojo + [AZUL] * n_azul
    alto_util = h - (0.9 if titulo else 0.2)
    for i, c in enumerate(colores):
        fi, co = divmod(i, cols)
        en_fila = min(cols, n - fi * cols)
        cx = x + w / 2 + (co - (en_fila - 1) / 2) * r * 2.5
        cy = y + alto_util / 2 + ((filas - 1) / 2 - fi) * r * 2.5 + 0.1
        ax.add_patch(plt.Circle((cx, cy), r, color=c, zorder=2))
    if titulo:
        ax.text(x + w / 2, y + h - 0.35, titulo, ha="center", va="top", fontsize=10.5, color=NEGRO, zorder=3)


def _gini(n1, n):
    return 0.0 if n == 0 else 1 - (n1 / n) ** 2 - (1 - n1 / n) ** 2


def gini_pelotas(casos=((10, 0), (8, 2), (5, 5))):
    """(A) Qué mide Gini, con pelotas: cajas con distinta mezcla y la cuenta hecha a mano.
    casos: (rojas, azules) por caja."""
    fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 36); ax.set_ylim(0, 15); ax.axis("off")
    etiquetas = ["pura", "casi pura", "revuelta al 50/50", "otra"]
    for k, (nr, na) in enumerate(casos):
        x = 1 + k * 12
        n = nr + na; p1, p0 = nr / n, na / n; g = _gini(nr, n)
        _caja_pelotas(ax, x, 6.6, 10, 7.6, nr, na, titulo=f"{nr} con víctimas · {na} solo daños")
        ax.text(x + 5, 5.75, f"p(rojo) = {nr}/{n} = {p1:.1f}".replace(".", ","), ha="center", fontsize=10.5, color=ROJO)
        ax.text(x + 5, 4.95, f"p(azul) = {na}/{n} = {p0:.1f}".replace(".", ","), ha="center", fontsize=10.5, color=AZUL)
        ax.text(x + 5, 3.95, f"G = 1 − {p1:.1f}² − {p0:.1f}²".replace(".", ","), ha="center", fontsize=11.5, color=NEGRO)
        ax.text(x + 5, 3.05, f"= 1 − {p1**2:.2f} − {p0**2:.2f}".replace(".", ","), ha="center", fontsize=11.5, color=NEGRO)
        col = AZUL if g < 0.1 else (ROJO if g >= 0.45 else "#D97C1F")
        ax.text(x + 5, 1.55, f"G = {g:.2f}".replace(".", ","), ha="center", fontsize=19, fontweight="bold", color=col)
        ax.text(x + 5, 0.45, etiquetas[k] if k < 3 else "", ha="center", fontsize=10, color=GRIS, style="italic")
    ax.set_title("Impureza de Gini: qué tan revuelta está una caja", fontsize=13)
    return _leyenda(fig, "G = 0 cuando todas las pelotas son del mismo color (caja pura); con dos colores, el máximo es "
                         "0,5 (mitad y mitad). El árbol busca preguntas que dejen cajas con G bajo.")


def gini_dos_cortes():
    """(A) Gini ponderado: la misma caja de 10 incidentes partida por dos preguntas distintas.
    Gana la pregunta con el promedio (ponderado por tamaño) más bajo."""
    fig, ax = plt.subplots(figsize=(13, 6.6)); ax.set_xlim(0, 40); ax.set_ylim(0, 20); ax.axis("off")
    _caja_pelotas(ax, 15, 14.2, 10, 5.4, 6, 4, titulo="10 incidentes: 6 con víctimas, 4 solo daños", cols=10)
    ax.text(20, 13.4, "G = 1 − 0,6² − 0,4² = 0,48", ha="center", fontsize=11, color=NEGRO)
    cortes = [("Pregunta A: ¿hora ≤ 6?", (4, 0), (2, 4), 1.0), ("Pregunta B: ¿velocidad ≤ 50?", (3, 2), (3, 2), 21.0)]
    resumen = []
    for titulo, (r1, a1), (r2, a2), x0 in cortes:
        ax.text(x0 + 9, 11.9, titulo, ha="center", fontsize=12.5, fontweight="bold", color=NEGRO)
        ax.annotate("", xy=(x0 + 9, 12.5), xytext=(20, 13.0), arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.2))
        n1, n2 = r1 + a1, r2 + a2; g1, g2 = _gini(r1, n1), _gini(r2, n2); gp = (n1 * g1 + n2 * g2) / 10
        resumen.append(gp)
        for j, (rr, aa, gg, nn, lado) in enumerate([(r1, a1, g1, n1, "sí"), (r2, a2, g2, n2, "no")]):
            x = x0 + j * 9.4
            ax.text(x + 4.2, 11.0, lado, ha="center", fontsize=11, color=GRIS, fontweight="bold")
            _caja_pelotas(ax, x, 6.3, 8.4, 4.4, rr, aa, cols=5)
            ax.text(x + 4.2, 5.5, f"{rr} rojas · {aa} azules", ha="center", fontsize=10, color=NEGRO)
            ax.text(x + 4.2, 4.5, f"G = {gg:.2f}".replace(".", ","), ha="center", fontsize=12, fontweight="bold",
                    color=AZUL if gg < 0.1 else NEGRO)
        ax.text(x0 + 9, 2.9, f"ponderado = {n1}/10·{g1:.2f} + {n2}/10·{g2:.2f}".replace(".", ","), ha="center", fontsize=11.5, color=NEGRO)
        ax.text(x0 + 9, 1.4, f"= {gp:.2f}".replace(".", ","), ha="center", fontsize=20, fontweight="bold",
                color=AZUL if gp == min(resumen) and len(resumen) == 1 else NEGRO)
    gana = int(np.argmin(resumen))
    ax.add_patch(FancyBboxPatch((1 + 20 * gana - 0.4, 0.4), 18.8, 12.3, boxstyle="round,pad=0,rounding_size=0.4",
                                fc="none", ec=AZUL, lw=2.4, zorder=0))
    ax.text(1 + 20 * gana + 18.2, 0.8, "gana", ha="right", fontsize=12, color=AZUL, fontweight="bold")
    return _leyenda(fig, "Cada lado se pesa por cuántos casos tiene. La pregunta A deja un lado puro (G = 0) y baja el "
                         "promedio a 0,27; la B no separa nada (0,48 = igual que antes). El árbol se queda con A.")


def arbol_mini_datos():
    """(A) Los 10 incidentes de las pelotas, en una tabla y en el plano, y TODOS los cortes posibles de 'hora'
    con su Gini ponderado: el árbol prueba cada uno y se queda con el más bajo."""
    hora = np.array([1, 2, 3, 5, 7, 10, 13, 17, 19, 22])
    vel = np.array([75, 40, 62, 85, 45, 70, 35, 55, 80, 48])
    y = np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.35], wspace=0.35)
    axt = fig.add_subplot(gs[0]); axt.axis("off")
    filas = [[f"{h} h", f"{v} km/h", "sí" if c else "no"] for h, v, c in zip(hora, vel, y)]
    t = axt.table(cellText=filas, colLabels=["hora", "velocidad", "víctimas"], loc="center", cellLoc="center",
                  colWidths=[0.28, 0.42, 0.34])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.15, 1.45)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#DDDDDD")
        if i == 0: c.set_facecolor("#F0F0F0"); c.set_text_props(fontweight="bold")
        elif j == 2: c.set_text_props(color=ROJO if y[i - 1] else AZUL, fontweight="bold")
    axt.set_title("10 incidentes", fontsize=12)
    axp = fig.add_subplot(gs[1])
    axp.scatter(hora[y == 1], vel[y == 1], s=90, color=ROJO, label="con víctimas", zorder=3)
    axp.scatter(hora[y == 0], vel[y == 0], s=90, color=AZUL, label="solo daños", zorder=3)
    axp.axvline(6, color=NEGRO, ls="--", lw=1.6); axp.text(6.4, 30, "¿hora ≤ 6?", fontsize=10.5)
    axp.set_xlim(0, 24); axp.set_ylim(28, 98); axp.set_xlabel("hora"); axp.set_ylabel("velocidad (km/h)")
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=9, frameon=False, handletextpad=0.2); axp.set_title("En el plano", fontsize=12, pad=18)
    axg = fig.add_subplot(gs[2])
    orden = np.sort(hora); umbrales = (orden[:-1] + orden[1:]) / 2; gps = []
    for u in umbrales:
        izq, der = y[hora <= u], y[hora > u]
        gps.append((len(izq) * _gini(izq.sum(), len(izq)) + len(der) * _gini(der.sum(), len(der))) / len(y))
    gps = np.array(gps); k = gps.argmin()
    barras = axg.bar(range(len(umbrales)), gps, color=["#D0D7E2"] * len(gps))
    barras[k].set_color(AZUL)
    for i, g in enumerate(gps):
        axg.text(i, g + 0.008, f"{g:.2f}".replace(".", ","), ha="center", fontsize=8.5, color=NEGRO if i != k else AZUL,
                 fontweight="bold" if i == k else "normal")
    axg.set_xticks(range(len(umbrales))); axg.set_xticklabels([f"≤{u:g}" for u in umbrales], fontsize=8.5, rotation=45)
    axg.axhline(0.48, color=GRIS, ls=":", lw=1); axg.text(-0.5, 0.515, "línea punteada = sin cortar (0,48)", ha="left", fontsize=8.5, color=GRIS)
    axg.set_ylim(0, 0.55); axg.set_ylabel("Gini ponderado"); axg.set_xlabel("corte probado en 'hora'")
    axg.set_title(f"Los {len(umbrales)} cortes posibles en 'hora'", fontsize=12)
    return _leyenda(fig, "Entre cada par de horas vecinas hay un corte posible. El árbol los calcula todos (y los de "
                         f"velocidad) y elige el más bajo: hora ≤ {umbrales[k]:g}, Gini {gps[k]:.2f}".replace("0.", "0,") + ".")


def _texto_pregunta(nombre, umbral, dummies=True):
    """Convierte 'clase_Choque <= 0.5' en '¿Es Choque?' y 'hora_num <= 7.5' en '¿hora ≤ 7?'.
    Devuelve (texto, invertido): invertido=True si el lado '≤' significa 'no'."""
    if dummies and "_" in nombre and abs(umbral - 0.5) < 1e-9:
        var, val = nombre.split("_", 1)
        return f"¿{var} = {val}?", True
    nombres = {"hora_num": "hora", "anio": "año", "dia_semana": "día (0 = lunes)", "mes": "mes"}
    v = nombres.get(nombre, nombre)
    u = int(np.floor(umbral)) if abs(umbral - round(umbral) - 0.5) < 1e-9 else round(umbral, 1)
    return f"¿{v} ≤ {u}?", False


def arbol_reglas(modelo, nombres, max_depth=2, class_names=("solo daños", "con víctimas"), figsize=(13, 5.0), fila=None):
    """(B) El árbol entrenado, dibujado como diagrama de flujo legible: cada caja es una pregunta en palabras,
    con cuántos casos llegan y una barra con la mezcla de clases; las ramas dicen sí/no; las hojas dicen
    qué predice. Se recorta a `max_depth` niveles. `fila` (opcional): una fila de X para marcar su camino."""
    t = modelo.tree_; nombres = list(nombres); total = t.n_node_samples[0]
    valores = t.value[:, 0, :]; valores = valores / valores.sum(axis=1, keepdims=True)
    pos, hojas = {}, [0]
    def ubicar(nodo, prof):
        izq, der = t.children_left[nodo], t.children_right[nodo]
        if izq == -1 or prof == max_depth:
            pos[nodo] = (hojas[0], prof); hojas[0] += 1; return pos[nodo][0]
        txt, inv = _texto_pregunta(nombres[t.feature[nodo]], t.threshold[nodo])
        orden = (der, izq) if inv else (izq, der)          # 'sí' siempre a la izquierda
        xs = [ubicar(h, prof + 1) for h in orden]
        pos[nodo] = (sum(xs) / 2, prof); return pos[nodo][0]
    ubicar(0, 0)
    camino = set()
    if fila is not None:
        nodo, f = 0, np.asarray(fila, dtype=float)
        while True:
            camino.add(nodo); prof = pos[nodo][1]
            if t.children_left[nodo] == -1 or prof == max_depth: break
            nodo = t.children_left[nodo] if f[t.feature[nodo]] <= t.threshold[nodo] else t.children_right[nodo]
    n_hojas = hojas[0]
    fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
    ax.set_xlim(-0.6, n_hojas - 0.4); ax.set_ylim(-max_depth - 0.32, 0.4)
    W, H = 0.86, 0.5
    def caja(nodo):
        x, prof = pos[nodo]; yv = -prof
        n = t.n_node_samples[nodo]; p1 = valores[nodo, 1]
        hoja = t.children_left[nodo] == -1 or prof == max_depth
        en_camino = nodo in camino
        ec = "#E8B500" if en_camino else ("#BBBBBB" if not hoja else (ROJO if p1 >= 0.5 else AZUL))
        ax.add_patch(FancyBboxPatch((x - W / 2, yv - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="white" if not hoja else ("#FBECEA" if p1 >= 0.5 else "#E9F0F9"), ec=ec,
                                    lw=3 if en_camino else 1.4, zorder=2))
        if not hoja:
            txt, _ = _texto_pregunta(nombres[t.feature[nodo]], t.threshold[nodo])
            ax.text(x, yv + 0.12, txt, ha="center", va="center", fontsize=11.5, fontweight="bold", color=NEGRO, zorder=3)
        else:
            pred = class_names[1] if p1 >= 0.5 else class_names[0]
            ax.text(x, yv + 0.12, f"→ {pred}", ha="center", va="center", fontsize=11.5, fontweight="bold",
                    color=ROJO if p1 >= 0.5 else AZUL, zorder=3)
        # barra con la mezcla
        bx, by, bw, bh = x - W / 2 + 0.06, yv - 0.15, W - 0.12, 0.07
        ax.add_patch(plt.Rectangle((bx, by), bw * (1 - p1), bh, color=AZUL, zorder=3))
        ax.add_patch(plt.Rectangle((bx + bw * (1 - p1), by), bw * p1, bh, color=ROJO, zorder=3))
        ax.text(x, yv - 0.035, f"{n / total:.0%} de los casos · {p1:.0%} graves".replace("%", " %"), ha="center",
                va="center", fontsize=8.6, color="#555555", zorder=3)
        if not hoja:
            txt, inv = _texto_pregunta(nombres[t.feature[nodo]], t.threshold[nodo])
            si, no = (t.children_right[nodo], t.children_left[nodo]) if inv else (t.children_left[nodo], t.children_right[nodo])
            for hijo, etq in ((si, "sí"), (no, "no")):
                hx, hp = pos[hijo]
                col = "#E8B500" if (nodo in camino and hijo in camino) else "#999999"
                ax.annotate("", xy=(hx, -hp + H / 2), xytext=(x, yv - H / 2),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6 if col != "#999999" else 1.3), zorder=1)
                mx, my = (x + hx) / 2, (yv - H / 2 + (-hp + H / 2)) / 2
                ax.text(mx, my, etq, ha="center", va="center", fontsize=10, fontweight="bold",
                        color=AZUL if etq == "sí" else GRIS,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"), zorder=3)
                caja(hijo)
    caja(0)
    ax.set_title(f"Las primeras {max_depth} preguntas que aprendió el árbol (tiene {modelo.get_depth()} niveles)", fontsize=13)
    return _leyenda(fig, "Léalo de arriba abajo, como un diagrama de flujo. Cada caja dice cuántos casos llegan y la barra, "
                         "su mezcla (azul = solo daños, rojo = con víctimas). La hoja predice el color mayoritario.")


def arbol_recorrido(fila=(3, 80)):
    """(A) El mismo árbol de dos maneras: como diagrama de flujo y como mapa de rectángulos, con las hojas numeradas
    igual en los dos. Una fila nueva (hora, velocidad) recorre las preguntas y cae en una hoja."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = _datos_sinteticos(300)
    m = DecisionTreeClassifier(max_depth=2, min_samples_leaf=15, random_state=0).fit(X, y)
    t = m.tree_; val = t.value[:, 0, :] / t.value[:, 0, :].sum(axis=1, keepdims=True)
    fig = plt.figure(figsize=(15, 6.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    SEP = 1.3                                            # separación horizontal entre hojas
    pos, cont = {}, [0]
    def _ub(n, d):
        if t.children_left[n] == -1:
            pos[n] = (cont[0] * SEP, -d * 1.15); cont[0] += 1; return pos[n][0]
        a, b = _ub(t.children_left[n], d + 1), _ub(t.children_right[n], d + 1)
        pos[n] = ((a + b) / 2, -d * 1.15); return pos[n][0]
    _ub(0, 0)
    ax.set_xlim(-0.65, (cont[0] - 1) * SEP + 0.65); ax.set_ylim(-2.3 - 0.55, 0.55)
    f = np.array(fila, dtype=float); camino = [0]; nodo = 0
    while t.children_left[nodo] != -1:
        nodo = t.children_left[nodo] if f[t.feature[nodo]] <= t.threshold[nodo] else t.children_right[nodo]; camino.append(nodo)
    nombres = ["hora", "velocidad"]; unid = [" h", " km/h"]
    hojas = [n for n in range(t.node_count) if t.children_left[n] == -1]
    num = {h: i + 1 for i, h in enumerate(hojas)}
    W, H = 1.18, 0.86
    for n in range(t.node_count):
        x, yv = pos[n]; p1 = val[n, 1]; hoja = t.children_left[n] == -1; enc = n in camino
        ax.add_patch(FancyBboxPatch((x - W / 2, yv - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=("#FBECEA" if p1 >= .5 else "#E9F0F9") if hoja else "white",
                                    ec="#E8B500" if enc else "#BBBBBB", lw=3.2 if enc else 1.3, zorder=2))
        if hoja:
            ax.text(x, yv + 0.3, f"hoja {num[n]}", ha="center", va="center", fontsize=10, color="#555555", zorder=3)
            ax.text(x, yv + 0.13, "con víctimas" if p1 >= .5 else "solo daños", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=ROJO if p1 >= .5 else AZUL, zorder=3)
        else:
            ax.text(x, yv + 0.18, f"¿{nombres[t.feature[n]]} ≤ {t.threshold[n]:.0f}{unid[t.feature[n]]}?", ha="center",
                    va="center", fontsize=12, fontweight="bold", color=NEGRO, zorder=3)
            for hijo, etq in ((t.children_left[n], "sí"), (t.children_right[n], "no")):
                hx, hy = pos[hijo]; on = enc and hijo in camino
                ax.annotate("", xy=(hx, hy + H / 2), xytext=(x, yv - H / 2),
                            arrowprops=dict(arrowstyle="-|>", color="#E8B500" if on else "#999999", lw=2.8 if on else 1.3))
                ax.text((x + hx) / 2, (yv - H / 2 + hy + H / 2) / 2, etq, ha="center", va="center", fontsize=11, fontweight="bold",
                        color=AZUL if etq == "sí" else GRIS, bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none"), zorder=3)
        ax.text(x, yv - 0.04, f"{t.n_node_samples[n]} casos", ha="center", va="center", fontsize=9.5, color="#555555", zorder=3)
        ax.text(x, yv - 0.32, f"{p1:.0%} con víctimas".replace("%", " %"), ha="center", va="center", fontsize=9, color=ROJO, zorder=3)
        bx = x - W / 2 + .1; bw = W - .2
        ax.add_patch(plt.Rectangle((bx, yv - .2), bw * (1 - p1), .08, color=AZUL, zorder=3))
        ax.add_patch(plt.Rectangle((bx + bw * (1 - p1), yv - .2), bw * p1, .08, color=ROJO, zorder=3))
    ax.set_title("Como diagrama de flujo", fontsize=13)
    # mapa
    axm = fig.add_subplot(gs[1])
    xx, yy = np.meshgrid(np.linspace(0, 24, 300), np.linspace(20, 100, 300))
    pr = m.predict_proba(pd.DataFrame({"hora": xx.ravel(), "velocidad": yy.ravel()}))[:, 1].reshape(xx.shape)
    axm.contourf(xx, yy, pr, levels=[0, .5, 1], colors=["#E9F0F9", "#FBECEA"])
    axm.scatter(X["hora"][y == 1], X["velocidad"][y == 1], s=12, color=ROJO, alpha=.6)
    axm.scatter(X["hora"][y == 0], X["velocidad"][y == 0], s=12, color=AZUL, alpha=.6)
    hoja_de = m.apply(pd.DataFrame({"hora": xx.ravel(), "velocidad": yy.ravel()})).reshape(xx.shape)
    for h in hojas:
        msk = hoja_de == h
        if msk.any():
            axm.text(xx[msk].mean(), yy[msk].mean(), str(num[h]), ha="center", va="center", fontsize=20, fontweight="bold",
                     color="white", bbox=dict(boxstyle="circle,pad=.25", fc=ROJO if val[h, 1] >= .5 else AZUL, ec="white"))
    def cortes(n, x0, x1, y0, y1):
        if t.children_left[n] == -1: return
        if t.feature[n] == 0:
            axm.plot([t.threshold[n]] * 2, [y0, y1], color=NEGRO, lw=1.8)
            cortes(t.children_left[n], x0, t.threshold[n], y0, y1); cortes(t.children_right[n], t.threshold[n], x1, y0, y1)
        else:
            axm.plot([x0, x1], [t.threshold[n]] * 2, color=NEGRO, lw=1.8)
            cortes(t.children_left[n], x0, x1, y0, t.threshold[n]); cortes(t.children_right[n], x0, x1, t.threshold[n], y1)
    cortes(0, 0, 24, 20, 100)
    axm.scatter(*fila, s=260, marker="*", color="#E8B500", edgecolor=NEGRO, lw=1, zorder=5)
    axm.annotate(f"nuevo: {fila[0]} h, {fila[1]} km/h", xy=fila, xytext=(fila[0] + 3, fila[1] - 7), fontsize=10, bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85),
                 arrowprops=dict(arrowstyle="->", color=NEGRO))
    axm.set_xlim(0, 24); axm.set_ylim(20, 100); axm.set_xlabel("hora"); axm.set_ylabel("velocidad (km/h)")
    axm.set_title("Como mapa de rectángulos", fontsize=12.5)
    hoja_final = num[camino[-1]]
    return _leyenda(fig, f"Cada hoja del diagrama es un rectángulo del mapa (mismo número). El incidente nuevo (★) responde "
                         f"las preguntas por el camino amarillo y cae en la hoja {hoja_final}: esa es su predicción.")


# ================================================================== aviso de ✏️ pendientes (versión estudiante)
def _aviso_todo(lineas):
    """Transformador de IPython: si la celda todavía tiene un ``...`` con un comentario TODO en la
    misma línea, no la ejecuta y muestra un mensaje claro. Así "Ejecutar todo" se detiene justo en el
    primer ✏️ sin completar, en vez de seguir con valores vacíos y fallar mucho más abajo."""
    pend = [l.strip() for l in lineas if ("." * 3) in l.split("#")[0] and "TODO" in l.split("#", 1)[-1]]
    if not pend:
        return lineas
    msj = ("✏️ Esta celda tiene un ejercicio sin completar. Reemplace los tres puntos de esta línea "
           "y vuelva a correrla:  " + pend[0][:140])
    return [f"raise RuntimeError({msj!r})\n"]


def _activar_aviso_todo():
    try:
        ip = get_ipython()  # noqa: F821 (solo existe en Jupyter/Colab)
    except NameError:
        return
    if all(getattr(f, "__name__", "") != "_aviso_todo" for f in ip.input_transformers_post):
        ip.input_transformers_post.append(_aviso_todo)


_activar_aviso_todo()


# ================================================================== M1 · S2–S3 — figuras pedagógicas (A) añadidas 6-oct-2026
# Regla 6.4: la intuición se VE antes de correr el código, sobre datos pequeños o sintéticos.

def _serie_sintetica(rng, n=48, base_rojo=1900, base_azul=1300, ruido=120):
    rojo = base_rojo + rng.normal(0, ruido, n)
    azul = base_azul + rng.normal(0, ruido * 0.8, n)
    return rojo.clip(0), azul.clip(0)


def tres_hipotesis(mes_corte=30, n=48):
    """(A) Antes de ver la serie real: cómo se vería la gráfica de incidentes por mes bajo cada hipótesis.
    (a) la ciudad se volvió más peligrosa → el rojo SUBE; (b) cambió el registro → el rojo sigue igual y el
    azul DESAPARECE; (c) casualidad → nada cambia en el corte. Datos inventados: solo sirven para comparar formas."""
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    titulos = ["(a) la ciudad se volvió más peligrosa", "(b) cambió la forma de registrar", "(c) casualidad del árbol"]
    x = np.arange(n)
    for k, ax in enumerate(axes):
        rojo, azul = _serie_sintetica(rng, n)
        if k == 0:
            rojo[mes_corte:] *= 1.7; azul[mes_corte:] *= 0.95
        elif k == 1:
            azul[mes_corte:] *= 0.02
        ax.stackplot(x, rojo, azul, colors=[ROJO, AZUL], alpha=0.85, labels=["con víctimas", "solo daños"])
        ax.axvline(mes_corte, color=NEGRO, ls="--", lw=1.2)
        ax.set_ylim(0, 5600)
        ax.text(mes_corte + 0.8, 5450, "el corte que\nencontró el árbol", fontsize=8.5, va="top", color=NEGRO)
        ax.set_title(titulos[k], fontsize=11)
        ax.set_xlabel("mes"); ax.set_xticks([])
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("incidentes por mes"); axes[0].legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    return _leyenda(fig, "cada hipótesis deja una huella distinta. Fíjese en el ROJO después del corte: ¿sube, sigue igual "
                         "o no pasa nada? Con eso en la cabeza, mire ahora la serie real.")


def imputar_y_marcar(valores=(6.25, None, 6.21, None, 6.30, 6.18)):
    """(A) Qué hace 'imputar + marcar' con una columna que tiene vacíos: se rellena con la mediana Y se agrega una
    columna falta_x de 0/1 para que el modelo sepa cuáles filas eran vacías."""
    v = [np.nan if x is None else x for x in valores]
    mediana = float(np.nanmedian(v))
    fig, ax = plt.subplots(figsize=(12, 4.4)); ax.set_xlim(0, 24); ax.set_ylim(0, 8.8); ax.axis("off")
    def tabla(x0, cols, filas, anchos, titulo, color_enc):
        ax.text(x0 + sum(anchos) / 2, 8.3, titulo, ha="center", fontsize=11.5, fontweight="bold", color="#555555")
        xs = np.cumsum([x0] + anchos)
        for j, c in enumerate(cols):
            ax.add_patch(plt.Rectangle((xs[j], 6.6), anchos[j], 0.9, fc=color_enc[j], ec="white", lw=2))
            ax.text(xs[j] + anchos[j] / 2, 7.05, c, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            for i, fila in enumerate(filas):
                yy = 6.6 - (i + 1) * 0.9
                val = fila[j]
                vacio = isinstance(val, float) and np.isnan(val)
                fc = "#FBECEA" if (vacio or (j > 0 and fila[0] == "x")) else ("#F7F8FA" if i % 2 else "white")
                ax.add_patch(plt.Rectangle((xs[j], yy), anchos[j], 0.9, fc=fc, ec="white", lw=2))
                texto = "NaN" if vacio else (f"{val:.2f}" if isinstance(val, float) else str(val))
                ax.text(xs[j] + anchos[j] / 2, yy + 0.45, texto, ha="center", va="center", fontsize=10.5,
                        color=ROJO if vacio else NEGRO, fontweight="bold" if vacio else "normal")
    antes = [[x] for x in v]
    tabla(0.5, ["latitud"], antes, [4.0], "Antes", [AZUL])
    despues = []
    for x in v:
        if np.isnan(x):
            despues.append([mediana, 1])
        else:
            despues.append([x, 0])
    xs_d = 11.0
    tabla(xs_d, ["latitud", "falta_coord"], despues, [4.0, 4.4], "Después: imputar + marcar", [AZUL, ROJO])
    # marcar las celdas imputadas
    for i, x in enumerate(v):
        if np.isnan(x):
            yy = 6.6 - (i + 1) * 0.9
            ax.add_patch(plt.Rectangle((xs_d, yy), 4.0, 0.9, fc="none", ec=ROJO, lw=2))
    ax.annotate("", xy=(xs_d - 0.3, 4.2), xytext=(5.0, 4.2), arrowprops=dict(arrowstyle="->", color=NEGRO, lw=2))
    ax.text(8.0, 4.9, f"mediana = {mediana:.2f}", ha="center", va="bottom", fontsize=10.5, family="monospace", color=NEGRO)
    ax.text(8.0, 3.6, "fillna(mediana)\n+ isna().astype(int)", ha="center", va="top", fontsize=9.5, family="monospace", color=GRIS)
    ax.text(15.2, 0.55, "el modelo ve el valor típico Y sabe que esa fila venía vacía", ha="center", fontsize=10, color=ROJO)
    return _leyenda(fig, "rellenar con la mediana evita botar filas; la columna de 0/1 conserva la información de que el dato "
                         "faltaba (que a veces es lo que de verdad informa, como el N/D de hoy).")


def memorizar_vs_aprender(n=260, semilla=11):
    """(A) Sobreajuste con datos sintéticos de 2 variables: el mismo problema con un árbol corto y uno sin límite.
    Puntos llenos = entrenamiento; puntos huecos = prueba. El árbol sin límite dibuja islas alrededor de cada punto
    de entrenamiento y falla más en los huecos."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    X, y = _datos_sinteticos(n, semilla)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=semilla, stratify=y)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    configs = [(1, "demasiado simple"), (3, "justo"), (None, "memoriza")]
    xx, yy = np.meshgrid(np.linspace(0, 24, 220), np.linspace(20, 100, 220))
    malla = pd.DataFrame({"hora": xx.ravel(), "velocidad": yy.ravel()})
    for ax, (d, nombre) in zip(axes, configs):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
        z = m.predict_proba(malla)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap=CMAP_PROB, alpha=0.5)
        cajas_hojas(ax, m, (0, 24), (20, 100), lw=0.5, alpha=0.4)
        ax.scatter(X_tr["hora"], X_tr["velocidad"], c=y_tr, cmap=CMAP_CLASES, s=18, edgecolor="white", lw=0.4)
        for c, col in ((0, AZUL), (1, ROJO)):
            ax.scatter(X_te["hora"][y_te == c], X_te["velocidad"][y_te == c], s=34, facecolors="none", edgecolors=col, lw=1.4)
        a_tr, a_te = m.score(X_tr, y_tr), m.score(X_te, y_te)
        etiqueta = "sin límite" if d is None else str(d)
        ax.set_title(f"max_depth = {etiqueta} · {nombre}\nentrenamiento {a_tr:.2f} · prueba {a_te:.2f} · {m.get_n_leaves()} hojas", fontsize=10.5)
        ax.set_xlabel("hora")
    axes[0].set_ylabel("velocidad")
    axes[0].scatter([], [], c=NEGRO, s=18, label="punto lleno = entrenamiento")
    axes[0].scatter([], [], facecolors="none", edgecolors=NEGRO, s=34, label="punto hueco = prueba (nunca visto)")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return _leyenda(fig, "a la derecha el fondo dibuja islas alrededor de puntos de entrenamiento sueltos: acierta todos los llenos "
                         "y falla más huecos que el árbol del medio. Memorizar no es aprender.")


def brecha_train_test():
    """(A) La forma típica de la curva de error contra la complejidad: entrenamiento baja siempre; prueba baja, toca
    un mínimo y vuelve a subir. Tres zonas: subajuste, punto justo, sobreajuste. Esquema, no datos reales."""
    c = np.linspace(0.05, 1, 200)
    tr = 0.55 * np.exp(-3.2 * c) + 0.05
    te = 0.55 * np.exp(-3.2 * c) + 0.05 + 0.45 * (c - 0.35).clip(0) ** 1.6
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(c, tr, color=AZUL, lw=2.4, label="error en entrenamiento")
    ax.plot(c, te, color=ROJO, lw=2.4, label="error en prueba")
    i = int(np.argmin(te)); ax.axvline(c[i], color=NEGRO, ls="--", lw=1)
    ax.axvspan(0, 0.2, color="#F7F8FA"); ax.axvspan(0.62, 1, color="#FBECEA", alpha=0.6)
    ax.text(0.1, 0.52, "SUBAJUSTE\nmodelo demasiado simple:\nfalla en los dos", ha="center", fontsize=9.5, color="#555555")
    ax.text(c[i], 0.52, "PUNTO JUSTO\nla prueba es mínima", ha="center", fontsize=9.5, color=NEGRO, fontweight="bold")
    ax.text(0.81, 0.52, "SOBREAJUSTE\nentrenamiento sigue bajando,\nprueba vuelve a subir", ha="center", fontsize=9.5, color=ROJO)
    ax.annotate("", xy=(0.85, tr[int(0.85 * 199)]), xytext=(0.85, te[int(0.85 * 199)]),
                arrowprops=dict(arrowstyle="<->", color=NEGRO, lw=1.4))
    ax.text(0.87, (tr[int(0.85 * 199)] + te[int(0.85 * 199)]) / 2, "la brecha:\nel síntoma", fontsize=9.5, va="center")
    ax.set_xlabel("complejidad del modelo (profundidad del árbol, k, número de variables…)"); ax.set_ylabel("error")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 0.65); ax.legend(loc="lower center", bbox_to_anchor=(0.45, 0.0))
    ax.set_title("Más complejo no es mejor: la curva que se repite en todo el curso")
    return _leyenda(fig, "el modelo que conviene está donde la curva ROJA es mínima, no donde la azul. Si hay brecha grande "
                         "entre las dos, el modelo está memorizando.")


def target_encoding_paso_a_paso(destacar_unicas=False):
    """(A) Target encoding con una tabla de 8 incidentes: cada dirección se reemplaza por el promedio de la y en esa
    dirección. Con destacar_unicas=True se marca el problema: la dirección que aparece una sola vez recibe su propia y."""
    filas = [("Cra 43 # 10-20", 1), ("Cra 43 # 10-20", 0), ("Cra 43 # 10-20", 1), ("Cl 33 # 65-10", 0),
             ("Cl 33 # 65-10", 0), ("Cl 80 # 50-12", 1), ("Cra 65 # 48-05", 0), ("Cl 10 # 42-30", 1)]
    d = pd.DataFrame(filas, columns=["direccion", "con_victimas"])
    conteo = d["direccion"].map(d["direccion"].value_counts())
    d["tasa_dir"] = d.groupby("direccion")["con_victimas"].transform("mean")
    fig, ax = plt.subplots(figsize=(13, 5.4)); ax.set_xlim(0, 26); ax.set_ylim(0, 10.8); ax.axis("off")
    cols = ["direccion", "con_victimas (y)", "veces que aparece", "tasa_dir"]
    anchos = [6.2, 4.2, 4.4, 4.0]; x0 = 0.6; xs = np.cumsum([x0] + anchos)
    encab = [AZUL, ROJO, GRIS, MODULO[1]]
    for j, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((xs[j], 9.0), anchos[j], 0.95, fc=encab[j], ec="white", lw=2))
        ax.text(xs[j] + anchos[j] / 2, 9.47, c, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    for i, (_, f) in enumerate(d.iterrows()):
        yy = 9.0 - (i + 1) * 0.95
        unica = conteo.iloc[i] == 1
        resaltar = destacar_unicas and unica
        valores = [f["direccion"], str(int(f["con_victimas"])), str(int(conteo.iloc[i])), f"{f['tasa_dir']:.2f}".replace(".", ",")]
        for j, v in enumerate(valores):
            fc = "#FBECEA" if resaltar else ("#F7F8FA" if i % 2 else "white")
            ax.add_patch(plt.Rectangle((xs[j], yy), anchos[j], 0.95, fc=fc, ec="white", lw=2))
            ax.text(xs[j] + anchos[j] / 2, yy + 0.47, v, ha="center", va="center", fontsize=10.5,
                    color=ROJO if (resaltar and j in (1, 3)) else NEGRO, fontweight="bold" if (resaltar and j in (1, 3)) else "normal")
        if resaltar:
            ax.add_patch(plt.Rectangle((xs[0], yy), sum(anchos), 0.95, fc="none", ec=ROJO, lw=2))
    xt = xs[-1] + 0.9
    if not destacar_unicas:
        ax.text(xt, 8.4, "La idea", fontsize=12.5, fontweight="bold", color=NEGRO)
        ax.text(xt, 7.6, "Para cada dirección:\ntasa = promedio de la y\nde sus incidentes", fontsize=10.5, va="top", color=NEGRO)
        ax.text(xt, 5.3, "Cra 43: (1 + 0 + 1) / 3 = 0,67\nCl 33: (0 + 0) / 2 = 0,00", fontsize=10, va="top", family="monospace", color="#555555")
        ax.text(xt, 3.4, "Una columna numérica\nen vez de 30 000 dummies.\nSuena razonable…", fontsize=10.5, va="top", color=NEGRO)
        ax.text(xt, 1.2, 'df.groupby("direccion")["con_victimas"].mean()', fontsize=8.5, family="monospace", color=GRIS)
        ley = "cada dirección se resume en un número: la fracción de sus incidentes que tuvo víctimas. Es una técnica real (target encoding)."
    else:
        ax.text(xt, 8.4, "El problema", fontsize=12.5, fontweight="bold", color=ROJO)
        ax.text(xt, 7.6, "Si la dirección aparece\nUNA sola vez, el promedio\nes de un solo incidente:", fontsize=10.5, va="top", color=NEGRO)
        ax.text(xt, 5.1, "tasa_dir  =  y", fontsize=15, fontweight="bold", color=ROJO, family="monospace")
        ax.text(xt, 4.1, "La columna nueva ES la\nrespuesta, copiada.\nY la mitad de las direcciones\nde Medellín aparece una vez.", fontsize=10.5, va="top", color=NEGRO)
        ley = "en las filas rojas tasa_dir coincide exactamente con la y. El modelo no aprende sobre calles: lee la respuesta."
    return _leyenda(fig, ley)


def fit_solo_con_train():
    """(A) La regla de oro del preprocesamiento: todo lo que se APRENDE de los datos (media, desviación, mediana, tasa por
    categoría) se aprende con .fit() SOLO sobre entrenamiento, y luego se APLICA con .transform() a las dos partes."""
    fig, ax = plt.subplots(figsize=(12.5, 4.8)); ax.set_xlim(0, 25); ax.set_ylim(0, 9.6); ax.axis("off")
    def caja(x, y, w, h, t, fc, ec, color="white", fs=11, peso="bold"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=color, fontsize=fs, fontweight=peso, linespacing=1.35)
    caja(0.5, 5.6, 7.4, 2.6, "ENTRENAMIENTO\nX_train, y_train", AZUL, AZUL)
    caja(0.5, 1.0, 7.4, 2.6, "PRUEBA\nX_test, y_test", ROJO, ROJO)
    caja(10.3, 5.3, 6.6, 3.2, "lo aprendido\n\nmedia y desviación\nmediana · tasa por dirección", "white", NEGRO, color=NEGRO, fs=10, peso="normal")
    ax.text(13.6, 8.85, ".fit(X_train)", ha="center", fontsize=11, family="monospace", color=AZUL, fontweight="bold")
    caja(19.4, 5.6, 5.2, 2.6, "X_train\ntransformado", "#EEF3FA", AZUL, color=AZUL)
    caja(19.4, 1.0, 5.2, 2.6, "X_test\ntransformado", "#FBECEA", ROJO, color=ROJO)
    ax.annotate("", xy=(10.3, 6.9), xytext=(7.9, 6.9), arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=2.4, mutation_scale=18))
    ax.text(9.1, 7.3, "aprende", ha="center", fontsize=9.5, color=AZUL)
    ax.annotate("", xy=(19.4, 6.9), xytext=(16.9, 6.9), arrowprops=dict(arrowstyle="-|>", color=NEGRO, lw=2, mutation_scale=18))
    ax.text(18.15, 7.3, ".transform()", ha="center", fontsize=9.5, family="monospace", color=NEGRO)
    ax.annotate("", xy=(19.4, 2.3), xytext=(13.6, 5.3), arrowprops=dict(arrowstyle="-|>", color=NEGRO, lw=2, mutation_scale=18,
                                                                        connectionstyle="arc3,rad=-0.15"))
    ax.text(15.0, 2.7, ".transform()", ha="center", fontsize=9.5, family="monospace", color=NEGRO)
    ax.annotate("", xy=(10.3, 5.9), xytext=(7.9, 2.3), arrowprops=dict(arrowstyle="-|>", color=ROJO, lw=2.2, mutation_scale=18, ls="--"))
    ax.text(8.5, 4.6, "NUNCA", ha="center", fontsize=11, color=ROJO, fontweight="bold", rotation=52,
            bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none"))
    ax.text(12.5, 0.4, "La prueba se transforma con lo aprendido en entrenamiento; jamás aporta a lo aprendido.",
            ha="center", fontsize=10.5, color=NEGRO)
    return _leyenda(fig, "la flecha roja punteada es la que nunca debe existir. Si la prueba entra al .fit() de cualquier cosa "
                         "—un escalador, una tasa, un modelo—, el examen deja de ser examen.")
