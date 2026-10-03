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
arbol_reglas, mapa_hora_dia, arbol_profundidades.
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
    ax.scatter(X[xcol], X[ycol], c=y, cmap=CMAP_CLASES, s=14, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(titulo)


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
def arbol_reglas(modelo, nombres, max_depth=2, class_names=("solo daños", "con víctimas"), figsize=(13, 5.5)):
    """(B) El árbol entrenado con los datos del curso, dibujado. Se recorta a `max_depth`
    niveles para que se lea."""
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(modelo, feature_names=list(nombres), class_names=list(class_names), max_depth=max_depth,
              filled=True, rounded=True, impurity=False, proportion=True, fontsize=9, ax=ax)
    ax.set_title(f"Las primeras {max_depth} preguntas que aprendió el árbol (de {modelo.get_depth()} niveles)")
    return _leyenda(fig, "Léalo de arriba abajo: cada caja es una pregunta; 'value' es la proporción de cada clase "
                         "que llega ahí y 'samples' qué fracción de los datos pasa por esa caja.")


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
