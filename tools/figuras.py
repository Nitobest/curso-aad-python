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
