
import json, os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate,RepeatedStratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
DATOS = "datasets"
OUT = "salidas"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
AZUL, NARANJA, VERDE, GRIS = "#2a6f97", "#e07a5f", "#81b29a", "#8d99ae"

#CARGA DE DATOS
def cargar_fifa():
    df = pd.read_csv(f"{DATOS}/fifa23_jugadores.csv")
    y = df["posicion"]
    X = df.drop(columns=["id_jugador", "nombre", "posicion_original", "posicion"]).copy()
    X["pie_izquierdo"] = (X.pop("pie_preferido") == "Izquierdo").astype(int)
    return df, X, y

GENEROS_ES = {"Drama": "drama", "Comedy": "comedia", "Thriller": "suspenso", "Action": "accion",
              "Romance": "romance", "Adventure": "aventura", "Crime": "crimen", "Science Fiction": "ciencia_ficcion",
              "Horror": "terror", "Family": "familiar", "Fantasy": "fantasia", "Mystery": "misterio",
              "Animation": "animacion", "History": "historia", "Music": "musica", "War": "belica",
              "Documentary": "documental", "Western": "western", "Foreign": "extranjera", "TV Movie": "pelicula_tv"}

def cargar_tmdb():
    df = pd.read_csv(f"{DATOS}/tmdb_peliculas.csv")
    df = df[(df.presupuesto >= 1e5) & (df.ingresos >= 1e5)].dropna(subset=["duracion_min", "anio_estreno"]).copy()
    df["retorno"] = df.ingresos / df.presupuesto
    y = pd.Series(np.where(df.retorno >= 2, "exito", "no_exito"), index=df.index, name="clase")
    X = pd.DataFrame(index=df.index)
    X["log_presupuesto"] = np.log10(df.presupuesto)
    for c in ["duracion_min", "anio_estreno", "mes_estreno", "num_productoras", "num_paises", "num_idiomas",
              "num_palabras_clave", "tiene_pagina_web", "tiene_eslogan", "num_reparto", "num_equipo"]:
        X[c] = df[c]
    X["idioma_ingles"] = (df.idioma_original == "en").astype(int)
    gen = df.generos.fillna("").str.split("|")
    for en, es in GENEROS_ES.items():
        X["genero_" + es] = gen.apply(lambda g: int(en in g))
    X = X.loc[:, X.std() > 0]
    return df, X, y

#ANALISIS EXPLORATORIO
COLORES = {"Portero": GRIS, "Defensa": AZUL, "Mediocampista": VERDE, "Delantero": NARANJA,
           "exito": VERDE, "no_exito": NARANJA}

def eda(nombre, df, X, y):
    res = {"filas_originales": int(len(pd.read_csv(f"{DATOS}/{'fifa23_jugadores' if nombre=='fifa' else 'tmdb_peliculas'}.csv"))),
           "filas_modelo": int(X.shape[0]), "variables_modelo": int(X.shape[1]),
           "nulos": int(X.isna().sum().sum()), "clases": y.value_counts().to_dict()}
    X.assign(clase=y.values).to_csv(f"{OUT}/{nombre}_dataset_modelo.csv", index=False)
    X.describe().T[["mean", "std", "min", "max"]].round(3).to_csv(f"{OUT}/{nombre}_estadisticos.csv")
    def eta(col):
        m = col.mean(); ss_t = ((col - m) ** 2).sum()
        ss_b = sum(len(g) * (g.mean() - m) ** 2 for _, g in col.groupby(y.values))
        return np.sqrt(ss_b / ss_t) if ss_t > 0 else 0
    rel = X.apply(eta).sort_values(ascending=False)
    res["top_variables"] = rel.head(8).round(3).to_dict()
    rel.round(4).to_csv(f"{OUT}/{nombre}_relacion_clase.csv")

    vc = y.value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.bar(vc.index, vc.values, color=[COLORES[c] for c in vc.index])
    for i, v in enumerate(vc.values): ax.text(i, v, f"{v:,}\n({v/len(y):.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Registros"); ax.set_title(f"Distribución de clases – {nombre.upper()}"); ax.set_ylim(0, vc.max() * 1.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{nombre}_clases.png"); plt.close(fig)

    top = rel.head(12).index
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(X[top].corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f",
                annot_kws={"size": 6}, ax=ax, cbar_kws={"shrink": .7})
    ax.set_title(f"Correlación entre las 12 variables más relacionadas con la clase – {nombre.upper()}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/{nombre}_correlacion.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, col in zip(axs, rel.head(4).index):
        sns.boxplot(x=y.values, y=X[col].values, ax=a, hue=y.values, order=list(vc.index), legend=False,
                    palette=COLORES, fliersize=1)
        a.set_title(col, fontsize=9); a.tick_params(axis="x", labelsize=7, rotation=20)
    fig.suptitle(f"Variables que más separan las clases – {nombre.upper()}")
    fig.tight_layout(); fig.savefig(f"{OUT}/{nombre}_boxplots.png"); plt.close(fig)

    if nombre == "tmdb":
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in ["no_exito", "exito"]:
            d = df[y == c]
            ax.scatter(d.presupuesto, d.ingresos, s=5, alpha=.4, color=COLORES[c], label=c)
        lim = [1e5, 4e9]; ax.plot(lim, [2 * v for v in lim], "k--", lw=1, label="ingresos = 2 × presupuesto")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Presupuesto (USD)"); ax.set_ylabel("Ingresos (USD)")
        ax.set_title("Presupuesto vs ingresos – TMDB"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/tmdb_presupuesto_ingresos.png"); plt.close(fig)
        gen_cols = [c for c in X.columns if c.startswith("genero_")]
        tasa = pd.Series({c.replace("genero_", ""): (y[X[c] == 1] == "exito").mean()
                          for c in gen_cols if X[c].sum() >= 30}).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.barh(tasa.index, tasa.values, color=AZUL)
        ax.axvline((y == "exito").mean(), color=NARANJA, ls="--", lw=1, label="promedio general")
        ax.set_xlabel("Proporción de éxitos"); ax.set_title("Tasa de éxito por género (≥30 películas)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/tmdb_exito_genero.png"); plt.close(fig)
        res["tasa_exito_genero"] = tasa.round(3).to_dict()
        res["retorno_mediana"] = round(float(df.retorno.median()), 3)
        res["anios"] = [int(df.anio_estreno.min()), int(df.anio_estreno.max())]
        res["presupuesto_mediana_musd"] = round(float(df.presupuesto.median() / 1e6), 1)
    if nombre == "fifa":
        res["valoracion_media"] = round(float(X.valoracion_general.mean()), 2)
        res["edad_media"] = round(float(X.edad.mean()), 2)
        res["posiciones_originales"] = df.posicion_original.value_counts().to_dict()
    return res

#RED NEURONAL
def crear_mlp(capas, lr, act="relu", alpha=1e-4, batch=32, max_iter=300):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=tuple(capas), activation=act, solver="adam", learning_rate_init=lr,
        alpha=alpha, batch_size=batch, max_iter=max_iter, early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1, random_state=SEED))

def evaluar(h, Xtr, ytr, Xte, yte):
    modelo = crear_mlp(h["capas"], h["lr"], h["act"], h["alpha"], h["batch"])
    t = time.time(); modelo.fit(Xtr, ytr); seg = time.time() - t
    p, pr, mlp = modelo.predict(Xte), modelo.predict_proba(Xte), modelo[-1]
    return {"accuracy": round(accuracy_score(yte, p), 4), "f1_macro": round(f1_score(yte, p, average="macro"), 4),
            "loss_prueba": round(log_loss(yte, pr, labels=mlp.classes_), 4),
            "loss_entrenamiento": round(float(mlp.loss_), 4), "epocas": int(mlp.n_iter_),
            "parametros": int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
            "segundos": round(seg, 2), "curva_loss": [float(v) for v in mlp.loss_curve_],
            "curva_val": [float(v) for v in mlp.validation_scores_],
            "matriz_confusion": confusion_matrix(yte, p, labels=mlp.classes_).tolist(), "clases": list(mlp.classes_)}

# ALGORITMO GENÉTICO
ACTS = ["relu", "tanh", "logistic"]
BATCHES = [16, 32, 64, 128]
LIM = np.array([[1, 3], [4, 128], [4, 128], [4, 128], [-4, -1], [0, 2], [-6, -1], [0, 3]], float)
ENTEROS = [0, 1, 2, 3, 5, 7]
LAMBDA = 0.5 

def decodificar(c):
    return {"capas": [int(c[1]), int(c[2]), int(c[3])][:int(c[0])], "lr": float(10 ** c[4]),
            "act": ACTS[int(c[5])], "alpha": float(10 ** c[6]), "batch": BATCHES[int(c[7])]}

def reparar(c):
    c = np.clip(c, LIM[:, 0], LIM[:, 1]); c[ENTEROS] = np.round(c[ENTEROS]); return c

def aleatorio(rng):
    return reparar(rng.uniform(LIM[:, 0], LIM[:, 1]))

def algoritmo_genetico(X, y, pob=12, gen=10, p_cruce=0.9, p_mut=0.2, elite=2, k_torneo=3, seed=SEED):
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cache = {}
    def aptitud(c):
        k = tuple(np.round(c, 3))
        if k not in cache:
            h = decodificar(c)
            r = cross_validate(crear_mlp(h["capas"], h["lr"], h["act"], h["alpha"], h["batch"], max_iter=150),
                               X, y, cv=cv, scoring=["accuracy", "neg_log_loss"], n_jobs=2)
            acc, loss = r["test_accuracy"].mean(), -r["test_neg_log_loss"].mean()
            cache[k] = (acc - LAMBDA * loss, acc, loss)
        return cache[k][0]
    P = [aleatorio(rng) for _ in range(pob)]
    hist = []
    for g in range(gen):
        fit = np.array([aptitud(c) for c in P]); o = np.argsort(-fit)
        mk = tuple(np.round(P[o[0]], 3))
        hist.append({"generacion": g + 1, "mejor": round(float(fit.max()), 4), "promedio": round(float(fit.mean()), 4),
                     "peor": round(float(fit.min()), 4), "mejor_accuracy": round(float(cache[mk][1]), 4),
                     "mejor_loss": round(float(cache[mk][2]), 4), "mejor_individuo": decodificar(P[o[0]])})
        print(f"  gen {g+1:2d} aptitud mejor={fit.max():.4f} prom={fit.mean():.4f} | {decodificar(P[o[0]])}", flush=True)
        if g == gen - 1: break
        nueva = [P[i].copy() for i in o[:elite]]                                  
        def torneo():                                                              
            idx = rng.choice(pob, k_torneo, replace=False); return P[idx[np.argmax(fit[idx])]]
        while len(nueva) < pob:
            a, b = torneo().copy(), torneo().copy()
            if rng.random() < p_cruce:                                             
                m = rng.random(len(a)) < 0.5; a, b = np.where(m, a, b), np.where(m, b, a)
            for h in (a, b):                                                       
                mm = rng.random(len(h)) < p_mut
                h[mm] += rng.normal(0, 0.25, mm.sum()) * (LIM[mm, 1] - LIM[mm, 0])
                nueva.append(reparar(h))
        P = nueva[:pob]
    return decodificar(P[o[0]]), {"aptitud": round(cache[mk][0], 4), "cv_accuracy": round(cache[mk][1], 4),
                                  "cv_loss": round(cache[mk][2], 4)}, hist, len(cache)

# FLUJO POR DATASET
BASE = {"capas": [8], "lr": 0.01, "act": "logistic", "alpha": 1e-4, "batch": 32}

def correr(nombre, df, X, y, n_ga=None):
    print(f"\n=== {nombre} ===", flush=True)
    info = eda(nombre, df, X, y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    base = evaluar(BASE, Xtr, ytr, Xte, yte)
    print("  base:", {k: base[k] for k in ["accuracy", "f1_macro", "loss_prueba"]}, flush=True)
    Xga, yga = Xtr, ytr
    if n_ga and len(Xtr) > n_ga:
        Xga, _, yga, _ = train_test_split(Xtr, ytr, train_size=n_ga, stratify=ytr, random_state=SEED)
    t = time.time(); mejor, apt, hist, evals = algoritmo_genetico(Xga, yga); t_ga = time.time() - t
    opt = evaluar(mejor, Xtr, ytr, Xte, yte); opt["aptitud_ga"] = apt
    print("  optimizada:", mejor, {k: opt[k] for k in ["accuracy", "f1_macro", "loss_prueba"]}, flush=True)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    robusto = {}
    for et, h in [("base", BASE), ("optimizada", mejor)]:
        r = cross_validate(crear_mlp(h["capas"], h["lr"], h["act"], h["alpha"], h["batch"]), X, y, cv=rcv,
                           scoring=["accuracy", "f1_macro", "neg_log_loss"], n_jobs=2)
        robusto[et] = {"accuracy_media": round(float(r["test_accuracy"].mean()), 4),
                       "accuracy_std": round(float(r["test_accuracy"].std()), 4),
                       "f1_media": round(float(r["test_f1_macro"].mean()), 4),
                       "loss_media": round(float(-r["test_neg_log_loss"].mean()), 4),
                       "loss_std": round(float(r["test_neg_log_loss"].std()), 4),
                       "accuracy_folds": [round(float(v), 4) for v in r["test_accuracy"]]}
    print("  robusto:", {k: {kk: v[kk] for kk in ["accuracy_media", "loss_media"]} for k, v in robusto.items()}, flush=True)

    # figuras de resultados
    g = [h["generacion"] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(g, [h["mejor"] for h in hist], "o-", color=AZUL, label="Mejor individuo")
    ax.plot(g, [h["promedio"] for h in hist], "s--", color=NARANJA, label="Promedio de la población")
    ax.set_xlabel("Generación"); ax.set_ylabel("Aptitud = accuracy − 0.5·pérdida")
    ax.set_title(f"Evolución del algoritmo genético – {nombre.upper()}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{nombre}_evolucion_ga.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(base["curva_loss"], color=NARANJA, label="Base"); axs[0].plot(opt["curva_loss"], color=AZUL, label="Optimizada (AG)")
    axs[0].set_xlabel("Época"); axs[0].set_ylabel("Pérdida de entrenamiento (log-loss)"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(base["curva_val"], color=NARANJA, label="Base"); axs[1].plot(opt["curva_val"], color=AZUL, label="Optimizada (AG)")
    axs[1].set_xlabel("Época"); axs[1].set_ylabel("Accuracy en validación interna"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.suptitle(f"Curvas de entrenamiento – {nombre.upper()}"); fig.tight_layout()
    fig.savefig(f"{OUT}/{nombre}_curvas.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
    for a, r, t_ in zip(axs, [base, opt], ["Base", "Optimizada (AG)"]):
        sns.heatmap(r["matriz_confusion"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=r["clases"], yticklabels=r["clases"], annot_kws={"size": 8})
        a.set_title(f"{t_} – accuracy {r['accuracy']:.1%}"); a.set_xlabel("Predicho"); a.set_ylabel("Real")
        a.tick_params(labelsize=7)
    fig.suptitle(f"Matrices de confusión (conjunto de prueba) – {nombre.upper()}"); fig.tight_layout()
    fig.savefig(f"{OUT}/{nombre}_confusion.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    et = ["Base", "Optimizada"]; cs = [NARANJA, AZUL]
    axs[0].bar(et, [robusto["base"]["accuracy_media"], robusto["optimizada"]["accuracy_media"]],
               yerr=[robusto["base"]["accuracy_std"], robusto["optimizada"]["accuracy_std"]], color=cs, capsize=6)
    axs[1].bar(et, [robusto["base"]["loss_media"], robusto["optimizada"]["loss_media"]],
               yerr=[robusto["base"]["loss_std"], robusto["optimizada"]["loss_std"]], color=cs, capsize=6)
    lo = min(robusto["base"]["accuracy_media"], robusto["optimizada"]["accuracy_media"])
    axs[0].set_ylim(max(0, lo - 0.1), 1); axs[0].set_title("Accuracy (más alto = mejor)")
    axs[1].set_title("Pérdida log-loss (más bajo = mejor)")
    for a, k in zip(axs, ["accuracy_media", "loss_media"]):
        for i, e in enumerate(["base", "optimizada"]):
            a.text(i, robusto[e][k], f"{robusto[e][k]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(f"Comparación con validación cruzada repetida (5×2) – {nombre.upper()}"); fig.tight_layout()
    fig.savefig(f"{OUT}/{nombre}_comparacion.png"); plt.close(fig)
    for r in (base, opt): r.pop("curva_loss"); r.pop("curva_val")
    return {"eda": info, "base_hiper": BASE, "base": base, "opt_hiper": mejor, "opt": opt, "historial_ga": hist,
            "cv_repetida": robusto, "evaluaciones_unicas": evals, "segundos_ga": round(t_ga, 1),
            "n_train": len(Xtr), "n_test": len(Xte), "n_ga": len(Xga)}

if __name__ == "__main__":
    cual = sys.argv[1] if len(sys.argv) > 1 else "ambos"
    if cual in ("fifa", "ambos"):
        json.dump(correr("fifa", *cargar_fifa(), n_ga=5000), open(f"{OUT}/resultados_fifa.json", "w"), indent=2, ensure_ascii=False)
    if cual in ("tmdb", "ambos"):
        json.dump(correr("tmdb", *cargar_tmdb()), open(f"{OUT}/resultados_tmdb.json", "w"), indent=2, ensure_ascii=False)
    print("\nListo ->", OUT)
