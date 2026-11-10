import json, warnings, numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
import lab7_neurogenetico as L
out = {}
for nombre, carga, n_ga in [("fifa", L.cargar_fifa, 5000), ("tmdb", L.cargar_tmdb, None)]:
    df, X, y = carga(); R = json.load(open(f"salidas/resultados_{nombre}.json"))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=L.SEED)
    Xga, yga = Xtr, ytr
    if n_ga: Xga, _, yga, _ = train_test_split(Xtr, ytr, train_size=n_ga, stratify=ytr, random_state=L.SEED)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=L.SEED)
    B = L.BASE
    r = cross_validate(L.crear_mlp(B["capas"], B["lr"], B["act"], B["alpha"], B["batch"], max_iter=150), Xga, yga, cv=cv,
                       scoring=["accuracy", "neg_log_loss"], n_jobs=2)
    acc, loss = r["test_accuracy"].mean(), -r["test_neg_log_loss"].mean()
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=L.SEED)
    otros = {}
    for et, m in [("regresion_logistica", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                  ("gradient_boosting", HistGradientBoostingClassifier(random_state=L.SEED))]:
        rr = cross_validate(m, X, y, cv=rcv, scoring=["accuracy", "neg_log_loss"], n_jobs=2)
        otros[et] = {"accuracy_media": round(rr["test_accuracy"].mean(), 4), "accuracy_std": round(rr["test_accuracy"].std(), 4),
                     "loss_media": round(-rr["test_neg_log_loss"].mean(), 4)}
    a = np.array(R["cv_repetida"]["base"]["accuracy_folds"]); b = np.array(R["cv_repetida"]["optimizada"]["accuracy_folds"])
    t, p = stats.ttest_rel(b, a)
    out[nombre] = {"aptitud_base_criterio_ga": {"aptitud": round(acc - L.LAMBDA * loss, 4), "cv_accuracy": round(acc, 4), "cv_loss": round(loss, 4)},
                   "aptitud_optima_ga": R["opt"]["aptitud_ga"], "aptitud_gen1_mejor": R["historial_ga"][0]["mejor"],
                   "otros_modelos": otros, "prueba_t_pareada": {"diferencia_media": round(float((b - a).mean()), 4), "t": round(float(t), 3), "p": round(float(p), 4)}}
    o = out[nombre]
    print(f"\n{nombre.upper()} - comparacion con otros modelos")
    print(f"  Regresion logistica: {o['otros_modelos']['regresion_logistica']['accuracy_media']*100:.1f} %")
    print(f"  Gradient boosting:   {o['otros_modelos']['gradient_boosting']['accuracy_media']*100:.1f} %")
    print(f"  Diferencia entre las dos redes: {'no significativa' if o['prueba_t_pareada']['p'] > 0.05 else 'significativa'} (p = {o['prueba_t_pareada']['p']:.2f})")
json.dump(out, open("salidas/analisis_critico.json", "w"), indent=2)

# Gráfica
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
nombres = ["Regresión\nlogística", "Gradient\nboosting", "Red\ninicial", "Red del\nAG"]
valores = {n: [out[n]["otros_modelos"]["regresion_logistica"]["accuracy_media"], out[n]["otros_modelos"]["gradient_boosting"]["accuracy_media"],
               json.load(open(f"salidas/resultados_{n}.json"))["cv_repetida"]["base"]["accuracy_media"],
               json.load(open(f"salidas/resultados_{n}.json"))["cv_repetida"]["optimizada"]["accuracy_media"]] for n in ["fifa", "tmdb"]}
fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
x = np.arange(4)
for desp, n, c, et in [(-0.2, "fifa", L.AZUL, "FIFA 23"), (0.2, "tmdb", "#AF7B51", "TMDB")]:
    barras = ax.bar(x + desp, [v * 100 for v in valores[n]], 0.4, label=et, color=c)
    for b in barras: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=8); ax.set_ylim(0, 100); ax.set_ylabel("% de aciertos", fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("Todos los modelos llegan al mismo techo", fontsize=10, pad=12)
fig.tight_layout(); fig.savefig("salidas/modelos.png"); plt.close(fig)
