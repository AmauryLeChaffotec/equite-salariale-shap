"""Analyse d'équité salariale : Gradient Boosting (XGBoost / LightGBM) + valeurs SHAP.

Lancer :  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

st.set_page_config(page_title="Équité salariale · GBM + SHAP", page_icon="⚖️", layout="wide")

EXAMPLE = Path(__file__).parent / "data" / "employes_exemple.csv"
COLORS = {"a": "#E07A1F", "b": "#2A7F9E"}  # groupe étudié / groupe de référence
GENRE_HINTS = ("genre", "sexe", "gender", "sex")
ID_HINTS = ("id", "matricule", "employe", "employee")
TARGET_HINTS = ("salaire", "salary", "remuneration", "rémunération")


# --------------------------------------------------------------------------- données
@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source, sep=None, engine="python")


def guess(columns, hints, default=0):
    for i, c in enumerate(columns):
        if any(h in c.lower() for h in hints):
            return i
    return default


# --------------------------------------------------------------------------- modèle
def make_model(name: str, n_estimators: int, max_depth: int, learning_rate: float):
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1,
        )
    return LGBMRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
        random_state=42, verbose=-1, n_jobs=-1,
    )


@st.cache_data(show_spinner="Entraînement du modèle et calcul des valeurs SHAP…")
def run_model(df: pd.DataFrame, features: tuple, target: str, model_name: str,
              n_estimators: int, max_depth: int, learning_rate: float):
    """Validation croisée 5 plis : prédiction et SHAP de chaque salarié viennent d'un modèle
    qui ne l'a PAS vu à l'entraînement (sinon le modèle « apprendrait » les écarts individuels)."""
    X = df[list(features)]
    y = df[target].to_numpy(dtype=float)

    # encodage : numériques tels quels, catégorielles en one-hot (SHAP re-sommé par variable d'origine)
    parts, groups, col = [], {}, 0
    for f in features:
        if pd.api.types.is_numeric_dtype(X[f]):
            block = X[[f]].astype(float)
        else:
            block = pd.get_dummies(X[f].astype(str), dtype=float)
        groups[f] = list(range(col, col + block.shape[1]))
        col += block.shape[1]
        parts.append(block)
    Xe = pd.concat(parts, axis=1)
    Xe.columns = [f"x{i}" for i in range(Xe.shape[1])]  # noms sûrs pour LightGBM

    n = len(df)
    pred = np.zeros(n)
    base = np.zeros(n)
    sv = np.zeros((n, len(features)))
    for train, test in KFold(5, shuffle=True, random_state=42).split(Xe):
        model = make_model(model_name, n_estimators, max_depth, learning_rate)
        model.fit(Xe.iloc[train], y[train])
        explainer = shap.TreeExplainer(model)
        raw = np.asarray(explainer.shap_values(Xe.iloc[test]))
        base[test] = float(np.ravel(explainer.expected_value)[0])
        for j, f in enumerate(features):
            sv[test, j] = raw[:, groups[f]].sum(axis=1)
        pred[test] = model.predict(Xe.iloc[test])

    metrics = {"R²": r2_score(y, pred), "MAE (€)": mean_absolute_error(y, pred),
               "MAE (%)": float(np.mean(np.abs(y - pred) / y)) * 100}
    shap_df = pd.DataFrame(sv, columns=list(features), index=df.index)
    return pred, base, shap_df, metrics


# --------------------------------------------------------------------------- barre latérale
st.sidebar.title("⚖️ Équité salariale")
st.sidebar.caption("Gradient Boosting + valeurs SHAP")

upload = st.sidebar.file_uploader("Fichier RH (CSV)", type=["csv"])
if upload is not None:
    data = load_csv(upload)
elif EXAMPLE.exists():
    data = load_csv(EXAMPLE)
    st.sidebar.info("Fichier d'exemple chargé (données fictives).")
else:
    st.error("Aucun fichier d'exemple trouvé. Lancez `python generate_data.py` ou importez un CSV.")
    st.stop()

cols = list(data.columns)
num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]

st.sidebar.subheader("Colonnes")
target = st.sidebar.selectbox("Salaire (cible)", num_cols, index=num_cols.index(
    next((c for c in num_cols if any(h in c.lower() for h in TARGET_HINTS)), num_cols[-1])))
gender_col = st.sidebar.selectbox("Genre (jamais utilisé par le modèle)", cols,
                                  index=guess(cols, GENRE_HINTS))
id_guess = guess(cols, ID_HINTS, default=-1)
id_col = st.sidebar.selectbox("Identifiant", ["(index)"] + cols, index=id_guess + 1)

candidates = [c for c in cols if c not in (target, gender_col) and c != id_col]
features = st.sidebar.multiselect("Variables objectives", candidates, default=candidates)
if not features:
    st.warning("Sélectionnez au moins une variable objective.")
    st.stop()

st.sidebar.subheader("Modèle")
model_name = st.sidebar.radio("Algorithme", ["XGBoost", "LightGBM"], horizontal=True)
with st.sidebar.expander("Hyperparamètres"):
    n_estimators = st.slider("Nombre d'arbres", 100, 800, 300, 50)
    max_depth = st.slider("Profondeur max", 2, 8, 4)
    learning_rate = st.select_slider("Learning rate", [0.02, 0.05, 0.1, 0.2], value=0.05)

groups_available = sorted(data[gender_col].dropna().astype(str).unique())
if len(groups_available) < 2:
    st.error("La colonne de genre doit contenir au moins deux valeurs.")
    st.stop()
st.sidebar.subheader("Comparaison")
default_a = next((i for i, g in enumerate(groups_available) if g.lower() in ("f", "femme", "femmes", "female", "w")), 0)
group_a = st.sidebar.selectbox("Groupe étudié", groups_available, index=default_a)
group_b = st.sidebar.selectbox("Groupe de référence", [g for g in groups_available if g != group_a])

# --------------------------------------------------------------------------- calculs
df = data.dropna(subset=[target, gender_col] + features).reset_index(drop=True)
if len(df) < 100:
    st.error("Au moins 100 lignes complètes sont nécessaires pour un modèle fiable.")
    st.stop()
dropped = len(data) - len(df)

pred, base, shap_df, metrics = run_model(df, tuple(features), target, model_name,
                                         n_estimators, max_depth, learning_rate)
res = pd.DataFrame({
    "id": df[id_col].astype(str) if id_col != "(index)" else df.index.astype(str),
    "genre": df[gender_col].astype(str),
    "salaire": df[target].astype(float),
    "predit": pred,
})
res["ecart"] = res["salaire"] - res["predit"]
res["ecart_pct"] = res["ecart"] / res["predit"] * 100
mask_a, mask_b = res["genre"] == group_a, res["genre"] == group_b
label = {group_a: f"{group_a}", group_b: f"{group_b}"}

# --------------------------------------------------------------------------- en-tête
st.title("Équité salariale : Gradient Boosting + SHAP")
st.caption(
    f"{len(df):,} salariés · {len(features)} variables objectives · modèle {model_name} · "
    f"le genre n'est **pas** une variable du modèle."
    + (f" · {dropped} lignes incomplètes ignorées." if dropped else "")
)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "⚖️ Écart de genre", "🔍 Analyse individuelle", "🚨 Alertes & budget"])

# --------------------------------------------------------------------------- onglet 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (validation croisée)", f"{metrics['R²']:.3f}")
    c2.metric("Erreur moyenne", f"{metrics['MAE (€)']:,.0f} €".replace(",", " "))
    c3.metric("Erreur moyenne (%)", f"{metrics['MAE (%)']:.1f} %")
    c4.metric("Salaire moyen", f"{res['salaire'].mean():,.0f} €".replace(",", " "))
    if metrics["R²"] < 0.6:
        st.warning("Le modèle explique peu les salaires : les écarts « inexpliqués » seront peu fiables. "
                   "Ajoutez des variables objectives (poste, niveau, ancienneté…).")

    left, right = st.columns(2)
    with left:
        st.subheader("Importance globale (SHAP)")
        imp = shap_df.abs().mean().sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2A7F9E",
                               hovertemplate="%{y}: %{x:,.0f} €<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Impact moyen |SHAP| (€)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Impact moyen de chaque variable sur la prédiction de salaire, en euros.")
    with right:
        st.subheader("Prédit vs réel")
        samp = res.sample(min(len(res), 1500), random_state=0)
        fig = px.scatter(samp, x="predit", y="salaire", color="genre", opacity=0.6,
                         color_discrete_map={group_a: COLORS["a"], group_b: COLORS["b"]},
                         labels={"predit": "Salaire prédit (€)", "salaire": "Salaire réel (€)", "genre": "Genre"})
        lim = [min(samp.predit.min(), samp.salaire.min()), max(samp.predit.max(), samp.salaire.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"),
                                 name="Parité", hoverinfo="skip"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Contribution d'une variable (dépendance SHAP)")
    feat = st.selectbox("Variable", features, index=0)
    dep = pd.DataFrame({"valeur": df[feat], "shap": shap_df[feat], "genre": res["genre"]})
    cmap = {group_a: COLORS["a"], group_b: COLORS["b"]}
    if pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() > 8:
        fig = px.scatter(dep, x="valeur", y="shap", color="genre", opacity=0.5, color_discrete_map=cmap)
    else:
        fig = px.box(dep.assign(valeur=dep["valeur"].astype(str)), x="valeur", y="shap", color="genre",
                     color_discrete_map=cmap)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title=feat, yaxis_title="Contribution au salaire (€)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Si, pour une même valeur, une courbe de genre se détache de l'autre, le modèle valorise "
               "différemment cette variable selon le groupe (souvent un effet de proxy ou d'interaction).")

    with st.expander("Aperçu des données"):
        st.dataframe(data.head(200), width="stretch")

# --------------------------------------------------------------------------- onglet 2
with tab2:
    if mask_a.sum() < 5 or mask_b.sum() < 5:
        st.warning("Groupes trop petits pour une comparaison fiable.")
    else:
        mean_a, mean_b = res.loc[mask_a, "salaire"].mean(), res.loc[mask_b, "salaire"].mean()
        raw = mean_a - mean_b
        unexplained = res.loc[mask_a, "ecart"].mean() - res.loc[mask_b, "ecart"].mean()
        explained = raw - unexplained  # = différence des salaires moyens prédits
        ref = mean_b

        st.subheader("Décomposition de l'écart de rémunération")
        c1, c2, c3 = st.columns(3)
        c1.metric("Écart brut", f"{raw:,.0f} €".replace(",", " "), f"{raw / ref * 100:+.1f} % vs référence", delta_color="off")
        c2.metric("dont expliqué par le profil", f"{explained:,.0f} €".replace(",", " "),
                  f"{explained / ref * 100:+.1f} %", delta_color="off")
        c3.metric("dont inexpliqué", f"{unexplained:,.0f} €".replace(",", " "),
                  f"{unexplained / ref * 100:+.1f} %", delta_color="off")

        a, b = res.loc[mask_a, "ecart"], res.loc[mask_b, "ecart"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        lo, hi = unexplained - 1.96 * se, unexplained + 1.96 * se
        verdict = ("statistiquement significatif" if p < 0.05 else "non significatif statistiquement")
        eur = lambda v: f"{v:,.0f} €".replace(",", " ")
        st.info(
            f"Écart inexpliqué : **{eur(unexplained)}** (IC 95 % : {eur(lo)} à {eur(hi)}), "
            f"test de Welch p = {p:.4f} → **{verdict}**."
        )

        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=[f"Salaire moyen {group_b}", "Effet du profil (poste, ancienneté…)", "Écart inexpliqué",
               f"Salaire moyen {group_a}"],
            y=[mean_b, explained, unexplained, 0],
            text=[f"{mean_b:,.0f}", f"{explained:+,.0f}", f"{unexplained:+,.0f}", f"{mean_a:,.0f}"],
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#2A7F9E"}}, decreasing={"marker": {"color": "#E07A1F"}},
            totals={"marker": {"color": "#6B7280"}},
        ))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="€",
                          yaxis_range=[min(mean_a, mean_b) * 0.85, max(mean_a, mean_b) * 1.05])
        st.plotly_chart(fig, width="stretch")

        l, r = st.columns(2)
        with l:
            st.subheader("Distribution des écarts individuels")
            fig = px.histogram(res[res.genre.isin([group_a, group_b])], x="ecart_pct", color="genre",
                               nbins=60, barmode="overlay", opacity=0.6,
                               color_discrete_map={group_a: COLORS["a"], group_b: COLORS["b"]},
                               labels={"ecart_pct": "Écart au salaire prédit (%)", "genre": "Genre"})
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
        with r:
            st.subheader("Écart inexpliqué par segment")
            seg_options = [c for c in features if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10]
            if seg_options:
                seg = st.selectbox("Segmenter par", seg_options, label_visibility="collapsed")
                tmp = res.assign(seg=df[seg].astype(str))
                tab = tmp.pivot_table(index="seg", columns="genre", values="ecart_pct", aggfunc=["mean", "count"])
                rows = []
                for s in tab.index:
                    na, nb = tab.loc[s, ("count", group_a)], tab.loc[s, ("count", group_b)]
                    if na >= 5 and nb >= 5:
                        rows.append({"segment": s, "écart (pts de %)": tab.loc[s, ("mean", group_a)] - tab.loc[s, ("mean", group_b)],
                                     f"n {group_a}": int(na), f"n {group_b}": int(nb)})
                seg_df = pd.DataFrame(rows).sort_values("écart (pts de %)") if rows else pd.DataFrame()
                if len(seg_df):
                    fig = px.bar(seg_df, x="écart (pts de %)", y="segment", orientation="h",
                                 hover_data=[f"n {group_a}", f"n {group_b}"], color_discrete_sequence=["#E07A1F"])
                    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Segments avec au moins 5 personnes dans chaque groupe.")
                else:
                    st.caption("Pas assez de personnes par segment.")

        st.warning(
            "**Lecture prudente.** Un écart « expliqué » peut lui-même refléter une inégalité (accès aux postes, "
            "promotions, évaluations). Les variables comme le poste ou la performance peuvent être des proxys du genre. "
            "SHAP décrit le modèle, pas une causalité : utilisez ces résultats comme point de départ d'une revue humaine."
        )

# --------------------------------------------------------------------------- onglet 3
with tab3:
    options = res["id"] + " · " + res["genre"] + " · " + res["salaire"].map(lambda v: f"{v:,.0f} €".replace(",", " "))
    pick = st.selectbox("Salarié", options.tolist())
    i = int(options[options == pick].index[0])
    row, contrib = res.loc[i], shap_df.loc[i].sort_values(key=np.abs, ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Salaire réel", f"{row.salaire / 1000:,.1f} k€".replace(",", " "))
    c2.metric("Salaire prédit", f"{row.predit / 1000:,.1f} k€".replace(",", " "))
    c3.metric("Écart inexpliqué", f"{row.ecart / 1000:+,.1f} k€".replace(",", " "), f"{row.ecart_pct:+.1f} %")

    top = contrib.head(7)
    other = contrib.iloc[7:].sum()
    names = [f"{f} = {df.loc[i, f]}" for f in top.index] + (["Autres variables"] if len(contrib) > 7 else [])
    vals = list(top.values) + ([other] if len(contrib) > 7 else [])
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(vals) + ["total", "relative", "total"],
        x=["Salaire moyen (base)"] + names + ["Salaire prédit", "Écart inexpliqué", "Salaire réel"],
        y=[base[i]] + vals + [0, row.ecart, 0],
        text=[f"{base[i] / 1000:.1f}k"] + [f"{v / 1000:+.1f}k" for v in vals] +
             [f"{row.predit / 1000:.1f}k", f"{row.ecart / 1000:+.1f}k", f"{row.salaire / 1000:.1f}k"],
        connector={"line": {"color": "gray"}},
        increasing={"marker": {"color": "#2A7F9E"}}, decreasing={"marker": {"color": "#E07A1F"}},
        totals={"marker": {"color": "#6B7280"}},
    ))
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="€")
    st.plotly_chart(fig, width="stretch")

    sign = "inférieur" if row.ecart < 0 else "supérieur"
    pos = [f"{f} ({df.loc[i, f]}) : {v / 1000:+.1f} k€" for f, v in top.items()]
    st.markdown(
        f"**{row['id']}** gagne **{row.salaire / 1000:.1f} k€**. Le modèle, sans connaître le genre, prédisait "
        f"**{row.predit / 1000:.1f} k€**, à partir de : " + " ; ".join(pos) + ".  \n"
        f"Reste un écart **{sign}** de **{abs(row.ecart) / 1000:.1f} k€** ({abs(row.ecart_pct):.1f} %) que le profil "
        f"ne justifie pas."
    )
    with st.expander("Profil complet"):
        st.dataframe(df.iloc[[i]].T.rename(columns={i: "valeur"}).astype(str), width="stretch")

# --------------------------------------------------------------------------- onglet 4
with tab4:
    st.subheader("Salaires sensiblement inférieurs au niveau prédit")
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Seuil d'alerte (écart < −x %)", 2, 20, 8)
    scope = c2.selectbox("Périmètre", ["Tous", f"Groupe {group_a}", f"Groupe {group_b}"])
    tol = c3.slider("Tolérance de rattrapage (% sous le prédit)", 0, 10, 0)

    pool = res if scope == "Tous" else res[res.genre == (group_a if scope.endswith(group_a) else group_b)]
    alerts = pool[pool.ecart_pct < -thr].copy()
    alerts["rattrapage"] = np.maximum(0, alerts.predit * (1 - tol / 100) - alerts.salaire)

    m1, m2, m3 = st.columns(3)
    m1.metric("Salariés en alerte", f"{len(alerts)}", f"{len(alerts) / max(len(pool), 1) * 100:.1f} % du périmètre", delta_color="off")
    m2.metric("Budget de rattrapage", f"{alerts.rattrapage.sum():,.0f} €".replace(",", " "))
    m3.metric("En % de la masse salariale", f"{alerts.rattrapage.sum() / res.salaire.sum() * 100:.2f} %")

    if len(alerts):
        by_g = alerts.genre.value_counts().rename_axis("genre").reset_index(name="alertes")
        by_g["part du groupe (%)"] = by_g.apply(lambda r: r.alertes / (res.genre == r.genre).sum() * 100, axis=1).round(1)
        st.dataframe(by_g, hide_index=True, width="content")

        show = alerts.sort_values("ecart_pct")[["id", "genre", "salaire", "predit", "ecart", "ecart_pct", "rattrapage"]]
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        st.download_button("Télécharger les alertes (CSV)", show.round(1).to_csv(index=False).encode("utf-8"),
                           file_name="alertes_equite_salariale.csv", mime="text/csv")
    else:
        st.success("Aucun salarié sous le seuil.")
