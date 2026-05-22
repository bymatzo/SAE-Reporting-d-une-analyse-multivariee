"""
Interface Streamlit — SAE 4.02
Reporting d'une analyse multivariée — Open Food Facts
"""

import sys
sys.path.insert(0, ".")

import io
import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import (
    load_data, detect_column_types, diagnose_missing,
    filter_missing, remove_duplicates, handle_outliers,
    impute_pca, impute_quali, recode_variables, METADATA_COLS,
)
from analysis import (
    run_pca, run_mca, run_ca,
    run_hca, run_kmeans, elbow, describe_clusters,
)
from visualisation import (
    plot_missing_heatmap, plot_univariate, plot_correlation_matrix, plot_boxplot,
    plot_scree, plot_correlation_circle, plot_individuals, plot_cos2_bar,
    plot_mca_modalities, plot_ca_biplot,
    plot_dendrogram, plot_elbow, plot_cluster_profile, plot_clusters_on_pca,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SAE 4.02 — Analyse multivariée",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "df_raw": None,
        "df": None,
        "rapport": None,
        "col_types": None,
        "filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_preprocessing(filepath, seuil_col, seuil_ligne, ncp, normalize_multilingual, api_key):
    """Pipeline complet mis en cache par paramètres."""
    df_raw = load_data(filepath)
    df = df_raw.drop(columns=[c for c in METADATA_COLS if c in df_raw.columns])
    df, rapport_filter = filter_missing(df, seuil_col=seuil_col, seuil_ligne=seuil_ligne)
    df, n_dupl = remove_duplicates(df)
    col_types = detect_column_types(df)
    quanti_cols = col_types["quanti"]
    quanti_miss = [c for c in quanti_cols if df[c].isnull().any()]
    if quanti_miss:
        df, rapport_pca = impute_pca(df, quanti_miss, ncp=ncp)
    else:
        rapport_pca = {"ncp_used": 0, "n_imputed": 0}
    quali_cols = col_types["quali"]
    quali_miss = [c for c in quali_cols if df[c].isnull().any()]
    if quali_miss:
        df, _ = impute_quali(df, quali_miss, quanti_cols=quanti_cols, method="mode")
    if normalize_multilingual:
        from normalisation_llm import normalize_multilingual_columns
        df = normalize_multilingual_columns(df, api_key=api_key or None)
    df = recode_variables(df)
    col_types = detect_column_types(df)
    rapport = {
        "shape_raw": df_raw.shape,
        "filter": rapport_filter,
        "duplicates_removed": n_dupl,
        "imputation": rapport_pca,
        "shape_final": df.shape,
        "multilingual_normalized": normalize_multilingual,
    }
    return df, rapport, col_types


@st.cache_data(show_spinner=False)
def _run_pca(df_json, active_cols, sup_quali, n_components):
    df = pd.read_json(io.StringIO(df_json))
    return run_pca(df, active_cols, supplementary_quali=sup_quali or None,
                   n_components=n_components)


@st.cache_data(show_spinner=False)
def _run_mca(df_json, active_cols, sup_cols, n_components):
    df = pd.read_json(io.StringIO(df_json))
    return run_mca(df, active_cols, supplementary_cols=sup_cols or None,
                   n_components=n_components)


@st.cache_data(show_spinner=False)
def _run_ca(df_json, col_row, col_col):
    df = pd.read_json(io.StringIO(df_json))
    return run_ca(df, col_row, col_col)


@st.cache_data(show_spinner=False)
def _run_elbow(df_json, cols, k_max):
    df = pd.read_json(io.StringIO(df_json))
    return elbow(df, cols, k_range=range(2, k_max + 1))


@st.cache_data(show_spinner=False)
def _run_kmeans(df_json, cols, k):
    df = pd.read_json(io.StringIO(df_json))
    return run_kmeans(df, cols, k=k)


@st.cache_data(show_spinner=False)
def _run_hca(df_json, cols):
    df = pd.read_json(io.StringIO(df_json))
    return run_hca(df, cols)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📊 Reporting d'une analyse multivariée")
st.caption("SAE 4.02 — Université Lumière Lyon 2 — Open Food Facts")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_preanalyse, tab_facto, tab_classif = st.tabs([
    "📁 Données",
    "📈 Pré-analyse",
    "🔬 Analyse factorielle",
    "🗂️ Classification",
])


# ===========================================================================
# TAB 1 — DONNÉES
# ===========================================================================

with tab_data:
    st.header("Chargement et préparation des données")

    uploaded = st.file_uploader(
        "Charger un fichier Open Food Facts (.xlsx ou .csv)",
        type=["xlsx", "csv"],
    )

    if uploaded:
        # Sauvegarde temporaire pour load_data
        import tempfile, os
        suffix = ".xlsx" if uploaded.name.endswith(".xlsx") else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.session_state["filename"] = uploaded.name
        st.session_state["tmp_path"] = tmp_path

    st.divider()

    # Paramètres de nettoyage
    with st.expander("⚙️ Paramètres de préparation", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            seuil_col = st.slider(
                "Seuil NaN colonnes (supprimer si >)",
                min_value=0.1, max_value=1.0, value=0.70, step=0.05,
                help="Colonnes avec plus de X% de valeurs manquantes sont supprimées",
            )
        with col2:
            seuil_ligne = st.slider(
                "Seuil NaN lignes (supprimer si >)",
                min_value=0.1, max_value=1.0, value=0.50, step=0.05,
            )
        with col3:
            ncp_imp = st.slider(
                "Composantes ACP pour imputation",
                min_value=1, max_value=8, value=2,
                help="Nombre de composantes utilisées par l'algorithme missMDA",
            )

        st.divider()
        normalize_ml = st.toggle(
            "🌍 Normaliser les colonnes multilingues (packaging, catégories, labels, pays, allergènes)",
            value=False,
            help=(
                "Traduit en français les valeurs multilingues via Claude API. "
                "Nécessite une clé API Anthropic. Les traductions sont mises en cache "
                "localement (.llm_translation_cache.json) pour les runs suivants."
            ),
        )
        anthropic_key = ""
        if normalize_ml:
            anthropic_key = st.text_input(
                "Clé API Anthropic",
                type="password",
                help="Laisser vide si la variable d'environnement ANTHROPIC_API_KEY est définie.",
            )

    if st.button("🚀 Lancer la préparation", type="primary",
                 disabled="tmp_path" not in st.session_state):
        with st.spinner("Préparation en cours…"):
            df, rapport, col_types = _run_preprocessing(
                st.session_state["tmp_path"], seuil_col, seuil_ligne, ncp_imp,
                normalize_ml, anthropic_key,
            )
            st.session_state["df"] = df
            st.session_state["rapport"] = rapport
            st.session_state["col_types"] = col_types
        st.success("Données prêtes !")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        rapport = st.session_state["rapport"]
        col_types = st.session_state["col_types"]

        # Résumé
        st.subheader("Résumé de la préparation")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Lignes initiales", rapport["shape_raw"][0])
        m2.metric("Colonnes initiales", rapport["shape_raw"][1])
        m3.metric("Lignes finales", rapport["shape_final"][0],
                  delta=-(rapport["shape_raw"][0] - rapport["shape_final"][0]))
        m4.metric("Colonnes finales", rapport["shape_final"][1],
                  delta=-(rapport["shape_raw"][1] - rapport["shape_final"][1]))
        m5.metric("Doublons supprimés", rapport["duplicates_removed"])

        st.caption(
            f"Valeurs imputées (ACP itérative, ncp={rapport['imputation']['ncp_used']}) : "
            f"{rapport['imputation']['n_imputed']:,}"
        )

        st.divider()

        # Aperçu et stats
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("Aperçu des données")
            st.dataframe(df.head(20), use_container_width=True)

        with col_right:
            st.subheader("Statistiques descriptives")
            quanti = col_types["quanti"]
            st.dataframe(
                df[quanti].describe().round(2).T,
                use_container_width=True,
            )

        st.divider()

        # Diagnostic NaN sur données brutes
        st.subheader("Valeurs manquantes avant nettoyage")
        raw = load_data(st.session_state["tmp_path"])
        raw2 = raw.drop(columns=[c for c in METADATA_COLS if c in raw.columns])
        miss = diagnose_missing(raw2)
        if not miss.empty:
            st.plotly_chart(
                plot_missing_heatmap(miss.head(40)),
                use_container_width=True,
            )
        else:
            st.info("Aucune valeur manquante détectée.")


# ===========================================================================
# TAB 2 — PRÉ-ANALYSE
# ===========================================================================

with tab_preanalyse:
    st.header("Pré-analyse")

    if st.session_state["df"] is None:
        st.info("Chargez et préparez les données dans l'onglet **Données** d'abord.")
    else:
        df = st.session_state["df"]
        col_types = st.session_state["col_types"]
        quanti_cols = col_types["quanti"]
        quali_cols = col_types["quali"]

        graphique = st.radio(
            "Graphique",
            ["Distribution", "Matrice de corrélation", "Boxplot croisé"],
            horizontal=True,
        )

        if graphique == "Distribution":
            all_cols = quanti_cols + quali_cols
            col = st.selectbox("Variable", all_cols)
            st.plotly_chart(plot_univariate(df, col), use_container_width=True)

        elif graphique == "Matrice de corrélation":
            selected = st.multiselect(
                "Variables quantitatives",
                quanti_cols,
                default=quanti_cols[:8],
            )
            if len(selected) >= 2:
                st.plotly_chart(plot_correlation_matrix(df, selected), use_container_width=True)
            else:
                st.warning("Sélectionnez au moins 2 variables.")

        elif graphique == "Boxplot croisé":
            c1, c2 = st.columns(2)
            with c1:
                q_col = st.selectbox("Variable quantitative", quanti_cols)
            with c2:
                ql_col = st.selectbox("Variable qualitative", quali_cols)
            st.plotly_chart(plot_boxplot(df, q_col, ql_col), use_container_width=True)


# ===========================================================================
# TAB 3 — ANALYSE FACTORIELLE
# ===========================================================================

with tab_facto:
    st.header("Analyse factorielle")

    if st.session_state["df"] is None:
        st.info("Chargez et préparez les données dans l'onglet **Données** d'abord.")
    else:
        df = st.session_state["df"]
        col_types = st.session_state["col_types"]
        quanti_cols = col_types["quanti"]
        quali_cols = col_types["quali"]
        df_json = df.to_json()

        methode = st.radio("Méthode", ["ACP", "ACM", "AFC"], horizontal=True)

        st.divider()

        # ---- ACP ----
        if methode == "ACP":
            with st.expander("⚙️ Paramètres ACP", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    active = st.multiselect(
                        "Variables actives (quantitatives)",
                        quanti_cols,
                        default=[c for c in [
                            "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
                            "carbohydrates_100g", "sugars_100g", "fiber_100g",
                            "proteins_100g", "salt_100g",
                        ] if c in quanti_cols] or quanti_cols[:6],
                    )
                with c2:
                    sup_quali = st.multiselect(
                        "Variables illustratives (qualitatives)",
                        quali_cols,
                        default=[c for c in ["nutriscore_grade", "nova_group"] if c in quali_cols],
                    )
                n_comp = st.slider("Nombre de composantes", 2, min(10, len(active) if active else 2), 5)
                c3, c4 = st.columns(2)
                with c3:
                    dim_x = st.selectbox("Axe horizontal", list(range(1, n_comp + 1)), index=0)
                with c4:
                    dim_y = st.selectbox("Axe vertical", list(range(1, n_comp + 1)), index=1)

            if len(active) < 2:
                st.warning("Sélectionnez au moins 2 variables actives.")
            else:
                with st.spinner("Calcul ACP…"):
                    res = _run_pca(df_json, active, sup_quali, n_comp)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_scree(res["explained_var"]), use_container_width=True)
                with col_b:
                    st.plotly_chart(
                        plot_correlation_circle(res["loadings"], dim_x, dim_y, cos2_var=res["cos2_var"]),
                        use_container_width=True,
                    )

                color_opt = st.selectbox(
                    "Colorier les individus par",
                    ["(aucun)"] + sup_quali,
                )
                color_series = df[color_opt] if color_opt != "(aucun)" else None
                st.plotly_chart(
                    plot_individuals(res["scores"], dim_x, dim_y,
                                     color_col=color_series,
                                     explained_var=res["explained_var"]),
                    use_container_width=True,
                )

                with st.expander("Qualité de représentation (cos²)"):
                    c5, c6 = st.columns(2)
                    with c5:
                        st.plotly_chart(plot_cos2_bar(res["cos2_var"], dim_x), use_container_width=True)
                    with c6:
                        st.plotly_chart(plot_cos2_bar(res["cos2_var"], dim_y), use_container_width=True)

                if sup_quali and res["supplementary"].get("quali"):
                    with st.expander("Coordonnées des variables illustratives"):
                        for col_name, coords in res["supplementary"]["quali"].items():
                            st.write(f"**{col_name}**")
                            st.dataframe(coords.round(3), use_container_width=True)

        # ---- ACM ----
        elif methode == "ACM":
            with st.expander("⚙️ Paramètres ACM", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    active = st.multiselect(
                        "Variables actives (qualitatives)",
                        quali_cols,
                        default=[c for c in [
                            "nutriscore_grade", "nova_group", "environmental_score_grade"
                        ] if c in quali_cols] or quali_cols[:3],
                    )
                with c2:
                    sup_cols = st.multiselect(
                        "Variables illustratives",
                        [c for c in quali_cols if c not in active],
                    )
                n_comp = st.slider("Nombre de composantes", 2, 10, 5)
                c3, c4 = st.columns(2)
                with c3:
                    dim_x = st.selectbox("Axe horizontal", list(range(1, n_comp + 1)), index=0)
                with c4:
                    dim_y = st.selectbox("Axe vertical", list(range(1, n_comp + 1)), index=1)

            if len(active) < 2:
                st.warning("Sélectionnez au moins 2 variables actives.")
            else:
                with st.spinner("Calcul ACM…"):
                    res = _run_mca(df_json, active, sup_cols, n_comp)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_scree(res["explained_var"]), use_container_width=True)
                with col_b:
                    st.plotly_chart(
                        plot_mca_modalities(res["col_coords"], dim_x, dim_y,
                                            explained_var=res["explained_var"]),
                        use_container_width=True,
                    )

                color_opt = st.selectbox(
                    "Colorier les individus par",
                    ["(aucun)"] + active + sup_cols,
                )
                color_series = df[color_opt] if color_opt != "(aucun)" else None
                st.plotly_chart(
                    plot_individuals(res["row_coords"], dim_x, dim_y,
                                     color_col=color_series,
                                     explained_var=res["explained_var"]),
                    use_container_width=True,
                )

                if sup_cols and res["supplementary"].get("quali"):
                    with st.expander("Coordonnées des variables illustratives"):
                        for col_name, coords in res["supplementary"]["quali"].items():
                            st.write(f"**{col_name}**")
                            st.dataframe(coords.round(3), use_container_width=True)

        # ---- AFC ----
        elif methode == "AFC":
            with st.expander("⚙️ Paramètres AFC", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    col_row = st.selectbox("Variable lignes", quali_cols, index=0)
                with c2:
                    col_col = st.selectbox(
                        "Variable colonnes",
                        [c for c in quali_cols if c != col_row],
                        index=0,
                    )
                with c3:
                    dim_x = st.selectbox("Axe horizontal", [1, 2, 3], index=0)
                with c4:
                    dim_y = st.selectbox("Axe vertical", [1, 2, 3], index=1)

            with st.spinner("Calcul AFC…"):
                res = _run_ca(df_json, col_row, col_col)

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(plot_scree(res["explained_var"]), use_container_width=True)
            with col_b:
                st.plotly_chart(
                    plot_ca_biplot(
                        res["row_coords"], res["col_coords"],
                        dim_x, dim_y,
                        explained_var=res["explained_var"],
                        row_label=col_row, col_label=col_col,
                    ),
                    use_container_width=True,
                )

            with st.expander("Tableau de contingence"):
                st.dataframe(res["contingency"], use_container_width=True)


# ===========================================================================
# TAB 4 — CLASSIFICATION
# ===========================================================================

with tab_classif:
    st.header("Classification non supervisée")

    if st.session_state["df"] is None:
        st.info("Chargez et préparez les données dans l'onglet **Données** d'abord.")
    else:
        df = st.session_state["df"]
        col_types = st.session_state["col_types"]
        quanti_cols = col_types["quanti"]
        quali_cols = col_types["quali"]
        df_json = df.to_json()

        # Sélection des variables
        with st.expander("⚙️ Variables et paramètres", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                cluster_cols = st.multiselect(
                    "Variables pour la classification",
                    quanti_cols,
                    default=[c for c in [
                        "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
                        "carbohydrates_100g", "sugars_100g", "fiber_100g",
                        "proteins_100g", "salt_100g",
                    ] if c in quanti_cols] or quanti_cols[:6],
                )
            with c2:
                k_max = st.slider("k maximum (méthode du coude)", 3, 15, 8)

        if len(cluster_cols) < 2:
            st.warning("Sélectionnez au moins 2 variables.")
        else:
            # Méthode du coude
            st.subheader("Méthode du coude")
            with st.spinner("Calcul de la courbe du coude…"):
                res_elbow = _run_elbow(df_json, cluster_cols, k_max)

            st.plotly_chart(plot_elbow(res_elbow), use_container_width=True)
            best_k = res_elbow["best_k_silhouette"]
            st.caption(f"Meilleur k selon la silhouette : **{best_k}**")

            st.divider()

            # K-moyennes
            st.subheader("K-Moyennes")
            k_choice = st.slider("Nombre de clusters k", 2, k_max, best_k)

            with st.spinner(f"Calcul k-moyennes (k={k_choice})…"):
                res_km = _run_kmeans(df_json, cluster_cols, k_choice)

            m1, m2 = st.columns(2)
            m1.metric("Inertie intra-clusters", f"{res_km['inertia']:.0f}")
            m2.metric("Score de silhouette", f"{res_km['silhouette']:.3f}")

            desc = describe_clusters(
                df, res_km["labels"],
                quanti_cols=cluster_cols,
                quali_cols=[c for c in ["nutriscore_grade", "nova_group"] if c in df.columns],
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(plot_cluster_profile(desc["quanti_profile"]), use_container_width=True)
            with col_b:
                # Projection des clusters sur l'ACP des mêmes variables
                with st.spinner("Projection sur ACP…"):
                    res_pca = _run_pca(df_json, cluster_cols, [], 5)
                st.plotly_chart(
                    plot_clusters_on_pca(res_pca["scores"], res_km["labels"],
                                         explained_var=res_pca["explained_var"]),
                    use_container_width=True,
                )

            with st.expander("Profil détaillé des clusters"):
                st.write("**Moyennes des variables quantitatives**")
                st.dataframe(desc["quanti_profile"].round(2), use_container_width=True)
                if "quali_profile" in desc:
                    for col_name, profile in desc["quali_profile"].items():
                        st.write(f"**{col_name} (%)**")
                        st.dataframe(profile, use_container_width=True)

            st.divider()

            # CAH (optionnel)
            with st.expander("🌳 Classification Ascendante Hiérarchique (CAH)"):
                st.caption("La CAH peut être longue sur de grands jeux de données.")
                if st.button("Calculer la CAH"):
                    with st.spinner("Calcul CAH…"):
                        res_hca = _run_hca(df_json, cluster_cols)
                    st.info(f"Nombre de clusters suggéré par la CAH : **{res_hca['n_clusters_auto']}**")
                    st.plotly_chart(plot_dendrogram(res_hca["linkage_matrix"]), use_container_width=True)
