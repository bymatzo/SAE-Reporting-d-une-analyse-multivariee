"""
Analyse multivariée — SAE 4.02
ACP, ACM, AFC, CAH, k-moyennes.
Toutes les fonctions retournent des dicts de DataFrames prêts pour visualisation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import prince


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _scale(df: pd.DataFrame, cols: list) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values.astype(float))
    return X, scaler


def _cos2_from_coords(coords: np.ndarray) -> np.ndarray:
    """cos2 ligne i / axe s = F_is² / sum_s(F_is²)."""
    norms = np.sum(coords ** 2, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return coords ** 2 / norms


def _contributions(coords: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """Contribution de l'individu i à l'axe s = F_is² / (n * λ_s)."""
    n = coords.shape[0]
    return coords ** 2 / (n * eigenvalues[np.newaxis, :])


# ---------------------------------------------------------------------------
# 1. ACP — Analyse en Composantes Principales
# ---------------------------------------------------------------------------

def run_pca(
    df: pd.DataFrame,
    active_cols: list,
    supplementary_quanti: list = None,
    supplementary_quali: list = None,
    n_components: int = None,
) -> dict:
    """
    ACP sur les colonnes actives (standardisation automatique).

    Retourne
    --------
    dict avec :
      - model          : sklearn PCA
      - scores         : DataFrame (n × n_components) — coordonnées individus
      - loadings       : DataFrame (p × n_components) — corrélations var/axes (cercle)
      - explained_var  : Series — % variance cumulée par axe
      - cos2_ind       : DataFrame — qualité représentation individus
      - cos2_var       : DataFrame — qualité représentation variables
      - contrib_ind    : DataFrame — contributions individus (%)
      - contrib_var    : DataFrame — contributions variables (%)
      - eigenvalues    : array
      - supplementary  : dict — projections des éléments supplémentaires
    """
    df_work = df[active_cols].dropna()
    idx = df_work.index
    X, scaler = _scale(df_work, active_cols)
    p = X.shape[1]

    n_comp = n_components or p
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    scores_arr = pca.transform(X)
    eigenvalues = pca.explained_variance_

    # Cercle des corrélations : corr(x_j, F_s)
    # = composante_sj * sqrt(lambda_s) (données standardisées)
    loadings_arr = pca.components_.T * np.sqrt(eigenvalues)

    axis_labels = [f"Dim {i+1}" for i in range(n_comp)]
    var_labels = active_cols

    scores = pd.DataFrame(scores_arr, index=idx, columns=axis_labels)
    loadings = pd.DataFrame(loadings_arr, index=var_labels, columns=axis_labels)

    explained_var = pd.Series(
        pca.explained_variance_ratio_ * 100,
        index=axis_labels,
        name="% variance",
    )

    cos2_ind = pd.DataFrame(
        _cos2_from_coords(scores_arr), index=idx, columns=axis_labels
    )
    # cos2 variable = loading² (loadings sont des corrélations ∈ [-1, 1])
    cos2_var = pd.DataFrame(
        loadings_arr ** 2, index=var_labels, columns=axis_labels
    )

    contrib_ind = pd.DataFrame(
        _contributions(scores_arr, eigenvalues) * 100,
        index=idx,
        columns=axis_labels,
    )
    # Contribution variable j à axe s = V_sj² (part de la variance expliquée)
    contrib_var = pd.DataFrame(
        pca.components_.T ** 2 * 100,
        index=var_labels,
        columns=axis_labels,
    )

    # Éléments supplémentaires
    supplementary = {}
    if supplementary_quanti:
        sup_q_cols = [c for c in supplementary_quanti if c in df.columns]
        if sup_q_cols:
            X_sup = scaler.transform(
                df.loc[idx, sup_q_cols].fillna(df[sup_q_cols].mean()).values
            )
            sup_scores = pd.DataFrame(
                (X_sup - scaler.mean_[:len(sup_q_cols)]) @ pca.components_[:, :len(sup_q_cols)].T,
                index=idx, columns=axis_labels,
            )
            sup_loadings = pd.DataFrame(
                np.corrcoef(
                    df.loc[idx, sup_q_cols].fillna(df[sup_q_cols].mean()).values.T,
                    scores_arr.T,
                )[:len(sup_q_cols), len(sup_q_cols):],
                index=sup_q_cols, columns=axis_labels,
            )
            supplementary["quanti"] = {"loadings": sup_loadings}

    if supplementary_quali:
        sup_qual_results = {}
        for col in supplementary_quali:
            if col not in df.columns:
                continue
            df_sub = df.loc[idx, [col]].copy()
            df_sub["__scores__"] = scores_arr[:, 0]
            modality_coords = {}
            for dim_idx, dim in enumerate(axis_labels):
                df_sub[dim] = scores_arr[:, dim_idx]
            groups = df_sub.groupby(col)[axis_labels].mean()
            sup_qual_results[col] = groups
        supplementary["quali"] = sup_qual_results

    return {
        "model": pca,
        "scaler": scaler,
        "scores": scores,
        "loadings": loadings,
        "explained_var": explained_var,
        "eigenvalues": eigenvalues,
        "cos2_ind": cos2_ind,
        "cos2_var": cos2_var,
        "contrib_ind": contrib_ind,
        "contrib_var": contrib_var,
        "supplementary": supplementary,
        "active_cols": active_cols,
        "index": idx,
    }


# ---------------------------------------------------------------------------
# 2. ACM — Analyse des Correspondances Multiples
# ---------------------------------------------------------------------------

def run_mca(
    df: pd.DataFrame,
    active_cols: list,
    supplementary_cols: list = None,
    n_components: int = 5,
) -> dict:
    """
    ACM sur les colonnes qualitatives actives (via prince.MCA).

    Retourne
    --------
    dict avec :
      - model        : prince.MCA
      - row_coords   : DataFrame — coordonnées individus
      - col_coords   : DataFrame — coordonnées modalités
      - explained_var: Series — % inertie par axe
      - cos2_rows    : DataFrame
      - cos2_cols    : DataFrame
      - contrib_rows : DataFrame
      - contrib_cols : DataFrame
    """
    df_work = df[active_cols].dropna().astype(str)
    idx = df_work.index

    mca = prince.MCA(n_components=n_components, random_state=42)
    mca = mca.fit(df_work)

    row_coords = mca.row_coordinates(df_work)
    col_coords = mca.column_coordinates(df_work)

    axis_labels = [f"Dim {i+1}" for i in range(n_components)]
    row_coords.columns = axis_labels
    col_coords.columns = axis_labels

    eigenvalues = np.array(mca.eigenvalues_)
    total_inertia = eigenvalues.sum()
    explained_var = pd.Series(
        eigenvalues / total_inertia * 100 if total_inertia > 0 else eigenvalues,
        index=axis_labels,
        name="% inertie",
    )

    cos2_rows = pd.DataFrame(
        _cos2_from_coords(row_coords.values),
        index=row_coords.index,
        columns=axis_labels,
    )
    cos2_cols = pd.DataFrame(
        _cos2_from_coords(col_coords.values),
        index=col_coords.index,
        columns=axis_labels,
    )

    contrib_rows = pd.DataFrame(
        _contributions(row_coords.values, eigenvalues[:n_components]) * 100,
        index=row_coords.index,
        columns=axis_labels,
    )
    contrib_cols = pd.DataFrame(
        _contributions(col_coords.values, eigenvalues[:n_components]) * 100,
        index=col_coords.index,
        columns=axis_labels,
    )

    # Variables supplémentaires qualitatives : barycentres des modalités
    supplementary = {}
    if supplementary_cols:
        sup_results = {}
        for col in supplementary_cols:
            if col not in df.columns:
                continue
            df_sup = df.loc[idx, [col]].copy().astype(str)
            df_sup[axis_labels] = row_coords.values
            sup_results[col] = df_sup.groupby(col)[axis_labels].mean()
        supplementary["quali"] = sup_results

    return {
        "model": mca,
        "row_coords": row_coords,
        "col_coords": col_coords,
        "explained_var": explained_var,
        "eigenvalues": eigenvalues,
        "cos2_rows": cos2_rows,
        "cos2_cols": cos2_cols,
        "contrib_rows": contrib_rows,
        "contrib_cols": contrib_cols,
        "supplementary": supplementary,
        "active_cols": active_cols,
        "index": idx,
    }


# ---------------------------------------------------------------------------
# 3. AFC — Analyse Factorielle des Correspondances
# ---------------------------------------------------------------------------

def run_ca(
    df: pd.DataFrame,
    col_row: str,
    col_col: str,
    n_components: int = 5,
) -> dict:
    """
    AFC sur le tableau de contingence croisant col_row et col_col.

    Retourne
    --------
    dict avec :
      - model        : prince.CA
      - contingency  : DataFrame — tableau de contingence
      - row_coords   : DataFrame — coordonnées des modalités lignes
      - col_coords   : DataFrame — coordonnées des modalités colonnes
      - explained_var: Series
    """
    df_work = df[[col_row, col_col]].dropna().astype(str)
    contingency = pd.crosstab(df_work[col_row], df_work[col_col])

    ca = prince.CA(n_components=n_components, random_state=42)
    ca = ca.fit(contingency)

    row_coords = ca.row_coordinates(contingency)
    col_coords = ca.column_coordinates(contingency)

    # n_components réel peut être < n_components demandé (limité par min(nrows, ncols) - 1)
    n_actual = row_coords.shape[1]
    axis_labels = [f"Dim {i+1}" for i in range(n_actual)]
    row_coords.columns = axis_labels
    col_coords.columns = axis_labels

    eigenvalues = np.array(ca.eigenvalues_)
    total_inertia = eigenvalues.sum()
    explained_var = pd.Series(
        eigenvalues / total_inertia * 100 if total_inertia > 0 else eigenvalues,
        index=axis_labels,
        name="% inertie",
    )

    return {
        "model": ca,
        "contingency": contingency,
        "row_coords": row_coords,
        "col_coords": col_coords,
        "explained_var": explained_var,
        "eigenvalues": eigenvalues,
        "col_row": col_row,
        "col_col": col_col,
    }


# ---------------------------------------------------------------------------
# 4. CAH — Classification Ascendante Hiérarchique
# ---------------------------------------------------------------------------

def run_hca(
    df: pd.DataFrame,
    cols: list,
    n_clusters: int = None,
    linkage_method: str = "ward",
    metric: str = "euclidean",
) -> dict:
    """
    CAH sur les colonnes spécifiées (standardisation automatique).

    Retourne
    --------
    dict avec :
      - linkage_matrix     : np.ndarray — matrice pour dendrogramme scipy
      - labels             : Series — cluster de chaque individu
      - n_clusters_auto    : int — nombre de clusters suggéré (saut maximal)
      - inertia_levels     : Series — inertie à chaque niveau de fusion
    """
    df_work = df[cols].dropna()
    idx = df_work.index
    X, _ = _scale(df_work, cols)

    Z = linkage(X, method=linkage_method, metric=metric)

    # Suggestion du nombre de clusters : saut maximal dans les distances de fusion
    distances = Z[:, 2]
    accelerations = np.diff(distances, 2)
    n_clusters_auto = int(accelerations[::-1].argmax()) + 2 if len(accelerations) > 0 else 3

    k = n_clusters or n_clusters_auto
    raw_labels = fcluster(Z, k, criterion="maxclust")
    labels = pd.Series(raw_labels, index=idx, name="cluster_hca")

    # Inertie inter-clusters à chaque niveau (derniers k niveaux)
    last_k = min(20, len(Z))
    inertia_levels = pd.Series(
        Z[-last_k:, 2][::-1],
        index=range(1, last_k + 1),
        name="distance_fusion",
    )

    return {
        "linkage_matrix": Z,
        "labels": labels,
        "n_clusters_used": k,
        "n_clusters_auto": n_clusters_auto,
        "inertia_levels": inertia_levels,
        "index": idx,
    }


# ---------------------------------------------------------------------------
# 5. K-moyennes
# ---------------------------------------------------------------------------

def run_kmeans(
    df: pd.DataFrame,
    cols: list,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> dict:
    """
    K-moyennes sur les colonnes spécifiées (standardisation automatique).

    Retourne
    --------
    dict avec :
      - model      : KMeans
      - labels     : Series — cluster de chaque individu
      - inertia    : float — inertie intra-clusters
      - silhouette : float
      - centers    : DataFrame — centroïdes dans l'espace original
    """
    df_work = df[cols].dropna()
    idx = df_work.index
    X, scaler = _scale(df_work, cols)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    km.fit(X)

    labels = pd.Series(km.labels_ + 1, index=idx, name="cluster_km")
    sil = float(silhouette_score(X, km.labels_)) if k > 1 else 0.0

    centers_scaled = km.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    centers = pd.DataFrame(centers_original, columns=cols)
    centers.index = [f"Cluster {i+1}" for i in range(k)]

    return {
        "model": km,
        "labels": labels,
        "inertia": float(km.inertia_),
        "silhouette": sil,
        "centers": centers,
        "k": k,
        "index": idx,
    }


def elbow(
    df: pd.DataFrame,
    cols: list,
    k_range: range = range(2, 11),
    n_init: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Méthode du coude : calcule inertie et silhouette pour chaque k.

    Retourne
    --------
    dict avec :
      - k_values          : list
      - inertias          : list
      - silhouette_scores : list
      - best_k_silhouette : int — k maximisant le score de silhouette
    """
    df_work = df[cols].dropna()
    X, _ = _scale(df_work, cols)

    k_values, inertias, silhouettes = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X)
        k_values.append(k)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X, km.labels_)) if k > 1 else 0.0
        silhouettes.append(sil)

    best_k = k_values[int(np.argmax(silhouettes))]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "best_k_silhouette": best_k,
    }


# ---------------------------------------------------------------------------
# 6. Description des clusters
# ---------------------------------------------------------------------------

def describe_clusters(
    df: pd.DataFrame,
    labels: pd.Series,
    quanti_cols: list = None,
    quali_cols: list = None,
) -> dict:
    """
    Profil moyen de chaque cluster sur les variables quantitatives,
    et distribution des modalités pour les variables qualitatives.

    Retourne
    --------
    dict avec :
      - quanti_profile : DataFrame — moyenne par cluster × variable
      - quali_profile  : dict col → DataFrame fréquences par cluster
      - cluster_sizes  : Series
    """
    df_work = df.copy()
    df_work["__cluster__"] = labels.reindex(df_work.index)
    df_work = df_work.dropna(subset=["__cluster__"])

    result = {"cluster_sizes": df_work["__cluster__"].value_counts().sort_index()}

    if quanti_cols:
        cols = [c for c in quanti_cols if c in df_work.columns]
        result["quanti_profile"] = df_work.groupby("__cluster__")[cols].mean().round(2)

    if quali_cols:
        quali_profiles = {}
        for col in quali_cols:
            if col not in df_work.columns:
                continue
            ct = pd.crosstab(df_work["__cluster__"], df_work[col], normalize="index") * 100
            quali_profiles[col] = ct.round(1)
        result["quali_profile"] = quali_profiles

    return result
