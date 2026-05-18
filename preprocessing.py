"""
Preprocessing pipeline — SAE 4.02
Chargement, nettoyage et imputation des données Open Food Facts.
Générique : aucun nom de colonne codé en dur.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Colonnes metadata à exclure par défaut (non pertinentes pour l'analyse)
# ---------------------------------------------------------------------------
METADATA_COLS = {
    "code", "url", "creator", "created_t", "created_datetime",
    "last_modified_t", "last_modified_datetime", "last_modified_by",
    "last_updated_t", "last_updated_datetime", "abbreviated_product_name",
    "generic_name", "packaging_tags", "packaging_en", "packaging_text",
    "brands_tags", "brands_en", "categories_tags", "categories_en",
    "origins_tags", "origins_en", "manufacturing_places_tags",
    "labels_tags", "labels_en", "emb_codes", "emb_codes_tags",
    "first_packaging_code_geo", "cities", "cities_tags",
    "countries_tags", "countries_en", "ingredients_tags",
    "ingredients_analysis_tags", "traces_tags", "traces_en",
    "additives_tags", "additives_en", "food_groups_tags", "food_groups_en",
    "states", "states_tags", "states_en", "popularity_tags",
    "data_quality_errors_tags", "main_category", "main_category_en",
    "image_url", "image_small_url", "image_ingredients_url",
    "image_ingredients_small_url", "image_nutrition_url",
    "image_nutrition_small_url", "last_image_t", "last_image_datetime",
    "brand_owner", "owner", "pnns_groups_1", "pnns_groups_2",
    "food_groups", "nutrient_levels_tags",
}

# Bornes physiques pour les valeurs nutritionnelles (/100g ou /100ml)
NUTRITIONAL_BOUNDS = {
    "energy-kj_100g":     (0, 3700),
    "energy-kcal_100g":   (0, 900),
    "energy_100g":        (0, 3700),
    "fat_100g":           (0, 100),
    "saturated-fat_100g": (0, 100),
    "carbohydrates_100g": (0, 100),
    "sugars_100g":        (0, 100),
    "fiber_100g":         (0, 100),
    "proteins_100g":      (0, 100),
    "salt_100g":          (0, 100),
    "sodium_100g":        (0, 40),
    "alcohol_100g":       (0, 100),
}


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Charge un fichier Excel ou CSV et retourne un DataFrame brut."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
    else:
        df = pd.read_excel(filepath)
    return df


# ---------------------------------------------------------------------------
# 2. Détection automatique des types
# ---------------------------------------------------------------------------

def detect_column_types(df: pd.DataFrame, exclude_cols: set = None) -> dict:
    """
    Retourne {"quanti": [...], "quali": [...], "excluded": [...]}
    en détectant automatiquement les types des colonnes restantes.
    """
    if exclude_cols is None:
        exclude_cols = set()

    quanti, quali, excluded = [], [], []
    for col in df.columns:
        if col in exclude_cols:
            excluded.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            quanti.append(col)
        else:
            quali.append(col)

    return {"quanti": quanti, "quali": quali, "excluded": excluded}


# ---------------------------------------------------------------------------
# 3. Diagnostic des valeurs manquantes
# ---------------------------------------------------------------------------

def diagnose_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame résumant le taux de NaN par colonne,
    trié du plus manquant au moins manquant.
    """
    total = len(df)
    missing = df.isnull().sum()
    rate = missing / total
    summary = pd.DataFrame({
        "n_missing": missing,
        "pct_missing": (rate * 100).round(2),
    })
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def missing_pattern_matrix(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Retourne une matrice présence/absence (0=observé, 1=manquant)
    pour visualiser le dispositif de données manquantes (cf. cours slide 12).
    """
    sub = df[cols] if cols else df
    return sub.isnull().astype(int)


# ---------------------------------------------------------------------------
# 4. Filtrage colonnes / lignes
# ---------------------------------------------------------------------------

def filter_missing(
    df: pd.DataFrame,
    seuil_col: float = 0.70,
    seuil_ligne: float = 0.50,
) -> tuple[pd.DataFrame, dict]:
    """
    Supprime les colonnes avec > seuil_col de NaN,
    puis les lignes avec > seuil_ligne de NaN.
    Retourne (df_filtré, rapport).
    """
    n_cols_init, n_rows_init = df.shape

    cols_dropped = df.columns[df.isnull().mean() > seuil_col].tolist()
    df = df.drop(columns=cols_dropped)

    rows_mask = df.isnull().mean(axis=1) > seuil_ligne
    n_rows_dropped = rows_mask.sum()
    df = df[~rows_mask].reset_index(drop=True)

    rapport = {
        "cols_init": n_cols_init,
        "cols_dropped": cols_dropped,
        "cols_remaining": df.shape[1],
        "rows_init": n_rows_init,
        "rows_dropped": int(n_rows_dropped),
        "rows_remaining": df.shape[0],
    }
    return df, rapport


# ---------------------------------------------------------------------------
# 5. Doublons
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, subset: list = None) -> tuple[pd.DataFrame, int]:
    """
    Supprime les doublons sur `subset` (défaut: product_name + brands).
    En cas de doublon, conserve la ligne la plus complète (moins de NaN).
    """
    if subset is None:
        candidates = [c for c in ["product_name", "brands"] if c in df.columns]
        subset = candidates if candidates else None

    n_init = len(df)

    if subset:
        df = (
            df.assign(_n_missing=df.isnull().sum(axis=1))
            .sort_values("_n_missing")
            .drop_duplicates(subset=subset, keep="first")
            .drop(columns="_n_missing")
            .reset_index(drop=True)
        )
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    return df, n_init - len(df)


# ---------------------------------------------------------------------------
# 6. Valeurs aberrantes
# ---------------------------------------------------------------------------

def handle_outliers(
    df: pd.DataFrame,
    quanti_cols: list,
    method: str = "clip",
) -> tuple[pd.DataFrame, dict]:
    """
    Détecte et traite les valeurs aberrantes sur les colonnes quantitatives.
    - method='clip' : winsorisation aux bornes physiques si connues, sinon IQR×3
    - method='remove' : suppression des lignes aberrantes
    Retourne (df_corrigé, rapport).
    """
    rapport = {}
    df = df.copy()

    for col in quanti_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Bornes : physiques si connues, sinon IQR × 3
        if col in NUTRITIONAL_BOUNDS:
            low, high = NUTRITIONAL_BOUNDS[col]
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 3 * iqr
            high = q3 + 3 * iqr

        mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        n_outliers = mask.sum()

        if n_outliers > 0:
            rapport[col] = {"n_outliers": int(n_outliers), "low": low, "high": high}
            if method == "clip":
                df[col] = df[col].clip(lower=low, upper=high)
            elif method == "remove":
                df.loc[mask, col] = np.nan

    return df, rapport


# ---------------------------------------------------------------------------
# 7. Imputation par ACP itérative régularisée (variables quantitatives)
#    Algorithme missMDA — Josse & Husson (2009)
# ---------------------------------------------------------------------------

def _msep_cv(X_std: np.ndarray, ncp: int, n_folds: int = 5) -> float:
    """
    Estime le MSEP pour un nombre de composantes donné
    par validation croisée Leave-one-out approché (GCV).
    X_std doit être centré-réduit, sans NaN.
    """
    n, p = X_std.shape
    # Masque aléatoire reproductible : on cache ~10% des valeurs observées
    rng = np.random.default_rng(42)
    mask = rng.random((n, p)) < 0.10

    X_cv = X_std.copy()
    X_cv[mask] = np.nan

    X_imp = _iterative_pca(X_cv, ncp=ncp, regularized=True)
    msep = np.mean((X_std[mask] - X_imp[mask]) ** 2)
    return msep


def _iterative_pca(
    X: np.ndarray,
    ncp: int,
    regularized: bool = True,
    n_iter_max: int = 1000,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    ACP itérative régularisée sur une matrice X avec NaN.
    Retourne la matrice complétée (même shape que X).

    Algorithme (Josse & Husson 2009) :
      ① Init par la moyenne de chaque colonne
      ② Boucle :
         (a) SVD tronquée à ncp composantes
         (b) Régularisation des valeurs singulières
         (c) Reconstruction X̂ = M + U·D_reg·V'
         (d) Mise à jour des cases manquantes uniquement
         (e) Mise à jour des moyennes
      ③ Convergence sur ||X^l - X^(l-1)||² < eps
    """
    X = X.copy().astype(float)
    n, p = X.shape
    R = (~np.isnan(X)).astype(float)  # masque : 1=observé, 0=manquant

    # Init : remplacement par la moyenne de chaque colonne
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j in range(p):
        X[np.isnan(X[:, j]), j] = col_means[j]

    X_prev = X.copy()

    for _ in range(n_iter_max):
        # Centrage
        means = X.mean(axis=0)
        X_centered = X - means

        # SVD tronquée
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U = U[:, :ncp]
        s = s[:ncp]
        Vt = Vt[:ncp, :]

        # Régularisation des valeurs singulières (slide 32)
        if regularized and ncp < p:
            # Estimation de sigma² à partir des composantes non retenues
            s_full = np.linalg.svd(X_centered, compute_uv=False)
            rss = np.sum(s_full[ncp:] ** 2)
            ddl = (n - 1 - ncp) * (p - ncp)
            sigma2 = rss / ddl if ddl > 0 else 0.0
            s_reg = np.maximum(s ** 2 - sigma2, 0) / np.maximum(s, 1e-10)
        else:
            s_reg = s

        # Reconstruction
        X_hat = means + (U * s_reg) @ Vt

        # Mise à jour : on ne touche que les cases manquantes
        X_new = R * X_prev + (1 - R) * X_hat

        # Mise à jour des moyennes pour la prochaine itération
        X = X_new.copy()

        # Convergence
        diff = np.sum((X_new - X_prev) ** 2)
        norm = np.sum(X_prev ** 2) + 1e-10
        if diff / norm < eps:
            break

        X_prev = X_new.copy()

    return X


def choose_ncp(
    X: np.ndarray,
    ncp_max: int = 5,
    n_folds: int = 5,
) -> tuple[int, list]:
    """
    Choisit le nombre de composantes optimal par validation croisée (MSEP).
    Retourne (ncp_optimal, liste des MSEP pour ncp=1..ncp_max).
    """
    # Colonnes sans aucun NaN pour construire une référence
    complete_cols = ~np.isnan(X).any(axis=0)
    X_ref = X[:, complete_cols]
    if X_ref.shape[1] < 2:
        # Pas assez de colonnes complètes → on prend 2 par défaut
        return 2, []

    # Normalisation
    means = np.nanmean(X_ref, axis=0)
    stds = np.nanstd(X_ref, axis=0)
    stds[stds == 0] = 1.0
    X_std = (X_ref - means) / stds

    ncp_max = min(ncp_max, X_std.shape[1] - 1, X_std.shape[0] - 2)
    msep_values = []
    for ncp in range(1, ncp_max + 1):
        msep = _msep_cv(X_std, ncp=ncp, n_folds=n_folds)
        msep_values.append(msep)

    best_ncp = int(np.argmin(msep_values)) + 1
    return best_ncp, msep_values


def impute_pca(
    df: pd.DataFrame,
    quanti_cols: list,
    ncp: int = None,
    ncp_max: int = 5,
    regularized: bool = True,
    n_iter_max: int = 1000,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute les valeurs manquantes des colonnes quantitatives
    par ACP itérative régularisée.
    Si ncp=None, le nombre de composantes est choisi par validation croisée.
    Retourne (df_imputé, rapport).
    """
    df = df.copy()
    sub = df[quanti_cols].copy()
    X = sub.values.astype(float)

    # Normalisation avant imputation (obligatoire pour que l'ACP soit cohérente)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds

    msep_values = []
    if ncp is None:
        ncp, msep_values = choose_ncp(X, ncp_max=ncp_max)

    X_imp_std = _iterative_pca(X_std, ncp=ncp, regularized=regularized,
                               n_iter_max=n_iter_max, eps=eps)

    # Dénormalisation
    X_imp = X_imp_std * stds + means

    # Clip post-imputation : les valeurs nutritionnelles sont toujours >= 0
    # et respectent les bornes physiques connues
    for j, col in enumerate(quanti_cols):
        low = NUTRITIONAL_BOUNDS.get(col, (0, None))[0]
        high = NUTRITIONAL_BOUNDS.get(col, (None, None))[1]
        if low is not None:
            X_imp[:, j] = np.maximum(X_imp[:, j], low)
        if high is not None:
            X_imp[:, j] = np.minimum(X_imp[:, j], high)

    df[quanti_cols] = X_imp

    rapport = {
        "ncp_used": ncp,
        "msep_values": msep_values,
        "n_imputed": int(sub.isnull().sum().sum()),
    }
    return df, rapport


# ---------------------------------------------------------------------------
# 8. Imputation des variables qualitatives
# ---------------------------------------------------------------------------

def impute_quali(
    df: pd.DataFrame,
    quali_cols: list,
    quanti_cols: list = None,
    method: str = "mode",
    k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute les variables qualitatives par mode ou KNN.
    - method='mode' : remplace par la modalité la plus fréquente
    - method='knn'  : KNN sur les variables quantitatives déjà imputées
    Retourne (df_imputé, rapport).
    """
    df = df.copy()
    rapport = {}

    cols_with_missing = [c for c in quali_cols if c in df.columns and df[c].isnull().any()]

    if method == "mode":
        for col in cols_with_missing:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                rapport[col] = {"method": "mode", "value": mode_val[0]}

    elif method == "knn" and quanti_cols:
        available_quanti = [c for c in quanti_cols if c in df.columns and df[c].isnull().sum() == 0]
        if len(available_quanti) < 2:
            # Fallback sur le mode si pas assez de variables quanti disponibles
            return impute_quali(df, quali_cols, method="mode")

        scaler = StandardScaler()
        X_quanti = scaler.fit_transform(df[available_quanti])

        for col in cols_with_missing:
            mask_obs = df[col].notna()
            mask_miss = df[col].isna()

            if mask_obs.sum() < k:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "unknown")
                rapport[col] = {"method": "mode_fallback"}
                continue

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_quanti[mask_obs], df.loc[mask_obs, col])
            df.loc[mask_miss, col] = knn.predict(X_quanti[mask_miss])
            rapport[col] = {"method": "knn", "k": k}

    return df, rapport


# ---------------------------------------------------------------------------
# 9. Recodage et création de variables
# ---------------------------------------------------------------------------

def recode_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées utiles pour l'analyse.
    Toutes les créations sont conditionnelles à la présence de la colonne source.
    """
    df = df.copy()

    if "nova_group" in df.columns:
        df["ultra_transforme"] = (
            pd.to_numeric(df["nova_group"], errors="coerce") == 4
        ).map({True: "oui", False: "non"})

    if "nutriscore_grade" in df.columns:
        grade_map = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        df["nutriscore_num"] = (
            df["nutriscore_grade"].str.lower().map(grade_map)
        )

    if "additives_n" in df.columns:
        df["additives_cat"] = pd.cut(
            pd.to_numeric(df["additives_n"], errors="coerce"),
            bins=[-1, 0, 3, 7, np.inf],
            labels=["aucun", "peu (1-3)", "modéré (4-7)", "beaucoup (8+)"],
        )

    return df


# ---------------------------------------------------------------------------
# 10. Pipeline complet
# ---------------------------------------------------------------------------

def preprocess(
    filepath: str,
    seuil_col: float = 0.70,
    seuil_ligne: float = 0.50,
    outlier_method: str = "clip",
    imputation_method_quali: str = "mode",
    ncp: int = None,
    ncp_max: int = 5,
    extra_exclude_cols: list = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline complet de préparation des données.

    Paramètres
    ----------
    filepath : str
        Chemin vers le fichier Excel ou CSV.
    seuil_col : float
        Taux de NaN au-delà duquel une colonne est supprimée (défaut 0.70).
    seuil_ligne : float
        Taux de NaN au-delà duquel une ligne est supprimée (défaut 0.50).
    outlier_method : str
        'clip' (winsorisation) ou 'remove' (mise à NaN).
    imputation_method_quali : str
        'mode' ou 'knn'.
    ncp : int or None
        Nombre de composantes pour l'imputation ACP. None = choix auto par CV.
    ncp_max : int
        Nombre max de composantes testé lors de la CV.
    extra_exclude_cols : list
        Colonnes supplémentaires à exclure (en plus des metadata).

    Retourne
    --------
    df_clean : pd.DataFrame
        Données nettoyées et imputées.
    rapport : dict
        Rapport détaillé de chaque étape.
    """
    rapport = {}

    # 1. Chargement
    df = load_data(filepath)
    rapport["shape_init"] = df.shape

    # 2. Exclusion metadata
    exclude = METADATA_COLS.copy()
    if extra_exclude_cols:
        exclude.update(extra_exclude_cols)
    df = df.drop(columns=[c for c in exclude if c in df.columns])

    # 3. Détection des types
    col_types = detect_column_types(df)
    rapport["col_types"] = {k: len(v) for k, v in col_types.items()}

    # 4. Diagnostic manquantes (avant filtrage)
    rapport["missing_before"] = diagnose_missing(df).to_dict()

    # 5. Filtrage colonnes / lignes
    df, rapport_filter = filter_missing(df, seuil_col=seuil_col, seuil_ligne=seuil_ligne)
    rapport["filter"] = rapport_filter

    # Mise à jour des types après filtrage
    col_types = detect_column_types(df)
    quanti_cols = col_types["quanti"]
    quali_cols = col_types["quali"]

    # 6. Doublons
    df, n_dupl = remove_duplicates(df)
    rapport["duplicates_removed"] = n_dupl

    # 7. Valeurs aberrantes
    df, rapport_outliers = handle_outliers(df, quanti_cols, method=outlier_method)
    rapport["outliers"] = rapport_outliers

    # 8. Imputation quantitative (ACP itérative régularisée)
    quanti_with_missing = [c for c in quanti_cols if df[c].isnull().any()]
    if quanti_with_missing:
        df, rapport_pca = impute_pca(df, quanti_with_missing, ncp=ncp, ncp_max=ncp_max)
        rapport["imputation_quanti"] = rapport_pca

    # 9. Imputation qualitative
    quali_with_missing = [c for c in quali_cols if df[c].isnull().any()]
    if quali_with_missing:
        df, rapport_quali = impute_quali(
            df, quali_with_missing,
            quanti_cols=quanti_cols,
            method=imputation_method_quali,
        )
        rapport["imputation_quali"] = rapport_quali

    # 10. Recodage
    df = recode_variables(df)

    rapport["shape_final"] = df.shape
    rapport["missing_after"] = int(df.isnull().sum().sum())

    return df, rapport
