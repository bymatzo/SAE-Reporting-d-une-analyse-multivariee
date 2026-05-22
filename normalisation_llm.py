"""
Normalisation multilingue des variables qualitatives via Claude API.

Colonnes traitées :
  - packaging, categories, labels  : texte libre multilingue → français
  - countries                      : codes/noms pays → français (lookup + LLM)
  - allergens, traces              : format "en:gluten" ou "fr:Avoine" → français unifié

Stratégie :
  1. Splitter chaque cellule par virgule → tokens uniques
  2. Lookup dans les dicts statiques (COUNTRY_MAP, ALLERGEN_MAP)
  3. LLM (Claude Haiku) pour les tokens inconnus, en batches de 60
  4. Cache disque JSON pour éviter de re-appeler l'API à chaque run
  5. Reconstruire les cellules depuis le mapping complet
"""

import json
import re
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # charge ANTHROPIC_API_KEY depuis .env si présent

# ---------------------------------------------------------------------------
# Cache disque
# ---------------------------------------------------------------------------

CACHE_PATH = Path(".llm_translation_cache.json")
LANG_PREFIX = re.compile(r"^[a-z]{2,3}:")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Dictionnaires statiques (évitent des appels LLM inutiles)
# ---------------------------------------------------------------------------

COUNTRY_MAP: dict[str, str] = {
    # codes préfixés
    "en:france": "France", "en:fr": "France", "fr": "France",
    "en:germany": "Allemagne", "en:de": "Allemagne",
    "en:united-kingdom": "Royaume-Uni", "en:gb": "Royaume-Uni",
    "en:spain": "Espagne", "en:es": "Espagne",
    "en:italy": "Italie", "en:it": "Italie",
    "en:belgium": "Belgique", "en:be": "Belgique",
    "en:switzerland": "Suisse", "en:ch": "Suisse",
    "en:netherlands": "Pays-Bas", "en:nl": "Pays-Bas",
    "en:united-states": "États-Unis", "en:us": "États-Unis",
    "en:canada": "Canada",
    "en:poland": "Pologne", "en:pl": "Pologne",
    "en:austria": "Autriche", "en:at": "Autriche",
    "en:portugal": "Portugal", "en:pt": "Portugal",
    "en:sweden": "Suède", "en:se": "Suède",
    "en:denmark": "Danemark", "en:dk": "Danemark",
    "en:finland": "Finlande", "en:fi": "Finlande",
    "en:ireland": "Irlande", "en:ie": "Irlande",
    "en:luxembourg": "Luxembourg", "en:lu": "Luxembourg",
    "en:czech-republic": "République Tchèque", "en:cz": "République Tchèque",
    "en:hungary": "Hongrie", "en:hu": "Hongrie",
    "en:romania": "Roumanie", "en:ro": "Roumanie",
    "en:slovakia": "Slovaquie", "en:sk": "Slovaquie",
    "en:greece": "Grèce", "en:gr": "Grèce",
    "en:norway": "Norvège", "en:no": "Norvège",
    "en:australia": "Australie", "en:au": "Australie",
    # noms anglais sans préfixe
    "united kingdom": "Royaume-Uni", "germany": "Allemagne",
    "spain": "Espagne", "italy": "Italie", "netherlands": "Pays-Bas",
    "belgium": "Belgique", "switzerland": "Suisse", "austria": "Autriche",
    "united states": "États-Unis", "canada": "Canada",
    "sweden": "Suède", "denmark": "Danemark", "norway": "Norvège",
    "finland": "Finlande", "ireland": "Irlande",
    "portugal": "Portugal", "poland": "Pologne", "greece": "Grèce",
    "australia": "Australie",
}

ALLERGEN_MAP: dict[str, str] = {
    "gluten": "Gluten",
    "milk": "Lait", "lait": "Lait",
    "eggs": "Œufs", "egg": "Œufs", "oeufs": "Œufs",
    "peanuts": "Arachides", "peanut": "Arachides", "arachides": "Arachides",
    "nuts": "Fruits à coque", "tree-nuts": "Fruits à coque",
    "soybeans": "Soja", "soybean": "Soja", "soy": "Soja", "soja": "Soja",
    "wheat": "Blé", "ble": "Blé",
    "sesame-seeds": "Sésame", "sesame": "Sésame", "sésame": "Sésame",
    "fish": "Poisson", "poisson": "Poisson",
    "shellfish": "Crustacés", "crustacés": "Crustacés",
    "celery": "Céleri", "céleri": "Céleri",
    "mustard": "Moutarde", "moutarde": "Moutarde",
    "lupins": "Lupins", "lupin": "Lupins",
    "molluscs": "Mollusques",
    "sulphur-dioxide-and-sulphites": "Dioxyde de soufre et sulfites",
    "sulphites": "Sulfites", "sulfites": "Sulfites",
    "gelatin": "Gélatine", "gélatine": "Gélatine",
    "barley": "Orge", "orge": "Orge",
    "oats": "Avoine", "avoine": "Avoine",
    "rye": "Seigle", "seigle": "Seigle",
    "spelt": "Épeautre",
    "gluten-de-ble": "Gluten de blé",
    "none": None,  # → supprimé
}

# Tokens à ignorer (codes internes, valeurs non significatives)
_DISCARD_PATTERN = re.compile(r"^\d+$|^[a-z]{2}:\d+")


# ---------------------------------------------------------------------------
# Appel LLM
# ---------------------------------------------------------------------------

def _call_llm(tokens: list[str], context: str, client: Anthropic) -> dict[str, str]:
    """
    Traduit une liste de tokens en français via Claude Haiku.
    Retourne un dict {token_original: traduction_fr}.
    Utilise le cache système pour réduire les coûts (prompt caching).
    """
    if not tokens:
        return {}

    system = (
        "Tu es un traducteur expert en données alimentaires. "
        "Tu reçois une liste de valeurs issues d'une base Open Food Facts. "
        "Traduis chaque valeur en français naturel, en minuscules sauf noms propres. "
        "Normalise les variantes (ex: 'Cardboard'='Carton', 'Box'='Boîte'). "
        "Si la valeur est déjà correcte en français, retourne-la telle quelle. "
        "Réponds UNIQUEMENT avec un objet JSON valide, sans commentaire."
    )

    user = (
        f'Colonne : "{context}"\n'
        f"Valeurs :\n{json.dumps(tokens, ensure_ascii=False)}\n\n"
        'Réponds avec : {"valeur_originale": "traduction_fr", ...}'
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        betas=["prompt-caching-2024-07-31"],
    )

    raw = resp.content[0].text
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback : retourne les originaux sans traduction
    return {t: t for t in tokens}


# ---------------------------------------------------------------------------
# Normalisation d'une colonne
# ---------------------------------------------------------------------------

def _normalize_column(
    series: pd.Series,
    col_name: str,
    base_map: dict[str, str],
    client: Anthropic,
    cache: dict,
    batch_size: int = 60,
) -> pd.Series:
    """
    Normalise une colonne à valeurs multiples (séparées par virgule).
    Étapes : lookup statique → cache disque → LLM pour les inconnus.
    """
    col_cache: dict = cache.setdefault(col_name, {})

    # 1. Collecter les tokens uniques
    unique_tokens: list[str] = (
        series.dropna()
        .str.split(r",\s*")
        .explode()
        .str.strip()
        .dropna()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    # 2. Construire le mapping complet
    mapping: dict[str, str | None] = {}
    to_translate: list[str] = []

    for tok in unique_tokens:
        key = tok.lower()
        stripped = LANG_PREFIX.sub("", tok.strip())
        stripped_lower = stripped.lower()

        # Tokens à ignorer (codes numériques, etc.)
        if _DISCARD_PATTERN.match(tok):
            mapping[tok] = None
            continue

        # Priorité : dict statique (avec ou sans préfixe) → cache → LLM
        if key in base_map:
            mapping[tok] = base_map[key]
        elif stripped_lower in base_map:
            mapping[tok] = base_map[stripped_lower]
        elif key in col_cache:
            mapping[tok] = col_cache[key]
        elif stripped_lower in col_cache:
            mapping[tok] = col_cache[stripped_lower]
        else:
            to_translate.append(tok)

    # 3. Appels LLM par batch pour les tokens inconnus
    if to_translate:
        for i in range(0, len(to_translate), batch_size):
            batch = to_translate[i: i + batch_size]
            translations = _call_llm(batch, col_name, client)
            for orig, fr in translations.items():
                col_cache[orig.lower()] = fr
                mapping[orig] = fr
        _save_cache(cache)

    # 4. Reconstruire les cellules
    def _rebuild(cell) -> str | float:
        if pd.isna(cell):
            return cell
        parts = [t.strip() for t in str(cell).split(",") if t.strip()]
        translated: list[str] = []
        seen: set[str] = set()
        for tok in parts:
            fr = mapping.get(tok, LANG_PREFIX.sub("", tok).strip())
            if fr is None:          # token à supprimer (ex: "none")
                continue
            fr_norm = fr.strip()
            if fr_norm.lower() not in seen and fr_norm:
                seen.add(fr_norm.lower())
                translated.append(fr_norm)
        return ", ".join(translated) if translated else float("nan")

    return series.apply(_rebuild)


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def normalize_multilingual_columns(
    df: pd.DataFrame,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Normalise les colonnes multilingues du DataFrame en français.

    Colonnes traitées (si présentes) :
      packaging, categories, labels, countries → traduction LLM + lookup
      allergens, traces                        → retrait préfixe langue + lookup + LLM

    Le cache des traductions est sauvegardé dans .llm_translation_cache.json
    pour éviter de re-appeler l'API à chaque exécution.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame issu du preprocessing (avant ou après imputation).
    api_key : str | None
        Clé API Anthropic. Si None, utilise la variable d'env ANTHROPIC_API_KEY.

    Retourne
    --------
    pd.DataFrame avec les colonnes cibles normalisées.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    cache = _load_cache()
    df = df.copy()

    text_cols = {
        "packaging":  {},
        "categories": {},
        "labels":     {},
        "countries":  COUNTRY_MAP,
    }
    allergen_cols = ["allergens", "traces"]

    for col, base_map in text_cols.items():
        if col in df.columns:
            print(f"  → Normalisation {col}…")
            df[col] = _normalize_column(df[col], col, base_map, client, cache)

    for col in allergen_cols:
        if col in df.columns:
            print(f"  → Normalisation {col}…")
            df[col] = _normalize_column(df[col], col, ALLERGEN_MAP, client, cache)

    return df


# ---------------------------------------------------------------------------
# Prompt système partagé pour les fonctions d'interprétation
# ---------------------------------------------------------------------------

_INTERP_SYSTEM = (
    "Tu es un statisticien expert en analyse multivariée appliquée à l'agroalimentaire. "
    "Tu interprètes des résultats d'ACP, d'ACM ou de k-moyennes de manière concise et pertinente. "
    "Réponds UNIQUEMENT avec un objet JSON valide selon le format demandé, sans commentaire ni markdown."
)


def _interp_call(user_prompt: str) -> str:
    """Appel LLM unique pour toutes les fonctions d'interprétation."""
    client = Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=[{"type": "text", "text": _INTERP_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
        betas=["prompt-caching-2024-07-31"],
    )
    return resp.content[0].text


def _parse_json(text: str) -> dict | list | None:
    match = re.search(r"[\[{][\s\S]*[\]}]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Interprétation des axes ACP
# ---------------------------------------------------------------------------

def interpret_pca_axes(
    loadings: pd.DataFrame,
    explained_var: pd.Series,
    n_axes: int = 2,
    context: str = "données alimentaires Open Food Facts",
) -> dict:
    """
    Génère un nom et une interprétation pour chaque axe ACP.

    Paramètres
    ----------
    loadings : DataFrame (variables × dimensions) — corrélations variables/axes
    explained_var : Series — % variance par axe
    n_axes : int — nombre d'axes à interpréter

    Retourne
    --------
    dict {"Dim 1": {"name": "...", "interpretation": "..."}, ...}
    """
    axes_data = {}
    for i in range(min(n_axes, len(explained_var))):
        dim = f"Dim {i + 1}"
        col = loadings[dim].sort_values()
        axes_data[dim] = {
            "variance_pct": round(float(explained_var.iloc[i]), 1),
            "pole_positif": {str(k): round(float(v), 3) for k, v in col.tail(4).items()},
            "pole_negatif": {str(k): round(float(v), 3) for k, v in col.head(4).items()},
        }

    prompt = (
        f"Contexte : {context}\n\n"
        f"Résultats ACP — loadings (corrélation variable/axe, entre -1 et 1) :\n"
        f"{json.dumps(axes_data, ensure_ascii=False, indent=2)}\n\n"
        "Pour chaque axe donne :\n"
        "- name : un titre court (3-6 mots) qui résume l'opposition entre les deux pôles\n"
        "- interpretation : 1-2 phrases expliquant ce que l'axe mesure\n\n"
        'Format : {"Dim 1": {"name": "...", "interpretation": "..."}, "Dim 2": ...}'
    )
    result = _parse_json(_interp_call(prompt))
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Interprétation des axes ACM
# ---------------------------------------------------------------------------

def interpret_mca_axes(
    col_coords: pd.DataFrame,
    explained_var: pd.Series,
    n_axes: int = 2,
    context: str = "données alimentaires Open Food Facts",
) -> dict:
    """
    Génère un nom et une interprétation pour chaque axe ACM.

    Paramètres
    ----------
    col_coords : DataFrame (modalités × dimensions) — coordonnées des modalités
    explained_var : Series — % inertie par axe

    Retourne
    --------
    dict {"Dim 1": {"name": "...", "interpretation": "..."}, ...}
    """
    def _get_var(label: str) -> str:
        return label.split("__")[0] if "__" in label else "?"

    axes_data = {}
    for i in range(min(n_axes, len(explained_var))):
        dim = f"Dim {i + 1}"
        col = col_coords[dim].sort_values()
        # Format "variable: modalité (coord)"
        def _fmt(row):
            lbl = str(row.name)
            var = _get_var(lbl)
            mod = lbl.split("__")[-1] if "__" in lbl else lbl
            return f"{var}: {mod}"

        axes_data[dim] = {
            "inertie_pct": round(float(explained_var.iloc[i]), 1),
            "pole_positif": [_fmt(col.iloc[-(j+1)]) for j in range(min(4, len(col)))],
            "pole_negatif": [_fmt(col.iloc[j]) for j in range(min(4, len(col)))],
        }

    prompt = (
        f"Contexte : {context}\n\n"
        f"Résultats ACM — modalités aux extrémités de chaque axe :\n"
        f"{json.dumps(axes_data, ensure_ascii=False, indent=2)}\n\n"
        "Pour chaque axe donne :\n"
        "- name : un titre court (3-6 mots) résumant l'opposition\n"
        "- interpretation : 1-2 phrases expliquant ce que l'axe distingue\n\n"
        'Format : {"Dim 1": {"name": "...", "interpretation": "..."}, "Dim 2": ...}'
    )
    result = _parse_json(_interp_call(prompt))
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Interprétation des clusters (k-moyennes)
# ---------------------------------------------------------------------------

def interpret_clusters(
    quanti_profile: pd.DataFrame,
    quali_profile: dict | None = None,
    cluster_sizes: pd.Series | None = None,
    context: str = "céréales alimentaires",
) -> dict:
    """
    Génère un nom et une description naturelle pour chaque cluster.

    Paramètres
    ----------
    quanti_profile : DataFrame (clusters × variables) — moyennes par cluster
    quali_profile  : dict col → DataFrame fréquences par cluster (optionnel)
    cluster_sizes  : Series — taille de chaque cluster (optionnel)

    Retourne
    --------
    dict {"1": {"name": "...", "description": "..."}, ...}
    """
    # Z-scores pour identifier les traits distinctifs (indépendamment de l'unité)
    means = quanti_profile.mean()
    stds = quanti_profile.std().replace(0, 1)
    z = ((quanti_profile - means) / stds).round(2)

    cluster_summaries = {}
    for cluster in quanti_profile.index:
        row = z.loc[cluster]
        summary: dict = {
            "taille": int(cluster_sizes[cluster]) if cluster_sizes is not None else None,
            "traits_eleves": {str(k): float(v) for k, v in row.nlargest(4).items()},
            "traits_faibles": {str(k): float(v) for k, v in row.nsmallest(4).items()},
        }
        if quali_profile:
            for col, ct in quali_profile.items():
                if cluster in ct.index:
                    dom = ct.loc[cluster].idxmax()
                    pct = float(ct.loc[cluster].max())
                    summary[f"modal_{col}"] = f"{dom} ({pct:.0f}%)"
        cluster_summaries[str(cluster)] = summary

    prompt = (
        f"Contexte : classification de {context} par k-moyennes.\n"
        "Les valeurs sont des z-scores (0 = moyenne globale, +1 = 1 écart-type au-dessus).\n\n"
        f"{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}\n\n"
        "Pour chaque cluster donne :\n"
        "- name : un nom court et imagé (ex: 'Céréales sportives', 'Ultra-sucrées')\n"
        "- description : 1-2 phrases décrivant le profil type du produit dans ce groupe\n\n"
        'Format : {"1": {"name": "...", "description": "..."}, "2": ...}'
    )
    result = _parse_json(_interp_call(prompt))
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Détection automatique des colonnes multilingues
# ---------------------------------------------------------------------------

def detect_multilingual_columns(
    df: pd.DataFrame,
    n_sample: int = 20,
) -> list[str]:
    """
    Détecte automatiquement les colonnes texte ayant des valeurs multilingues
    ou hétérogènes (préfixes 'en:'/'fr:', termes identiques en plusieurs langues,
    codes ISO pays, etc.).

    Utile pour appliquer normalize_multilingual_columns sur n'importe quel dataset.

    Paramètres
    ----------
    df : pd.DataFrame
    n_sample : int — nb de valeurs uniques échantillonnées par colonne

    Retourne
    --------
    list[str] — noms des colonnes à normaliser
    """
    client = Anthropic()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if not text_cols:
        return []

    samples = {}
    for col in text_cols:
        unique_vals = df[col].dropna().unique()
        samples[col] = [str(v) for v in unique_vals[:n_sample]]

    prompt = (
        "Tu examines des colonnes d'un jeu de données alimentaire.\n"
        "Identifie celles qui présentent des problèmes de langue ou d'hétérogénéité :\n"
        "  - valeurs en plusieurs langues (ex: 'Milk' et 'Lait' dans la même colonne)\n"
        "  - préfixes de langue ('en:', 'fr:', 'de:', etc.)\n"
        "  - codes ISO pays (en:fr, en:gb…)\n"
        "  - termes identiques écrits différemment selon la langue\n\n"
        f"Colonnes et échantillons :\n{json.dumps(samples, ensure_ascii=False, indent=2)}\n\n"
        'Réponds UNIQUEMENT avec une liste JSON : ["col1", "col2", ...]'
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    result = _parse_json(resp.content[0].text)
    return result if isinstance(result, list) else []
