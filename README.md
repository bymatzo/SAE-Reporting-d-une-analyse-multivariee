# SAE 4.02 — Reporting d'une analyse multivariée
**Analyse de l'offre de céréales vendues en France**
Université Lumière Lyon 2 — Adrien Guille

---

## Contexte

Ce projet s'appuie sur un extrait de la base [Open Food Facts](https://world.openfoodfacts.org/) : 7 374 céréales pour petit-déjeuner commercialisées en France, décrites par 210 variables (valeurs nutritionnelles, scores Nutri-Score / NOVA / Eco-Score, ingrédients, marques…).

L'objectif est de déchiffrer ce marché via des méthodes d'analyse multivariée (ACP, ACM, AFC) et de classification non supervisée (CAH, k-moyennes), le tout exposé dans une interface interactive générique.

---

## Stack technique

- **Langage** : Python 3.12
- **Interface** : Streamlit
- **Analyse** : scikit-learn, prince, numpy, pandas
- **Visualisation** : Plotly, matplotlib, seaborn
- **LLM** : Anthropic Claude Haiku (normalisation multilingue + interprétation)

---

## Structure du projet

```
.
├── app.py                   # interface Streamlit (point d'entrée)
├── preprocessing.py         # chargement, nettoyage, imputation
├── analysis.py              # ACP, ACM, AFC, CAH, k-moyennes
├── visualisation.py         # graphiques Plotly
├── normalisation_llm.py     # normalisation multilingue via Claude + interprétation LLM
├── requirements.txt
├── .env                     # clé ANTHROPIC_API_KEY (non versionné)
├── Données/
│   └── openfoodfacts_cereals_fr.xlsx
└── Sujet/
    ├── Sujet 2026 - Reporting dun analyse multivariée.pdf
    └── cours_missMDA_Spé_Stat.pdf
```

---

## Avancement

| Module | Branche | Statut |
|--------|---------|--------|
| `preprocessing.py` | `preprocessing` | Terminé |
| `analysis.py` | `analysis` | Terminé |
| `visualisation.py` | `visualisation` | Terminé |
| `app.py` | `app` | Terminé |
| `normalisation_llm.py` | `llm-normalisation` | Terminé |

---

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour activer la normalisation multilingue via LLM, créer un fichier `.env` à la racine :

```
ANTHROPIC_API_KEY=sk-ant-...
```

L'interface accepte n'importe quel fichier Excel ou CSV issu d'Open Food Facts — aucun nom de colonne n'est codé en dur.

---

## Scénarios d'analyse

### Scénario 1 — Profils nutritionnels (ACP + classification)
- **Variables actives** : énergie, lipides, acides gras saturés, glucides, sucres, fibres, protéines, sel
- **Variables illustratives** : Nutri-Score, groupe NOVA, marque
- **Méthodes** : ACP → CAH (Ward) + k-moyennes
- **Objectif métier** : identifier des familles de céréales (sportives, ultra-sucrées, riches en fibres…)

### Scénario 2 — Cohérence des scores (ACM)
- **Variables actives** : `nutriscore_grade`, `nova_group`, `environmental_score_grade`
- **Variables illustratives** : présence de labels bio, nombre d'additifs, marque
- **Méthode** : ACM
- **Objectif métier** : un produit bon nutritionnellement est-il aussi bon pour l'environnement ?

---

## preprocessing.py — Pipeline de traitement des données

### Vue d'ensemble

Le pipeline (`preprocess()`) enchaîne 11 étapes paramétrables, toutes avec rapport d'exécution :

```
Fichier brut (210 cols, 7374 lignes)
    ↓ exclusion metadata
    ↓ filtrage colonnes > 70 % NaN
    ↓ filtrage lignes > 50 % NaN
    ↓ dédoublonnage
    ↓ traitement des outliers
    ↓ imputation quanti (ACP itérative régularisée)
    ↓ imputation quali (mode ou KNN)
    ↓ normalisation multilingue LLM (optionnelle)
    ↓ recodage / variables dérivées
Données prêtes pour l'analyse (39 cols, ~6452 lignes)
```

### Fonctions disponibles

| Fonction | Rôle |
|----------|------|
| `load_data(filepath)` | Lecture générique Excel ou CSV |
| `detect_column_types(df)` | Détection automatique quanti / quali |
| `diagnose_missing(df)` | Taux de NaN par colonne, trié |
| `missing_pattern_matrix(df)` | Matrice présence/absence pour visualisation |
| `filter_missing(df, seuil_col, seuil_ligne)` | Suppression colonnes et lignes trop vides |
| `remove_duplicates(df)` | Dédoublonnage, conserve la ligne la plus complète |
| `handle_outliers(df, cols, method)` | Winsorisation ou suppression (bornes physiques / IQR×3) |
| `impute_pca(df, cols, ncp)` | Imputation ACP itérative régularisée (missMDA) |
| `choose_ncp(X, ncp_max)` | Choix du nombre de composantes par validation croisée (MSEP) |
| `impute_quali(df, cols, method)` | Imputation par mode ou KNN |
| `recode_variables(df)` | Création de `ultra_transforme`, `nutriscore_num`, `additives_cat` |
| `preprocess(filepath, **params)` | Pipeline complet en un appel |

### Résultats sur les données céréales

| Étape | Avant | Après |
|-------|-------|-------|
| Lignes | 7 374 | 6 452 |
| Colonnes | 210 | 39 |
| Valeurs imputées (quanti) | — | 19 825 |
| NaN restants | — | 269 |

---

## Gestion des valeurs manquantes — détail

### 1. Diagnostic préalable

Avant toute suppression, `diagnose_missing()` produit un tableau du taux de NaN par colonne et `missing_pattern_matrix()` génère une matrice binaire (0 = observé, 1 = manquant) permettant de visualiser les co-absences et de distinguer :

- **MCAR** (Missing Completely At Random) : absence aléatoire, sans lien avec les autres variables
- **MAR** (Missing At Random) : absence liée à d'autres variables observées
- **MNAR** (Missing Not At Random) : absence liée à la valeur elle-même (ex. valeurs extrêmes non renseignées)

### 2. Filtrage des colonnes et lignes trop vides

```python
filter_missing(df, seuil_col=0.70, seuil_ligne=0.50)
```

- Supprime les colonnes ayant **plus de 70 %** de NaN (information trop pauvre pour imputer)
- Supprime ensuite les lignes ayant **plus de 50 %** de NaN restants (individus non informatifs)
- Résultat : 210 → 39 colonnes, 7 374 → ~6 800 lignes (avant dédoublonnage)

### 3. Détection et traitement des valeurs aberrantes

```python
handle_outliers(df, quanti_cols, method='clip')
```

Deux types de bornes selon la colonne :

- **Bornes physiques connues** pour les variables nutritionnelles (ex. `fat_100g` ∈ [0, 100]) : définies dans `NUTRITIONAL_BOUNDS`
- **IQR × 3** pour les autres variables quantitatives : `[Q1 − 3×IQR, Q3 + 3×IQR]`

Deux méthodes de traitement :
- `'clip'` (winsorisation) : les valeurs hors bornes sont ramenées à la borne la plus proche — recommandé pour conserver les individus
- `'remove'` : les valeurs aberrantes sont mises à NaN (elles seront ensuite imputées)

### 4. Imputation des variables quantitatives — ACP itérative régularisée (missMDA)

**Pourquoi l'ACP itérative plutôt qu'une imputation simple ?**

L'imputation par la moyenne ou la médiane ignore les corrélations entre variables (ex. `fat_100g` et `saturated-fat_100g` sont fortement liées). L'ACP itérative exploite la structure de covariance du jeu de données pour produire des imputations cohérentes avec l'ensemble des variables observées.

**Algorithme (Josse & Husson, 2009) :**

```
① Initialisation : remplacer les NaN par la moyenne de chaque colonne
② Centrer-réduire la matrice
③ Boucle jusqu'à convergence (||X^l − X^(l−1)||² / ||X^(l−1)||² < ε) :
    (a) SVD tronquée à ncp composantes → U, s, V'
    (b) Régularisation des valeurs singulières :
           s_reg = max(s² − σ², 0) / s
        où σ² est estimé depuis les composantes non retenues (résidu)
    (c) Reconstruction : X̂ = μ + U · diag(s_reg) · V'
    (d) Mise à jour des cases manquantes uniquement :
           X_new = R ⊙ X_obs + (1 − R) ⊙ X̂
④ Dénormalisation + clip post-imputation (bornes physiques)
```

La **régularisation** (étape 3b) est essentielle : sans elle, la SVD surreprésente le signal et sur-impute vers les valeurs extrêmes. Elle revient à rétrécir les valeurs singulières vers zéro à hauteur du bruit estimé.

**Choix automatique du nombre de composantes `ncp` :**

```python
choose_ncp(X, ncp_max=5)
```

La fonction évalue chaque `ncp` de 1 à `ncp_max` par une validation croisée approximée :
- On masque aléatoirement ~10 % des valeurs observées (graine fixe pour reproductibilité)
- On impute avec `_iterative_pca` pour chaque `ncp`
- On calcule le **MSEP** (Mean Squared Error of Prediction) sur les valeurs masquées
- On retient le `ncp` minimisant le MSEP

**Paramètres exposés dans l'interface :**

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `ncp` | `None` (auto) | Nombre de composantes ACP pour l'imputation |
| `ncp_max` | 5 | Borne supérieure du balayage par CV |
| `regularized` | `True` | Active la régularisation des valeurs singulières |
| `n_iter_max` | 1000 | Nombre max d'itérations |
| `eps` | 1e-6 | Seuil de convergence |

### 5. Imputation des variables qualitatives

```python
impute_quali(df, quali_cols, method='mode')  # ou method='knn'
```

- **Mode** : remplace les NaN par la modalité la plus fréquente — rapide, robuste, adapté quand la variable est peu liée aux autres
- **KNN** (k=5 par défaut) : classe les individus manquants en utilisant leurs voisins les plus proches dans l'espace des variables quantitatives déjà imputées — plus précis quand les corrélations quanti/quali sont fortes
- **Fallback automatique** vers le mode si moins de k observations complètes sont disponibles pour une colonne

---

## normalisation_llm.py — Normalisation multilingue & interprétation LLM

### Normalisation des colonnes multilingues

Open Food Facts contient des champs texte hétérogènes (plusieurs langues, préfixes `en:`/`fr:`, codes ISO) qui rendent l'analyse quali impossible sans harmonisation.

**Colonnes traitées :**

| Colonne | Problème typique | Stratégie |
|---------|-----------------|-----------|
| `packaging` | `"Cardboard"`, `"Karton"`, `"Carton"` | Lookup + LLM |
| `categories` | `"en:cereals"`, `"Céréales petit-déjeuner"` | Lookup + LLM |
| `labels` | `"en:organic"`, `"Bio"`, `"Biologisch"` | Lookup + LLM |
| `countries` | `"en:fr"`, `"France"`, `"United Kingdom"` | Dict statique (`COUNTRY_MAP`) + LLM |
| `allergens` | `"en:gluten"`, `"fr:Avoine"`, `"Milk"` | Dict statique (`ALLERGEN_MAP`) + LLM |
| `traces` | idem allergens | idem |

**Stratégie en 4 étapes (par colonne) :**

1. **Tokenisation** : chaque cellule est découpée par virgule → tokens uniques
2. **Lookup statique** : les tokens connus sont résolus depuis `COUNTRY_MAP` ou `ALLERGEN_MAP` sans appel LLM
3. **Cache disque** (`.llm_translation_cache.json`) : les tokens déjà traduits lors d'un run précédent sont réutilisés directement
4. **LLM (Claude Haiku)** en batches de 60 tokens : uniquement pour les tokens inconnus du lookup et du cache

Ce mécanisme réduit drastiquement les coûts API : après le premier run, la quasi-totalité des tokens est servie depuis le cache.

**Détection automatique des colonnes à normaliser :**

```python
detect_multilingual_columns(df, n_sample=20)
```

Envoie un échantillon de chaque colonne texte à Claude Haiku, qui identifie celles présentant des préfixes de langue, du code ISO ou du mélange de langues.

### Interprétation LLM des résultats d'analyse

Trois fonctions génèrent automatiquement des libellés naturels à partir des sorties statistiques :

| Fonction | Entrée | Sortie |
|----------|--------|--------|
| `interpret_pca_axes(loadings, explained_var)` | Corrélations variables/axes + % variance | Nom + interprétation pour chaque axe ACP |
| `interpret_mca_axes(col_coords, explained_var)` | Coordonnées modalités + % inertie | Nom + interprétation pour chaque axe ACM |
| `interpret_clusters(quanti_profile, quali_profile)` | Z-scores par cluster + modalités dominantes | Nom imagé + description pour chaque cluster |

Toutes utilisent Claude Haiku avec prompt mis en cache (réduction de latence et de coût sur les appels répétés). En cas d'indisponibilité de l'API (crédits insuffisants, clé invalide, réseau), une `LLMUnavailableError` est levée et l'interface affiche un message explicite sans planter.

---

## analysis.py — Fonctions disponibles

| Fonction | Méthode | Retourne |
|----------|---------|---------|
| `run_pca(df, active_cols, ...)` | ACP (sklearn) | scores, loadings (cercle corrélations), cos², contributions, éléments supplémentaires |
| `run_mca(df, active_cols, ...)` | ACM (prince) | coords individus & modalités, inertie, cos², contributions |
| `run_ca(df, col_row, col_col, ...)` | AFC (prince) | tableau de contingence, coords lignes & colonnes, inertie |
| `run_hca(df, cols, ...)` | CAH (scipy Ward) | matrice de linkage, labels, nombre de clusters suggéré |
| `run_kmeans(df, cols, k, ...)` | K-moyennes (sklearn) | labels, inertie, silhouette, centroïdes |
| `elbow(df, cols, k_range)` | Méthode du coude | inertie + silhouette pour chaque k, meilleur k |
| `describe_clusters(df, labels, ...)` | — | profil moyen par cluster (quanti + quali) |

### Résultats scénario 1 — ACP + K-moyennes (k=7)

| Cluster | Profil | Nutri-Score dominant |
|---------|--------|----------------------|
| 1 | Riches en fibres et protéines, peu sucrés | A (63%) |
| 2 | Très sucrés, peu gras, peu de fibres | D (48%) |
| 3 | Très riches en protéines, peu de glucides | A (49%) |
| 4 | Gras, sucrés, caloriques | D–E (83%) |
| 5 | Gras modéré, sucrés, caloriques | C–D (82%) |
| 6 | Peu caloriques (produits réduits) | A–C (82%) |
| 7 | Extrêmement salés (outliers) | E (100%) |

### Résultats scénario 2 — ACM (inertie Dim1+Dim2 : 49.5 %)
Axe 1 oppose les produits bons (Nutri-Score A, NOVA 1–2, Eco-Score A/A+) aux mauvais (Nutri-Score D–E, NOVA 4, Eco-Score D–F).

---

## visualisation.py — Fonctions disponibles

| Fonction | Graphique |
|----------|-----------|
| `plot_missing_heatmap(missing_summary)` | Barplot horizontal du taux de NaN par colonne |
| `plot_univariate(df, col)` | Histogramme (quanti) ou barplot (quali) |
| `plot_correlation_matrix(df, cols)` | Heatmap de la matrice de corrélation |
| `plot_boxplot(df, quanti_col, quali_col)` | Boxplot croisé quanti × quali |
| `plot_scree(explained_var)` | Éboulis des valeurs propres + variance cumulée |
| `plot_correlation_circle(loadings, ...)` | Cercle des corrélations (ACP) |
| `plot_individuals(scores, ...)` | Graphe des individus coloré (ACP ou ACM) |
| `plot_cos2_bar(cos2, dim)` | Barplot des cos² pour une dimension |
| `plot_mca_modalities(col_coords, ...)` | Graphe des modalités coloré par variable (ACM) |
| `plot_ca_biplot(row_coords, col_coords, ...)` | Biplot AFC (modalités lignes + colonnes) |
| `plot_dendrogram(linkage_matrix)` | Dendrogramme (CAH) |
| `plot_elbow(elbow_result)` | Courbe du coude — inertie + silhouette |
| `plot_cluster_profile(quanti_profile)` | Radar chart des profils de clusters |
| `plot_clusters_on_pca(scores, labels, ...)` | Clusters projetés sur le plan ACP |

---

## Livrables et calendrier

| Date | Livrable |
|------|----------|
| 8 juin 15h30 | Point obligatoire à l'IUT |
| 22 juin 17h30 | Dossier PDF (≤ 4 pages) sur Moodle |
| 22 juin 20h00 | Interface Streamlit sur Moodle |
| 24 juin 8h00 | Évaluation individuelle sur table |
| 24 juin 10h00 | Soutenance + démonstration |

---

## Références

- Josse, J. & Husson, F. (2012). Handling missing values in exploratory multivariate data analysis methods. *Journal de la Société Française de Statistique*.
- Josse, J. & Husson, F. (2009). GCV for PCA — choosing the number of components. *COMPSTAT*.
- Open Food Facts — [https://world.openfoodfacts.org/](https://world.openfoodfacts.org/)
