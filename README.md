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

---

## Structure du projet

```
.
├── app.py                  # point d'entrée Streamlit (à venir)
├── preprocessing.py        # chargement, nettoyage, imputation (fait)
├── analysis.py             # ACP, ACM, AFC, CAH, k-moyennes (à venir)
├── visualisation.py        # graphiques Plotly (à venir)
├── plan.json               # plan de développement du projet
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
| `analysis.py` | `analysis` | À faire |
| `visualisation.py` | `visualisation` | À faire |
| `app.py` | `app` | À faire |

---

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
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

## preprocessing.py — Fonctions disponibles

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

### Gestion des valeurs manquantes

Pipeline en 5 étapes (paramètres ajustables dans l'interface) :

1. Diagnostic visuel (heatmap + pattern de co-absence)
2. Suppression des colonnes avec > `seuil_col` % de NaN (défaut : 70 %)
3. Suppression des lignes avec > `seuil_ligne` % de NaN (défaut : 50 %)
4. **Imputation par ACP itérative régularisée** pour les variables quantitatives (algorithme missMDA — Josse & Husson, 2009) : initialisation par la moyenne, SVD tronquée, régularisation des valeurs singulières, convergence itérative
5. Imputation par mode ou KNN pour les variables qualitatives

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
- Open Food Facts — [https://world.openfoodfacts.org/](https://world.openfoodfacts.org/)
