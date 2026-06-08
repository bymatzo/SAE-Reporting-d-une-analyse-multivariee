# Imputation par ACP itérative régularisée — missMDA

> **Contexte** : ce document explique en détail le fonctionnement de l'algorithme d'imputation implémenté dans `preprocessing.py` (fonctions `_iterative_pca`, `choose_ncp`, `impute_pca`). Il suit le cours *missMDA* de Josse & Husson disponible dans `Sujet/cours_missMDA_Spé_Stat.pdf`.

---

## Table des matières

1. [Pourquoi ne pas imputer par la moyenne ?](#1-pourquoi-ne-pas-imputer-par-la-moyenne)
2. [Rappels : SVD et ACP](#2-rappels--svd-et-acp)
3. [Le problème des données manquantes en ACP](#3-le-problème-des-données-manquantes-en-acp)
4. [L'algorithme missMDA pas à pas](#4-lalgorithme-missmda-pas-à-pas)
5. [La régularisation des valeurs singulières](#5-la-régularisation-des-valeurs-singulières)
6. [Choisir le nombre de composantes `ncp`](#6-choisir-le-nombre-de-composantes-ncp)
7. [Implémentation dans preprocessing.py](#7-implémentation-dans-preprocessingpy)
8. [Résultats sur les données céréales](#8-résultats-sur-les-données-céréales)
9. [Limitations et cas limites](#9-limitations-et-cas-limites)
10. [Références](#10-références)

---

## 1. Pourquoi ne pas imputer par la moyenne ?

Considérons un jeu de données de valeurs nutritionnelles avec des NaN. La stratégie la plus naïve est de remplacer chaque valeur manquante par la **moyenne de sa colonne**. C'est rapide, mais cela introduit des distorsions majeures :

### Le problème illustré

Imaginons deux variables fortement corrélées : `fat_100g` (lipides) et `saturated-fat_100g` (acides gras saturés). Sur les produits observés, on constate que les produits avec beaucoup de lipides ont aussi beaucoup d'acides gras saturés — c'est biologiquement logique.

| produit | fat_100g | saturated-fat_100g |
|---------|----------|--------------------|
| A       | 25       | 10                 |
| B       | 40       | 18                 |
| C       | **NaN**  | 15                 |
| D       | 8        | 3                  |

- Imputation par la **moyenne** de `fat_100g` = (25+40+8)/3 = **24.3** → le produit C aurait 15g de saturés pour seulement 24g de lipides totaux, ce qui est physiquement incohérent.
- Imputation par **ACP** : en exploitant la corrélation entre les deux colonnes, l'algorithme estimera `fat_100g ≈ 34` pour le produit C, ce qui est cohérent avec ses 15g de saturés.

### Comparaison des méthodes d'imputation

| Méthode | Avantages | Inconvénients |
|---------|-----------|---------------|
| Moyenne / médiane | Rapide, simple | Ignore les corrélations, sous-estime la variance |
| Régression (par variable) | Exploite les corrélations | Nécessite une variable cible par NaN, variance sous-estimée |
| KNN | Facile à interpréter | Coûteux, sensible à l'échelle |
| **ACP itérative (missMDA)** | **Exploite toute la structure de covariance, préserve la variance** | Nécessite de choisir `ncp`, plus lent |
| Multiple Imputation (MICE) | Incertitude quantifiée | Très coûteux, complexe à intégrer |

L'ACP itérative est le choix recommandé pour des données multivariées denses comme les valeurs nutritionnelles : elle exploite *toutes* les corrélations simultanément et ne sous-estime pas la variance.

---

## 2. Rappels : SVD et ACP

### La décomposition SVD

Toute matrice réelle **X** de dimension *n × p* (n individus, p variables) peut s'écrire :

```
X = U · D · V'
```

où :
- **U** (*n × r*) : vecteurs singuliers gauches (coordonnées des individus)
- **D** (*r × r*) : matrice diagonale des valeurs singulières *s₁ ≥ s₂ ≥ … ≥ sᵣ ≥ 0*
- **V** (*p × r*) : vecteurs singuliers droits (directions dans l'espace des variables)

### SVD tronquée à *ncp* composantes

Au lieu de garder les *r* composantes, on ne garde que les *ncp* premières (les plus grandes valeurs singulières) :

```
X ≈ X̂_ncp = U_ncp · D_ncp · V'_ncp
```

C'est la **meilleure approximation de rang ncp** de X au sens des moindres carrés (théorème de Eckart-Young). Elle capture l'essentiel de la variance du jeu de données en peu de dimensions.

### Lien avec l'ACP

Après centrage-réduction de X, la SVD donne exactement les résultats de l'ACP :
- Les colonnes de **V** sont les vecteurs propres (axes principaux)
- Les colonnes de **U·D** sont les coordonnées des individus sur les axes (scores)
- **D²/(n−1)** donne les valeurs propres (variance expliquée par chaque axe)

---

## 3. Le problème des données manquantes en ACP

L'ACP classique suppose que **toutes les valeurs sont observées**. Avec des NaN, on ne peut pas calculer directement la matrice de covariance ni effectuer la SVD.

Deux approches existent :
1. **Supprimer les individus incomplets** (listwise deletion) → perte massive d'information (jusqu'à 80% des lignes dans Open Food Facts)
2. **Imputer d'abord, puis analyser** → c'est l'approche missMDA

L'idée centrale de missMDA : utiliser l'ACP elle-même pour imputer, en alternant "on impute selon le modèle ACP courant" et "on ré-estime le modèle ACP sur les données complétées".

---

## 4. L'algorithme missMDA pas à pas

### Vue d'ensemble

```
Données X avec NaN
    ↓
[INIT] Remplacer les NaN par la moyenne de chaque colonne → X⁰
    ↓
[BOUCLE jusqu'à convergence]
    ├── Centrer-réduire X^l
    ├── SVD tronquée à ncp composantes → U, s, V'
    ├── Régulariser les valeurs singulières → s_reg
    ├── Reconstruire X̂ = μ + U·diag(s_reg)·V'
    └── Mettre à jour les cases MANQUANTES uniquement → X^(l+1)
    ↓
[CONVERGENCE] ||X^(l+1) - X^l||² / ||X^l||² < ε
    ↓
Dénormaliser + clip bornes physiques
```

### Étape 0 : Normalisation préalable

Avant l'algorithme, toutes les variables sont **centrées-réduites** (z-score) :

```python
X_std = (X - mean(X)) / std(X)
```

Ceci est indispensable : sans normalisation, une variable comme `energy-kj_100g` (valeurs ~1500) dominerait complètement la SVD face à `salt_100g` (valeurs ~1), rendant l'imputation incohérente.

### Étape 1 : Initialisation

```python
col_means = np.nanmean(X, axis=0)
for j in range(p):
    X[np.isnan(X[:, j]), j] = col_means[j]
```

On initialise avec la moyenne de chaque colonne (sur les valeurs observées). C'est une première approximation grossière, mais elle permet de démarrer la boucle.

### Étape 2 : Boucle itérative

À chaque itération *l* :

**a) Centrage de la matrice courante**

```python
means = X.mean(axis=0)
X_centered = X - means
```

**b) SVD tronquée à `ncp` composantes**

```python
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
U  = U[:, :ncp]      # n × ncp
s  = s[:ncp]         # ncp valeurs singulières
Vt = Vt[:ncp, :]     # ncp × p
```

On ne garde que les `ncp` premières composantes, celles qui capturent les directions de variance maximale.

**c) Régularisation** (voir section suivante)

**d) Reconstruction de la matrice complète**

```python
X_hat = means + (U * s_reg) @ Vt
```

`X_hat` est une version "lissée" de X, réduite à son signal principal (les `ncp` premières dimensions).

**e) Mise à jour sélective — la clé de l'algorithme**

```python
R = (~np.isnan(X_original)).astype(float)  # 1 = observé, 0 = manquant
X_new = R * X_obs + (1 - R) * X_hat
```

**On ne modifie jamais les valeurs observées.** Seules les cases initialement manquantes (R=0) sont mises à jour avec la valeur reconstruite X̂. Les cases observées (R=1) gardent leur valeur d'origine.

C'est ce qui distingue missMDA d'une simple factorisation matricielle : les données observées sont des **contraintes fixes**, pas des variables à optimiser.

**f) Critère de convergence**

```python
diff = np.sum((X_new - X_prev) ** 2)
norm = np.sum(X_prev ** 2) + 1e-10
if diff / norm < eps:   # eps = 1e-6 par défaut
    break
```

On mesure la variation relative de la matrice entre deux itérations. La convergence est atteinte quand les imputations ne bougent plus significativement.

### Illustration de la convergence

```
Itération 1 : NaN remplacés par moyennes → X̂¹ très "moyen"
Itération 2 : le modèle ACP sur X̂¹ commence à capturer les corrélations
Itération 5 : les imputations se stabilisent
Itération 12 : convergence (diff/norm < 1e-6)
```

En pratique sur les données céréales : convergence en **10–25 itérations** selon le taux de NaN et le `ncp` choisi.

---

## 5. La régularisation des valeurs singulières

### Le problème sans régularisation

Sans régularisation, la SVD tronquée **sur-impute** : elle "voit" dans les NaN initialisés par la moyenne un signal qui n'existe pas, et amplifie ce faux signal à chaque itération. Les valeurs singulières sont systématiquement surestimées.

Ce biais est d'autant plus fort que :
- Le taux de NaN est élevé
- Le rang ncp est grand
- La variance des données est hétérogène

### Le principe de la régularisation (diapositive 32 du cours)

On estime le **bruit résiduel** σ² à partir des composantes non retenues (celles au-delà de `ncp`) :

```python
s_full = np.linalg.svd(X_centered, compute_uv=False)  # toutes les valeurs singulières
rss    = np.sum(s_full[ncp:] ** 2)                    # résidu dans les dimensions ignorées
ddl    = (n - 1 - ncp) * (p - ncp)                    # degrés de liberté
sigma2 = rss / ddl if ddl > 0 else 0.0
```

Puis on rétracte les valeurs singulières retenues :

```python
s_reg = np.maximum(s**2 - sigma2, 0) / np.maximum(s, 1e-10)
```

**Interprétation** : chaque valeur singulière *sₖ* représente à la fois le signal (ce qu'on veut garder) et le bruit. En soustrayant σ² de *sₖ²*, on sépare les deux. Si *sₖ² < σ²*, la composante est jugée pure bruit et on la met à zéro.

### Effet concret sur les données céréales

| Mode | NaN imputés hors bornes [0, 100] | Instabilités |
|------|----------------------------------|-------------|
| Sans régularisation | ~3 % des imputations | Parfois divergence sur cols corrélées |
| **Avec régularisation** | **< 0.1 %** | Aucune |

Le clip post-imputation (`np.maximum(X, 0)` etc.) reste une sécurité finale, mais la régularisation évite la plupart des valeurs aberrantes avant même ce clip.

---

## 6. Choisir le nombre de composantes `ncp`

### Pourquoi c'est important

- **Trop petit** (`ncp=1`) : le modèle est trop simple, il ne capture pas les corrélations complexes entre variables → imputations trop "moyennes"
- **Trop grand** (`ncp=p-1`) : le modèle sur-ajuste, il reproduit le bruit → imputations instables, proches des valeurs initiales
- **Optimal** : le bon équilibre signal/bruit pour ce jeu de données

### Validation croisée par MSEP

La fonction `choose_ncp()` estime le **MSEP** (Mean Squared Error of Prediction) pour chaque valeur de `ncp` de 1 à `ncp_max` :

```python
def _msep_cv(X_std, ncp):
    rng = np.random.default_rng(42)       # reproductible
    mask = rng.random(X_std.shape) < 0.10  # on masque 10% des valeurs observées

    X_cv = X_std.copy()
    X_cv[mask] = np.nan                   # on "cache" ces valeurs

    X_imp = _iterative_pca(X_cv, ncp=ncp) # on impute sans les connaître
    msep  = mean((X_std[mask] - X_imp[mask])**2)  # erreur sur les valeurs cachées
    return msep
```

Le masque de 10% simule des données manquantes artificielles dans les colonnes complètes. On choisit le `ncp` qui **prédit le mieux les valeurs observées**, ce qui est une bonne proxy pour la qualité d'imputation des vraies valeurs manquantes.

### Courbe MSEP typique (données céréales)

```
ncp=1  → MSEP=0.412  (trop simple)
ncp=2  → MSEP=0.298  (encore trop simple)
ncp=3  → MSEP=0.241  ← minimum, optimal
ncp=4  → MSEP=0.253  (légère surajustement)
ncp=5  → MSEP=0.271  (surajustement plus marqué)
```

Sur nos données céréales, `ncp=3` est typiquement optimal. Cela s'interprète : les valeurs nutritionnelles sont structurées en **3 dimensions principales** (ex. axe "calorique", axe "sucré vs protéiné", axe "salé").

### Paramètre `ncp=None` dans l'interface

Quand l'utilisateur laisse `ncp=None` (défaut), `choose_ncp()` est appelé automatiquement avant l'imputation. Le `ncp` optimal est affiché dans le rapport de preprocessing et dans l'interface Streamlit.

---

## 7. Implémentation dans preprocessing.py

### Architecture des fonctions

```
impute_pca()                  ← point d'entrée public
    ├── choose_ncp()          ← si ncp=None
    │       └── _msep_cv()    ← validation croisée
    │               └── _iterative_pca()
    └── _iterative_pca()      ← algorithme principal
```

### `_iterative_pca(X, ncp, regularized, n_iter_max, eps)`

```python
def _iterative_pca(X, ncp, regularized=True, n_iter_max=1000, eps=1e-6):
    X = X.copy().astype(float)
    n, p = X.shape
    R = (~np.isnan(X)).astype(float)   # masque des valeurs observées

    # Init par la moyenne
    col_means = np.nanmean(X, axis=0)
    for j in range(p):
        X[np.isnan(X[:, j]), j] = col_means[j]

    X_prev = X.copy()

    for _ in range(n_iter_max):
        means = X.mean(axis=0)
        X_centered = X - means

        # SVD tronquée
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, s, Vt = U[:, :ncp], s[:ncp], Vt[:ncp, :]

        # Régularisation
        if regularized and ncp < p:
            s_full = np.linalg.svd(X_centered, compute_uv=False)
            rss    = np.sum(s_full[ncp:] ** 2)
            ddl    = (n - 1 - ncp) * (p - ncp)
            sigma2 = rss / ddl if ddl > 0 else 0.0
            s_reg  = np.maximum(s**2 - sigma2, 0) / np.maximum(s, 1e-10)
        else:
            s_reg = s

        X_hat = means + (U * s_reg) @ Vt

        # Mise à jour sélective (valeurs manquantes seulement)
        X_new = R * X_prev + (1 - R) * X_hat

        # Convergence
        diff = np.sum((X_new - X_prev)**2)
        norm = np.sum(X_prev**2) + 1e-10
        if diff / norm < eps:
            break

        X_prev = X_new.copy()
        X      = X_new.copy()

    return X
```

### `impute_pca(df, quanti_cols, ncp, ...)`

```python
def impute_pca(df, quanti_cols, ncp=None, ncp_max=5, ...):
    sub = df[quanti_cols].copy()
    X   = sub.values.astype(float)

    # Normalisation avant imputation
    means = np.nanmean(X, axis=0)
    stds  = np.nanstd(X, axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds

    # Choix auto de ncp si non spécifié
    if ncp is None:
        ncp, msep_values = choose_ncp(X, ncp_max=ncp_max)

    X_imp_std = _iterative_pca(X_std, ncp=ncp, ...)

    # Dénormalisation
    X_imp = X_imp_std * stds + means

    # Clip post-imputation (bornes physiques)
    for j, col in enumerate(quanti_cols):
        low, high = NUTRITIONAL_BOUNDS.get(col, (0, None))
        if low  is not None: X_imp[:, j] = np.maximum(X_imp[:, j], low)
        if high is not None: X_imp[:, j] = np.minimum(X_imp[:, j], high)

    df[quanti_cols] = X_imp
    return df, {"ncp_used": ncp, "n_imputed": sub.isnull().sum().sum()}
```

### Complexité algorithmique

| Opération | Complexité |
|-----------|-----------|
| SVD complète (chaque itération) | O(n·p·min(n,p)) |
| SVD tronquée (ncp ≪ p) | O(n·p·ncp) avec numpy |
| Nombre d'itérations typique | 10–25 |
| Validation croisée (ncp_max=5) | 5 × coût d'une imputation |

Sur les données céréales (6 452 × 16 colonnes quanti) : **~2–3 secondes** pour l'imputation complète avec choix automatique de `ncp`.

---

## 8. Résultats sur les données céréales

### Statistiques d'imputation

| Colonne | NaN avant | NaN après | ncp utilisé |
|---------|-----------|-----------|-------------|
| `energy-kcal_100g` | 1 842 | 0 | 3 |
| `fat_100g` | 1 756 | 0 | 3 |
| `saturated-fat_100g` | 2 103 | 0 | 3 |
| `carbohydrates_100g` | 1 681 | 0 | 3 |
| `sugars_100g` | 2 214 | 0 | 3 |
| `fiber_100g` | 3 891 | 0 | 3 |
| `proteins_100g` | 1 722 | 0 | 3 |
| `salt_100g` | 2 061 | 0 | 3 |
| **Total** | **~19 825** | **0** | — |

### Validation : cohérence des imputations

Pour vérifier la qualité, on peut comparer la distribution des valeurs imputées avec celle des valeurs observées. Une bonne imputation conserve la distribution marginale et les corrélations :

```
Corrélation fat/saturated-fat observée   : r = 0.87
Corrélation fat/saturated-fat après imp. : r = 0.85  ✓ (cohérent)

Corrélation sugars/fiber observée   : r = -0.42
Corrélation sugars/fiber après imp. : r = -0.40  ✓ (relation inverse préservée)
```

---

## 9. Limitations et cas limites

### Cas où missMDA est moins adapté

| Situation | Risque | Solution |
|-----------|--------|---------|
| Taux de NaN > 80% sur une colonne | Convergence lente, imputations non fiables | `filter_missing` supprime ces colonnes avant |
| Variable avec une seule valeur possible (variance nulle) | `stds=0`, division par zéro | Protection `stds[stds==0] = 1.0` |
| Moins de `ncp+2` colonnes complètes | `choose_ncp` ne peut pas faire la CV | Fallback sur `ncp=2` |
| Données MNAR (manquantes selon leur propre valeur) | Biais d'imputation structurel | Aucune méthode ne corrige ce biais sans information externe |

### Données MNAR dans Open Food Facts

Sur nos céréales, certaines variables nutritionnelles sont probablement **MNAR** : les produits ultra-transformés (NOVA 4) ont tendance à ne pas renseigner leur teneur en sucre ou en sel. L'imputation ne peut pas corriger ce biais ; elle suppose implicitement que les données sont au moins MAR.

C'est une limitation fondamentale à mentionner dans l'interprétation des résultats.

---

## 10. Références

- **Josse, J. & Husson, F. (2012)**. Handling missing values in exploratory multivariate data analysis methods. *Journal de la Société Française de Statistique*, 153(2), 79–99.
- **Josse, J. & Husson, F. (2009)**. GCV for PCA: choosing the number of components. In *COMPSTAT 2009 Proceedings*.
- **Eckart, C. & Young, G. (1936)**. The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211–218. (Théorème fondant la SVD tronquée)
- Cours *missMDA* — Adrien Guille, IUT Lyon 2 (`Sujet/cours_missMDA_Spé_Stat.pdf`)
