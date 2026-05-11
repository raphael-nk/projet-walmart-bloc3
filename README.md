# Prédiction des ventes hebdomadaires Walmart (Weekly_Sales)

Projet de machine learning supervisé dans le cadre de certification CDSD, Jedha (**bloc 3**). L’objectif est de prédire les **ventes hebdomadaires** (`Weekly_Sales`) de magasins Walmart à partir d’indicateurs économiques et de variables de contexte.

## Description du projet

- **Cible** : `Weekly_Sales` (ventes de la semaine).
- **Prédicteurs principaux** : 
   - indicateurs économiques (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`), 
   - indicateur de semaine de fête (`Holiday_Flag`), identifiant de magasin (`Store`), 
   - ainsi que des **features temporelles** dérivées de la date (`Year`, `Month`, `Day`, `DayOfWeek`).
- **Données brutes** : fichier `data/Walmart_Store_sales.csv` - environ **150 lignes** et **8 colonnes** (`Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`), avec une part importante de **valeurs manquantes** (environ **50 %** des lignes incomplètes selon l’analyse EDA).
- **Pipeline** : exploration (notebook 1), nettoyage et feature engineering (notebook 2), modélisation, validation croisée et sauvegarde des artefacts (notebook 3).

## Structure des fichiers

```text
projet-walmart-bloc3/
├── data/
│   └── Walmart_Store_sales.csv      # Jeu de données source
├── notebook/
│   ├── 01_data_exploration.ipynb    # EDA (Plotly, tests, corrélations, temporalité)
│   ├── 02_preprocessing.ipynb       # Prétraitement + dataset prêt pour le ML
│   └── 03_training_models.ipynb     # Modèles, GridSearchCV, comparaison, export .pkl
├── output/
│   ├── data/
│   │   ├── df_preprocessed.csv      # Données après prétraitement (entrée du notebook 3)
│   │   └── metrics_comparison.csv   # Métriques comparatives train/test des modèles
│   ├── images/                      # Figures Plotly exportées (PNG)
│   └── models/
│       ├── column_transformer.pkl   # ColumnTransformer (OneHotEncoder + numériques)
│       ├── best_model.pkl           # Meilleur modèle au sens du r2 test
│       ├── model_lr.pkl             # Régression linéaire
│       ├── model_ridge.pkl          # Ridge (meilleur α après recherche)
│       └── model_lasso.pkl          # Lasso (meilleur α après recherche)
├── pyproject.toml                   # Dépendances (projet géré avec uv)
├── uv.lock
├── main.py
├── requirements.txt         # Fichier listant les dépendances pour pip
└── README.md

```

## Installation des dépendances

**Prérequis** : Python **≥ 3.12** (voir `.python-version`).

### Option A - avec [uv](https://github.com/astral-sh/uv)

À la racine du projet :

```bash
uv sync
```

### Option B - avec pip et requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

> Un fichier `requirements.txt` listant toutes les dépendances nécessaires est fourni à la racine du projet.

Les bibliothèques centrales pour reproduire les notebooks sont notamment :

| Package        | Rôle principal                          |
|----------------|-----------------------------------------|
| `pandas`       | Manipulation des tableaux             |
| `numpy`        | Calculs numériques                    |
| `plotly`       | Visualisations interactives / export  |
| `kaleido`      | Export des figures Plotly en PNG      |
| `scikit-learn` | Prétraitement, modèles, métriques     |
| `scipy`        | Tests statistiques (notebook 1)       |
| `joblib`       | Chargement / sauvegarde des modèles   |

*(Les versions exactes sont figées dans `pyproject.toml` / `uv.lock`.)*

## Exécution des notebooks (ordre obligatoire)

Les chemins relatifs (`../output/...`, `../data/...`) supposent que le **répertoire de travail** est le dossier `notebook/` (comportement par défaut sous Jupyter / VS Code lorsque le notebook est ouvert depuis ce dossier).

1. **`notebook/01_data_exploration.ipynb`**  
   Analyse exploratoire : visualisations Plotly (environ **8 figures**), tests statistiques, analyse temporelle, matrices de corrélation, etc.

2. **`notebook/02_preprocessing.ipynb`**  
   Prétraitement : gestion des NaN (suppression / imputation par la médiane selon le pipeline défini), **feature engineering** sur la date (`Year`, `Month`, `Day`, `DayOfWeek`), filtrage des **outliers** (règle type **3σ** sur les variables continues), harmonisation des types — puis écriture de `output/data/df_preprocessed.csv`.

3. **`notebook/03_training_models.ipynb`**  
   Chargement de `df_preprocessed.csv`, **`ColumnTransformer`** avec **`OneHotEncoder`** sur `Store` (les autres colonnes explicatives passent en numérique), **split train/test 80/20** (`random_state=42`), entraînement de :
   - une **régression linéaire** (baseline),
   - **Ridge** et **Lasso** avec **`GridSearchCV`** (validation croisée **5 folds**),
   comparaison des trois modèles, sauvegarde du **meilleur** modèle, des trois estimateurs, du transformer et du fichier **`metrics_comparison.csv`**.

## Résultats clés

- **Meilleur modèle** : **régression linéaire** (`LinearRegression`), avec le meilleur **r2 sur le jeu de test** parmi les trois candidats.
- **Régularisation** : **Ridge** (alpha optimal proche de **0,1**) et **Lasso** (alpha optimal proche de **1000**) légèrement derrière sur le r2 test et le RMSE test.
- **Sur-apprentissage** : écart train/test sur le r2 modéré (**gap r2** inférieur à environ **5 %** pour les trois modèles dans les métriques exportées), ce qui suggère une généralisation acceptable compte tenu de la taille du jeu.
- **Interprétation** : le magasin (**`Store`**, via l’encodage one-hot) est le facteur le plus discriminant ; le **Lasso** tend à **mettre à zéro** plusieurs coefficients — typiquement des variables comme **`Holiday_Flag`**, **`DayOfWeek`**, et une modalité de magasin (ex. **`Store_19`**) selon l’exécution du notebook 3.

### Tableau comparatif des modèles (jeu de test)

Valeurs indicatives issues du dernier export `output/data/metrics_comparison.csv` (arrondis) :

| Modèle              | Hyperparamètre α | RMSE (test) | r2 (test) | gap r2 (train − test) |
|---------------------|------------------|------------:|----------:|----------------------:|
| LinearRegression    | —                | ≈ 151 200   | ≈ 0,933   | ≈ 0,043               |
| Ridge               | 0,1              | ≈ 163 500   | ≈ 0,921   | ≈ 0,053               |
| Lasso               | 1 000            | ≈ 166 500   | ≈ 0,918   | ≈ 0,056               |

*(Les valeurs exactes peuvent varier légèrement si vous ré-exécutez tout le pipeline ; le fichier CSV reste la référence chiffrée.)*

### Livrables générés

- Modèles : `best_model.pkl`, `model_lr.pkl`, `model_ridge.pkl`, `model_lasso.pkl`
- Prétraitement aligné sur l’entraînement : `column_transformer.pkl`
- Synthèse des métriques : `output/data/metrics_comparison.csv`

## Utiliser le modèle sauvegardé pour prédire sur de nouvelles données

Les prédictions doivent repasser par le **même `ColumnTransformer`** que celui entraîné dans le notebook 3 (mêmes colonnes, mêmes types, pas de colonne `Weekly_Sales` dans les entrées).

### Colonnes attendues pour une ligne « brute » avant transformation

Alignées sur l’exemple du notebook 3 (sans `Date` textuelle : les features calendaires sont déjà numériques) :

`Store`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Year`, `Month`, `Day`, `DayOfWeek`

### Exemple minimal (Python)

```python
import joblib
import pandas as pd

output_models_path = "output/models"  # adapter si vous lancez depuis notebook/

ct = joblib.load(f"{output_models_path}/column_transformer.pkl")
model = joblib.load(f"{output_models_path}/best_model.pkl")

new_data = pd.DataFrame({
    "Store": [4],
    "Holiday_Flag": [0],
    "Temperature": [55.0],
    "Fuel_Price": [3.45],
    "CPI": [220.5],
    "Unemployment": [7.5],
    "Year": [2012],
    "Month": [6],
    "Day": [15],
    "DayOfWeek": [4],
})

X_new = ct.transform(new_data)
prediction = model.predict(X_new)
print(f"Ventes prédites : ${prediction[0]:,.2f}")
```

**Points d’attention** :

- Le **`OneHotEncoder`** a été ajusté sur les magasins présents dans les données d’entraînement : une valeur de **`Store`** jamais vue peut provoquer une erreur (`handle_unknown='error'` dans le pipeline d’origine).
- Les nouvelles lignes doivent être **cohérentes** avec le prétraitement du notebook 2 (pas de NaN non gérés si le transformer ne les impute pas — ici le transformer attend des colonnes déjà au format du CSV prétraité).

---

Contexte : certification Jedha - Bloc 3 Machine Learning Supervisé
Auteurs : **RANJAKASOA Raphaël Marcellin**
