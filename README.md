<div align="center">

# 🫀 Prédiction de maladie cardiaque — Deep Learning

**Classification binaire sur le jeu de données Heart Disease (UCI)**  
*Réseau de neurones avec Keras / TensorFlow*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

</div>

## 📋 Table des matières

- [À propos du projet](#-à-propos-du-projet)
- [Jeu de données](#-jeu-de-données)
- [Architecture du modèle](#-architecture-du-modèle)
- [Structure du dépôt](#-structure-du-dépôt)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats & visualisations](#-résultats--visualisations)
- [Technologies utilisées](#-technologies-utilisées)
- [Auteur & licence](#-auteur--licence)

---

## 🎯 À propos du projet

Ce dépôt contient un **projet de classification binaire** en deep learning appliqué à la **prédiction de maladie cardiaque**. L’objectif est de prédire si un patient est atteint ou non d’une affection cardiaque (variable cible binaire) à partir de **13 caractéristiques cliniques** (âge, sexe, pression artérielle, cholestérol, etc.).

Le projet est organisé autour de **deux notebooks Jupyter** :

| Notebook | Données | Objectif |
|----------|---------|----------|
| `text_binary_classification_example.ipynb` | `heart.csv` (303 lignes) | Entraînement avec validation interne (split 78 % / 22 %) |
| `text_binary_classification_example_val.ipynb` | `heart_training_val.csv` + `heart_testing.csv` | Entraînement sur train/val puis **évaluation sur un jeu de test séparé** |

Le modèle utilisé est un **réseau de neurones fully connected** (Keras Sequential) avec une couche cachée et une sortie sigmoïde pour la classification binaire.

---

## 📊 Jeu de données

Les données proviennent du **UCI Machine Learning Repository** (Heart Disease Dataset). Elles sont fournies au format CSV dans le dossier `dataset/`.

### Fichiers

| Fichier | Description |
|---------|-------------|
| `heart.csv` | Jeu complet (303 patients, 14 colonnes) |
| `heart_training_val.csv` | Sous-ensemble train + validation (271 lignes) |
| `heart_testing.csv` | Jeu de test réservé pour l’évaluation finale |

### Variables (14 colonnes)

| Variable | Description |
|----------|-------------|
| `age` | Âge (années) |
| `sex` | Sexe (0 = femme, 1 = homme) |
| `cp` | Type de douleur thoracique (0–3) |
| `trestbps` | Pression artérielle au repos (mmHg) |
| `chol` | Cholestérol sérique (mg/dl) |
| `fbs` | Glycémie à jeun > 120 mg/dl (0/1) |
| `restecg` | Résultat électrocardiographique au repos (0–2) |
| `thalach` | Fréquence cardiaque max atteinte |
| `exang` | Angine induite par l’exercice (0/1) |
| `oldpeak` | Dépression du segment ST (exercice) |
| `slope` | Pente du segment ST (0–2) |
| `ca` | Nombre de vaisseaux principaux colorés (0–4) |
| `thal` | Résultat du test thal (0–3) |
| **`target`** | **Présence de maladie cardiaque (0 = non, 1 = oui)** |

Les notebooks chargent les CSV, séparent les features (`X`) et la cible (`y`), et n’utilisent **pas de prétraitement type StandardScaler** dans la version actuelle (les entrées sont utilisées telles quelles).

---

## 🧠 Architecture du modèle

Modèle **Keras Sequential** :

```
Input (13 features)
    ↓
Dense(11, activation='relu', input_dim=13)
    ↓
Dense(1, activation='sigmoid')
    ↓
Output (probabilité 0–1)
```

- **Perte :** `binary_crossentropy`
- **Optimiseur :** `adam`
- **Métrique :** `accuracy`
- **Entraînement :** 300 epochs, `validation_split=0.22` (pour le notebook sur `heart.csv`)

La prédiction finale est obtenue en seuillant la sortie à 0,5 : `(model.predict(X) > 0.5).astype("int32")`.

---

## 📁 Structure du dépôt

```
DL/
├── dataset/
│   ├── heart.csv                 # Données complètes
│   ├── heart_training_val.csv     # Train + validation
│   └── heart_testing.csv         # Jeu de test
├── text_binary_classification_example.ipynb      # Entraînement (heart.csv)
├── text_binary_classification_example_val.ipynb # Entraînement + évaluation sur test
├── requirements.txt
├── README.md
└── .ipynb_checkpoints/            # Checkpoints Jupyter (optionnel)
```

---

## 🚀 Installation

### Prérequis

- **Python** 3.8 ou supérieur  
- Environnement virtuel recommandé (`venv` ou `conda`)

### Étapes

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/VOTRE_UTILISATEUR/DL.git
   cd DL
   ```

2. **Créer et activer un environnement virtuel** (exemple avec `venv`)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Installer les dépendances**

   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer Jupyter** (si vous utilisez les notebooks)

   ```bash
   jupyter notebook
   ```

   Ou avec JupyterLab :

   ```bash
   pip install jupyterlab
   jupyter lab
   ```

---

## 📖 Utilisation

1. Ouvrir l’un des deux notebooks dans Jupyter.
2. Vérifier que le chemin des CSV est correct (par défaut : `dataset/heart.csv` ou `dataset/heart_training_val.csv` et `dataset/heart_testing.csv`).
3. Exécuter les cellules dans l’ordre (Run All ou cellule par cellule).
4. Dans le notebook **avec validation** (`text_binary_classification_example_val.ipynb`), le modèle est en plus évalué sur `heart_testing.csv` via `model.evaluate(X_test, y_test)`.

Les notebooks incluent notamment :

- Chargement et exploration des données (`head`, `shape`, `describe`, `info`, `isna`)
- Visualisations (répartition des variables, etc.)
- Définition du modèle Keras
- Compilation et entraînement
- Courbes de loss et d’accuracy (train / validation)
- Prédictions et, pour le notebook « val », évaluation sur le jeu de test

---

## 📈 Résultats & visualisations

Les notebooks génèrent :

- **Statistiques descriptives** et vérification des valeurs manquantes
- **Graphiques** de distribution des variables
- **Courbes d’apprentissage** : loss et accuracy en fonction du nombre d’epochs (train et validation)
- **Évaluation sur le jeu de test** (notebook `text_binary_classification_example_val.ipynb`) : loss et accuracy finales

Les performances dépendent du split et des hyperparamètres (epochs, taille de la couche cachée, etc.). Vous pouvez faire varier ces paramètres directement dans les cellules concernées.

---

## 🛠 Technologies utilisées

- **Python** — Langage principal  
- **NumPy** — Calcul numérique  
- **Pandas** — Manipulation des données (CSV, DataFrames)  
- **TensorFlow / Keras** — Réseau de neurones et entraînement  
- **Matplotlib** — Visualisations (courbes, distributions)  
- **Jupyter** — Notebooks interactifs  

---

## 👤 Auteur & licence

Projet réalisé dans le cadre d’un apprentissage du **deep learning** et de la **classification binaire** sur données de santé.

- **Licence :** MIT (ou indiquez la vôtre si différente).  
- Pour toute question ou suggestion, n’hésitez pas à ouvrir une *issue* ou une *pull request*.

---

<div align="center">

*Si ce projet vous a été utile, n’hésitez pas à lui donner une ⭐ sur GitHub.*

</div>
