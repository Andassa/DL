<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Heart%20Disease%20Classification-ff6b6b?style=for-the-badge&logo=heart&logoColor=white" alt="Deep Learning Heart Disease" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/NumPy-Pandas-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

<h1 align="center">🫀 Heart Disease Prediction — Binary Classification</h1>
<p align="center">
  <strong>Neural network–based binary classification on the UCI Heart Disease dataset</strong>
</p>
<p align="center">
  <em>Train, validate, and evaluate a fully connected model with Keras & TensorFlow</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-about">About</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-model">Model</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

<br />

---

## 📑 Table of contents

- [Quick Start](#-quick-start)
- [About the project](#-about)
- [Dataset](#-dataset)
- [Model architecture](#-model)
- [Repository structure](#-repository-structure)
- [Getting started](#-getting-started)
- [Usage](#-usage)
- [Results & visualizations](#-results--visualizations)
- [Tech stack](#-tech-stack)
- [License](#-license)

---

## ⚡ Quick Start

```bash
git clone https://github.com/Andassa/DL.git && cd DL
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
jupyter notebook
```

Then open **`text_binary_classification_example.ipynb`** or **`text_binary_classification_example_val.ipynb`** and run all cells.

---

## 🎯 About

This repository implements **binary classification** for **heart disease prediction** using a small **fully connected neural network**. The goal is to predict whether a patient has heart disease (**target = 1**) or not (**target = 0**) from **13 clinical features** (age, sex, blood pressure, cholesterol, ECG, etc.).

| Notebook | Data | Purpose |
|----------|------|--------|
| **`text_binary_classification_example.ipynb`** | `heart.csv` (303 rows) | Train with internal validation (78% / 22% split) |
| **`text_binary_classification_example_val.ipynb`** | `heart_training_val.csv` + `heart_testing.csv` | Train/validate, then **evaluate on a held-out test set** |

> **Summary:** Two Jupyter workflows — one for quick experimentation on the full dataset, one for a proper train/validation/test pipeline with final metrics on unseen data.

---

## 📊 Dataset

Data comes from the **UCI Machine Learning Repository** (Heart Disease Dataset). All CSVs are in the **`dataset/`** folder.

### Files

| File | Description |
|------|-------------|
| **`heart.csv`** | Full dataset — 303 patients, 14 columns |
| **`heart_training_val.csv`** | Train + validation subset — 271 rows |
| **`heart_testing.csv`** | Held-out test set for final evaluation |

### Features (14 columns)

| Variable | Description |
|----------|-------------|
| `age` | Age in years |
| `sex` | Sex (0 = female, 1 = male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (0/1) |
| `restecg` | Resting electrocardiographic results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (0/1) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored (0–4) |
| `thal` | Thalassemia test result (0–3) |
| **`target`** | **Heart disease present (0 = no, 1 = yes)** |

---

## 🧠 Model

**Keras Sequential** binary classifier:

```
┌─────────────────────────┐
│  Input  (13 features)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Dense(11, ReLU)        │  ← hidden layer
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Dense(1, Sigmoid)      │  ← output (probability)
└───────────┬─────────────┘
            │
            ▼
     Prediction (0 or 1)
```

| Setting | Value |
|---------|--------|
| **Loss** | `binary_crossentropy` |
| **Optimizer** | `adam` |
| **Metric** | `accuracy` |
| **Epochs** | 300 |
| **Validation split** | 0.22 (when using full `heart.csv`) |

Final class prediction: **`(model.predict(X) > 0.5).astype("int32")`**

---

## 📁 Repository structure

```
DL/
├── 📂 dataset/
│   ├── heart.csv
│   ├── heart_training_val.csv
│   └── heart_testing.csv
├── 📓 text_binary_classification_example.ipynb
├── 📓 text_binary_classification_example_val.ipynb
├── 📄 requirements.txt
├── 📄 README.md
└── .ipynb_checkpoints/
```

---

## 🚀 Getting started

### Prerequisites

- **Python 3.8+**
- Optional but recommended: **virtual environment** (`venv` or `conda`)

### Install

1. **Clone and enter the repo**

   ```bash
   git clone https://github.com/Andassa/DL.git
   cd DL
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

   Or **JupyterLab:**

   ```bash
   pip install jupyterlab
   jupyter lab
   ```

---

## 📖 Usage

1. Open one of the two notebooks in Jupyter.
2. Ensure CSV paths are correct (`dataset/heart.csv` or `dataset/heart_training_val.csv` and `dataset/heart_testing.csv`).
3. Run cells in order (**Run All** or step through).
4. In **`text_binary_classification_example_val.ipynb`**, the model is evaluated on the test set via **`model.evaluate(X_test, y_test)`**.

**What each notebook does:**

- Load and explore data (`head`, `shape`, `describe`, `info`, `isna`)
- Basic visualizations (distributions, etc.)
- Define and compile the Keras model
- Train with loss/accuracy curves (train & validation)
- Predictions and, in the “val” notebook, **test set evaluation**

---

## 📈 Results & visualizations

The notebooks produce:

- **Descriptive statistics** and missing-value checks  
- **Distribution plots** for the features  
- **Training curves**: loss and accuracy vs. epochs (train & validation)  
- **Test set evaluation** (val notebook): final loss and accuracy  

Performance depends on the split and hyperparameters (epochs, hidden size, etc.). You can tune these directly in the notebook cells.

---

## 🛠 Tech stack

| Tool | Role |
|------|------|
| **Python** | Main language |
| **NumPy** | Numerical computation |
| **Pandas** | Data loading & manipulation (CSV, DataFrames) |
| **TensorFlow / Keras** | Neural network definition and training |
| **Matplotlib** | Plots (curves, distributions) |
| **Jupyter** | Interactive notebooks |

---

## 📜 License

This project is available under the **MITY License**. Feel free to use it for learning or as a base for your own experiments.

---

<p align="center">
  <b>If this project helped you, consider giving it a ⭐ on GitHub.</b>
</p>
<p align="center">
  <sub>Heart Disease Binary Classification · Deep Learning · Keras · TensorFlow</sub>
</p>
