# Misinformation Detection on Social Media

## 📌 Project Description

This project uses predictive analytics and natural language processing (NLP) to detect text-based misinformation on social media platforms. By analyzing language patterns, engagement metrics, and post timing, the models aim to flag fake news content before it spreads. The project focuses on building and evaluating machine learning models such as Logistic Regression, Random Forest, and BERT using datasets from Kaggle and Twitter/X API.

---

## ⚙️ Dependencies

All dependencies are listed in `requirements.txt` or `environment.yml`, but key packages include:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn
- nltk
- transformers
- jupyter
- requests
- kagglehub

To install everything:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/daitanboy/misinformation_detection.git
cd misinformation_detection
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔄 Running the Data Pipeline

The project is structured using the Cookiecutter Data Science template. Follow these steps:

### 1. Load and clean the datasets:

```bash
python src/data/make_dataset.py
```

### 2. Feature engineering:

```bash
python src/features/build_features.py
```

---

## 🤖 Model Training and Evaluation

### 1. Train the model:

```bash
python src/models/train_model.py
```

### 2. Evaluate performance:

```bash
python src/models/predict_model.py
```

This script prints classification metrics and saves charts to `reports/figures/`.

---

## 🔁 Reproducing Results

To reproduce the full pipeline from scratch:

1. Download the raw datasets to `data/raw/`
2. Run:
```bash
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/predict_model.py
```
3. View outputs in `models/` and `reports/figures/`

---

## 📁 Project Structure

```
.
├── data/              # Raw, interim, and processed data
├── notebooks/         # Jupyter notebooks for EDA and modeling
├── src/               # Modular source code
├── models/            # Saved model files
├── reports/           # Visualizations and final reports
├── requirements.txt   # Python dependencies
├── README.md          # Project overview (this file)
```

---

## ✍️ Author

**Carlos Lopez Vento**  
Information Science @ University of Maryland  
[GitHub Profile](https://github.com/daitanboy)
