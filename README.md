# Misinformation Detection on Social Media

## Project Overview

This project applies machine learning and natural language processing (NLP) to detect text-based misinformation on social media platforms. The goal is to flag potentially harmful content early—before it goes viral—by training predictive models on real-world datasets. The final model achieves over 90% F1-score using TF-IDF features and logistic regression, with additional models evaluated for robustness.

Datasets include a labeled Kaggle fake/true news set and real-time posts pulled from the X (Twitter) API. All code follows the Cookiecutter Data Science structure.

## Features & Models

- Data cleaning & preprocessing (Twitter + Kaggle)
- TF-IDF vectorization
- Logistic Regression (best performance)
- Random Forest & XGBoost comparisons
- Real-time tweet inference via Twitter API
- Visualizations: confusion matrices, F1 score comparison
- Final Sprint 3 report and presentation

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:

- pandas, numpy
- scikit-learn, xgboost
- matplotlib, seaborn
- nltk
- tweepy (Twitter API)
- jupyter

## Setup & Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/daitanboy/misinformation_detection.git
cd misinformation_detection
```

### 2. Optional: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Workflow

This project uses a modular pipeline:

### Step 1: Load and Clean the Data

```bash
python src/data/prepare_data.py
```

### Step 2: Apply TF-IDF Vectorization

```bash
python src/features/build_features.py
```

### Step 3: Train Models

```bash
python src/models/train_model.py
```

### Step 4: Test Live Tweet Predictions (optional)

Make sure you configure your Twitter API bearer token.

```bash
python src/models/predict_live.py
```

## Project Structure

```
misinformation_detection/
├── docs/              # Additional documentation
├── models/            # Saved model files (.pkl, etc.)
├── notebooks/         # Jupyter notebooks (main = Sprint_3.ipynb)
├── references/        # Research references or data dictionaries
├── reports/           # Final report and presentation slides
│   └── figures/       # Confusion matrix, model comparisons, etc.
├── src/               # Source code modules
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Access

This project uses two main datasets. Due to GitHub file size restrictions, the raw data files are not included in the repository. Follow the steps below to download and organize them locally:

### 1. Kaggle Fake News Dataset
- Source: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- Files required:
  - True.csv
  - Fake.csv
- After downloading, place both files in:
  ```
  data/raw/
  ```

### 2. Twitter (X) API - Live Tweet Data
- Tweets are fetched dynamically using the Twitter API and the `predict_live.py` script.
- You’ll need a valid Bearer Token from Twitter’s Developer Platform.
- Insert your token into the `predict_live.py` script where indicated.

## Results

- Best Model: Logistic Regression
- F1 Score: ~92%
- Use Case: Supports NGOs or platforms in flagging misleading posts

Visuals and evaluation results are saved in `reports/figures/`.

## Author

Carlos Lopez Vento  
Information Science @ University of Maryland  
https://github.com/daitanboy
