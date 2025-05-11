# Misinformation Detection on Social Media

## Project Overview

This project leverages machine learning (ML) and natural language processing (NLP) to detect and classify misinformation on social media platforms like Twitter/X. The goal is to flag potentially harmful content early—before it spreads—using predictive models trained on real-world datasets. The final model achieves over 90% F1-score with Logistic Regression, and other models like Random Forest and XGBoost are also evaluated for robustness.

Datasets used include:
- A labeled Kaggle Fake/True News dataset
- Real-time posts fetched from the Twitter API

All code follows the **Cookiecutter Data Science** structure, ensuring modularity and clarity in the pipeline.

## Features & Models

- **Data Cleaning & Preprocessing** (Twitter + Kaggle)
- **TF-IDF Vectorization** for feature extraction
- **Logistic Regression** (best performing model)
- Comparisons with **Random Forest** and **XGBoost**
- Real-time predictions using the **Twitter API**
- Visualizations: **Confusion Matrices** and **F1 Score Comparison**
- Final Sprint 3 Report and Presentation

## Requirements

To run the project, install the dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:
- **pandas**, **numpy**
- **scikit-learn**, **xgboost**
- **matplotlib**, **seaborn**
- **nltk**
- **tweepy** (Twitter API)
- **jupyter**

## Setup & Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/daitanboy/misinformation_detection.git
cd misinformation_detection
```

### 2. Optional: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scriptsctivate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Workflow

This project follows a modular pipeline:

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

### Step 4: Test Live Tweet Predictions (Optional)
Make sure to configure your Twitter API **Bearer Token**.
```bash
python src/models/predict_live.py
```

## Project Structure

```
misinformation_detection/
├── docs/              # Additional documentation
├── models/            # Saved model files (.pkl)
├── notebooks/         # Jupyter notebooks (main = Sprint_3.ipynb)
├── references/        # Research papers or data dictionaries
├── reports/           # Final report and presentation slides
│   └── figures/       # Confusion matrix, model comparisons
├── src/               # Source code modules
│   ├── data/          # Data loading and cleaning
│   ├── features/      # Feature engineering (TF-IDF, etc.)
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Graphs and visual outputs
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Access

The datasets are not included in the repository due to size limitations. You will need to download and organize them locally.

### 1. Kaggle Fake News Dataset
- Source: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- Files:
  - `True.csv`
  - `Fake.csv`
- Place these files in:
  ```
  data/raw/
  ```

### 2. Twitter (X) API - Live Tweet Data
- Tweets are pulled dynamically using the **Twitter API** and the script `predict_live.py`.
- To fetch live data, you’ll need a valid **Bearer Token** from Twitter’s Developer Platform.
- Insert the token into the script as indicated in `predict_live.py`.

## Results

- **Best Model**: Logistic Regression
- **F1 Score**: ~92%
- **Use Case**: Supports NGOs and platforms in flagging misleading posts.
- Visuals and evaluation results are saved in `reports/figures/`.

## Author

Carlos Lopez Vento  
Information Science @ University of Maryland  
[GitHub Profile](https://github.com/daitanboy)
