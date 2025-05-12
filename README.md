# Misinformation Detection on Social Media

## Project Overview
This project applies machine learning and natural language processing (NLP) to detect misinformation on social media platforms. The goal is to build a pipeline that classifies text-based social media content as **fake** or **real** news. The project uses Kaggle's dataset and real-time tweets from Twitter to train and evaluate various machine learning models, such as Logistic Regression, Random Forest, and XGBoost.

## Project Structure

### Directory Structure

```
misinformation_detection/
├── docs/              # Documentation related to the project
├── models/            # Scripts and models used for training
├── references/        # Research papers and data references
├── reports/           # Final reports and presentation slides
├── requirements.txt   # Lists project dependencies
├── src/               # Source code (includes data processing, feature engineering, model training)
│   ├── data/          # Data-related scripts (prepare_data.py)
│   ├── features/      # Feature extraction scripts (build_features.py)
│   ├── models/        # Scripts for model training and predictions (train_model.py, predict_live.py)
├── LICENSE            # Licensing details for the project
├── .gitignore         # Git ignore file
├── README.md          # Project documentation
```

### Key Files:
- **`prepare_data.py`**: Script for loading and cleaning data.
- **`build_features.py`**: Script for transforming data (e.g., TF-IDF vectorization).
- **`train_model.py`**: Script for training models (Logistic Regression, Random Forest, XGBoost).
- **`predict_live.py`**: Script for making real-time predictions from live tweets pulled from the Twitter API.

## Setup Instructions

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/daitanboy/misinformation_detection.git
cd misinformation_detection
```

### 2. Install Dependencies
Install the required Python libraries listed in the `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Project

#### Step 1: Data Preparation
Preprocess the data and clean the raw datasets using the script `prepare_data.py`:
```bash
python src/data/prepare_data.py
```

#### Step 2: Feature Engineering
Use `build_features.py` to vectorize the cleaned data and split it into training and testing datasets:
```bash
python src/features/build_features.py
```

#### Step 3: Train the Models
Train the models (Logistic Regression, Random Forest, and XGBoost) using `train_model.py`:
```bash
python src/models/train_model.py
```

#### Step 4: Make Predictions on Live Tweets (Optional)
Run `predict_live.py` to get predictions on live tweets from Twitter:
```bash
python src/models/predict_live.py
```

## How to Replicate the Pipeline

1. **Prepare Data**:  
   - The `prepare_data.py` script downloads the raw dataset from Kaggle using the Kaggle API and cleans the text (removes URLs, mentions, and special characters).
   - It then saves the cleaned dataset as `cleaned_data.csv`.

2. **Feature Extraction**:  
   - `build_features.py` uses **TF-IDF vectorization** to convert text data into numerical features.
   - The data is split into training and testing sets using **train_test_split** from Scikit-learn.

3. **Model Training**:  
   - `train_model.py` trains **Logistic Regression**, **Random Forest**, and **XGBoost** models.
   - The models are evaluated using **cross-validation** with F1 scoring.
   - The best-performing model (Random Forest) is saved using **joblib** for later use.

4. **Live Predictions**:  
   - `predict_live.py` pulls live tweets from Twitter containing the phrase "fake news" and makes predictions using the trained Random Forest model.

## Results
- **Logistic Regression**: Achieved an F1 score of 0.994.
- **Random Forest**: Achieved the best F1 score of 0.997.
- **XGBoost**: Also performed well with an F1 score of 0.995.

## Visualizations
The project includes the following visualizations:
- **Confusion Matrices** for each model (Logistic Regression, Random Forest, XGBoost).
- **F1 Score Comparison**: A bar plot comparing the performance of the three models.

## License
This project is licensed under the MIT License.

## Author
Carlos Lopez Vento  
Information Science @ University of Maryland  
[GitHub Profile](https://github.com/daitanboy)
