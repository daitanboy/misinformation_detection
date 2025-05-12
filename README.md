# Misinformation Detection on Social Media

## Project Overview
This project uses machine learning and natural language processing (NLP) to detect misinformation on social media platforms, such as Twitter, by classifying news articles as **fake** or **real**. The project leverages Kaggle's dataset and Twitter API for training data.

## Directory Structure
- **`src/`**: Contains the main scripts (`prepare_data.py`, `build_features.py`, `train_model.py`, `predict_live.py`) that process and analyze data.
- **`reports/`**: Stores the final report, presentation slides, and figures.
- **`docs/`**: Contains documentation files, project goals, and relevant references.
- **`references/`**: Stores research papers and data dictionaries.
- **`requirements.txt`**: Lists the necessary Python packages.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/daitanboy/misinformation_detection.git
   cd misinformation_detection
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   - Preprocess the data using `prepare_data.py`:
     ```bash
     python src/data/prepare_data.py
     ```
   - Build features and split data:
     ```bash
     python src/features/build_features.py
     ```
   - Train models:
     ```bash
     python src/models/train_model.py
     ```

## Models
- **Logistic Regression**: Achieved a high F1 score of 0.99.
- **Random Forest**: Achieved the best F1 score of 0.99.
- **XGBoost**: Also performs well with an F1 score of 0.99.

## Visualizations
Included are confusion matrix visualizations and F1 score comparisons.

## License
This project is licensed under the MIT License.
rts NGOs and platforms in flagging misleading posts.
- Visuals and evaluation results are saved in `reports/figures/`.

## Author

Carlos Lopez Vento  
Information Science @ University of Maryland  
[GitHub Profile](https://github.com/daitanboy)
