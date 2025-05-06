"""
prepare_data.py
Handles loading and cleaning Kaggle and Twitter datasets.
"""

import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_kaggle_data(fake_path, true_path):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 0
    df_true['label'] = 1
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)
    return df_combined
