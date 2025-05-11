import kagglehub
import os
import pandas as pd
import re

# Loading and inspecting Kaggle dataset
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
subfolder_path = os.path.join(path, "News _dataset")
files = os.listdir(subfolder_path)
print("Files inside News _dataset:", files)

df_fake = pd.read_csv(os.path.join(subfolder_path, "Fake.csv"))
df_true = pd.read_csv(os.path.join(subfolder_path, "True.csv"))

# Adding labels and combining
df_fake["label"] = 0
df_true["label"] = 1
df_combined = pd.concat([df_fake, df_true], ignore_index=True)
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# Explorin and cleaning
df_combined.drop_duplicates(inplace=True)
print("\nMissing values:\n", df_combined.isnull().sum())

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
    
# Applying the cleaning function to the text column
df_combined['text'] = df_combined['text'].apply(clean_text)
