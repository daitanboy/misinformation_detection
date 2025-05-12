import kagglehub
import os
import pandas as pd
import re

# Load the dataset from Kaggle using kagglehub
path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
subfolder_path = os.path.join(path, "News _dataset")

# Check if the dataset files exist in the directory
files = os.listdir(subfolder_path)
print("Files in the dataset:", files)

# Load the CSV files from the dataset
df_fake = pd.read_csv(os.path.join(subfolder_path, "Fake.csv"))
df_true = pd.read_csv(os.path.join(subfolder_path, "True.csv"))

# Add labels for 'Fake' and 'True' news
df_fake['label'] = 0
df_true['label'] = 1

# Combine the two datasets into one
df_combined = pd.concat([df_fake, df_true], ignore_index=True)

# Shuffle the dataset to mix fake and real news
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# Drop any duplicates
df_combined.drop_duplicates(inplace=True)

# Clean the text (lowercase and remove unwanted characters)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Apply cleaning to the text column
df_combined['text'] = df_combined['text'].apply(clean_text)

# Save the cleaned data
df_combined.to_csv('cleaned_data.csv', index=False)
