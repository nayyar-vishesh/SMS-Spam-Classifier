import pandas as pd
import re
import os

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spam.csv")

def clean_text(text: str) -> str:
    """Preprocess text by lowercasing and removing non-alphanumeric characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def load_data(path: str = None) -> pd.DataFrame:
    """Load and clean the SMS dataset."""
    path = path or DEFAULT_DATA_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Load dataset
    df = pd.read_csv(path)
    
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns")

    # Clean data
    df = df.dropna(subset=['label', 'text'])
    df['text'] = df['text'].apply(clean_text)
    
    # Filter valid labels just in case
    df = df[df['label'].isin(['ham', 'spam'])]
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} messages.")
    print(df.head())
