import regex
import csv
import pandas as pd

def preprocess_text(text):
    """
    Cleans and normalizes text by:
    - Converting to lowercase
    - Removing special characters
    - Removing extra spaces
    """
    text = text.lower()  # Convert to lowercase
    text = regex.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = regex.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def load_twitter_data(file_path, sample_size=5000, random_state=42, delimiter=','):
    """
    Loads Twitter sentiment analysis dataset.

    Args:
    - file_path (str): Path to the CSV file.
    - sample_size (str): Number of samples to use.
    - delimiter (str): CSV delimiter (default: ',')

    Returns:
    - texts (list): List of preprocessed tweets
    - entities (list): List of entities
    - labels (list): List of sentiment labels
    """
    df = pd.read_csv(file_path, usecols=['Tweet_content', 'Entity', 'Sentiment'])

    # Sample data if dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    # Preprocess texts
    processed_texts = []
    for text, entity in zip(df['Tweet_content'], df['Entity']):
        combined_text = f"{text} [Entity] {entity}"
        processed_texts.append(preprocess_text(combined_text))

    return processed_texts, df['Entity'].tolist(), df['Sentiment'].tolist()