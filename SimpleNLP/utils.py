import regex
import csv


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


def load_data(file_path, delimiter=',', has_header=True, classification=True):
    """
    Loads data for text classification or text generation.

    **Text Classification (CSV format)**
    Expected format:
    ```
    text,label
    "I love this movie!",positive
    "This was terrible!",negative
    ```

    **Text Generation (Plain Text)**
    Expected format:
    ```
    This is an example text for training a language model.
    Another sentence continues here.
    ```

    Args:
    - file_path (str): Path to the data file.
    - delimiter (str): CSV delimiter (default: ',').
    - has_header (bool): Whether the file has a header row (classification only).
    - classification (bool): If `True`, loads labeled data; else, loads text for generation.

    Returns:
    - If classification:
      - texts (list): List of preprocessed text samples.
      - labels (list): List of corresponding labels.
    - If text generation:
      - texts (list): A single string containing the full corpus.
      - None (since no labels exist for generation tasks).
    """
    texts, labels = [], []

    if classification:
        # Load labeled classification data from CSV
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                next(reader)  # Skip header row

            for row in reader:
                if len(row) < 2:
                    continue  # Skip malformed rows
                text, label = row[0], row[1]
                texts.append(preprocess_text(text))
                labels.append(label)

        return texts, labels

    else:
        # Load raw text for text generation
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.read().replace("\n", " ")  # Join all lines into one continuous text

        return texts, None
