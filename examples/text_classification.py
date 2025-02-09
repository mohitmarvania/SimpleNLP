# Implementing basic text classification
import numpy as np
from SimpleNLP.SimpleNLP.tokenization import Tokenizer, BPETokenizer, NGramTokenizer
from SimpleNLP.SimpleNLP.embeddings import Word2Vec, TFIDFVectorizer
from SimpleNLP.SimpleNLP.classifiers import LogisticRegressionClassifier, FeedForwardNN
from SimpleNLP.SimpleNLP.utils import load_twitter_data, preprocess_text
from sklearn.metrics import accuracy_score, classification_report

def train_text_classification():
    # File path for training and testing data
    training_file = "/Users/mohit/Documents/Neural Network/SimpleNLP/datasets/twitter_training.csv"
    validating_file = "/Users/mohit/Documents/Neural Network/SimpleNLP/datasets/twitter_validation.csv"

    # load training data
    print("Loading training data...")
    train_texts, train_entites, train_labels = load_twitter_data(training_file, sample_size=7000)

    # load testing data
    print("Loading validation data...")
    val_texts, val_entities, val_labels = load_twitter_data(validating_file, sample_size=7000)

    # Initializing tokenizer
    print("Tokenizing texts...")
    tokenizer = Tokenizer(mode="word")
    train_tokens = [tokenizer.tokenize(text) for text in train_texts]
    val_tokens = [tokenizer.tokenize(text) for text in val_texts]

    # Initialize and fit vectorizer
    vectorizer = TFIDFVectorizer()
    vectorizer.fit(train_tokens)

    # Transform texts to vectors
    X_train = np.array([vectorizer.transform(sentence) for sentence in train_tokens])
    X_test = np.array([vectorizer.transform(sentence) for sentence in val_tokens])

    # Encode labels
    unique_labels = list(set(train_labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_train = np.array([label_to_idx[label] for label in train_labels])
    y_test = np.array([label_to_idx[label] for label in val_labels])

    # Train classifier
    print("Training classifier...")
    classifier = LogisticRegressionClassifier()
    classifier.fit(X_train, y_train)

    # Make predictions on test set
    print("Making predictions...")
    test_predictions = classifier.predict(X_test)

    # Convert numeric predictions back to labels
    predicted_labels = [unique_labels[p] for p in test_predictions]
    true_labels = [unique_labels[y] for y in y_test]

    # Calculate and print metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Function to predict sentiment for new tweets
    def predict_sentiment(tweet, entity):
        combined_text = f"{tweet} [Entity] {entity}"
        processed_text = preprocess_text(combined_text)
        tokens = tokenizer.tokenize(processed_text)
        X = vectorizer.transform(tokens)
        prediction = classifier.predict(np.array([X]))
        return unique_labels[prediction[0]]

    # Example predictions
    print("\nExample Predictions:")
    sample_tweets = [
        ("The gameplay mechanics are amazing!", "Game"),
        ("Let no elim go unnoticed. . . . NVIDIA Highlights automatically records your best moments in @FortniteGame "
         "on GFN!. . Share them with  ", "Nvidia"),
        ("The farmhouse is a mess!", "Housing")
    ]

    for tweet, entity in sample_tweets:
        sentiment = predict_sentiment(tweet, entity)
        print(f"\nTweet: {tweet}")
        print(f"Entity: {entity}")
        print(f"Predicted Sentiment: {sentiment}")


train_text_classification()
