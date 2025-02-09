import numpy as np
from sklearn.linear_model import LogisticRegression


# Logistic Regression Classifier
class LogisticRegressionClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)


# FeedForward Neural Network (FNN)
class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights & Biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivation(self, x):
        return np.where(x > 0, 1, 0)  # derivation of ReLU

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Avoiding log(0)
        return loss

    def forward(self, x):
        self.Z1 = np.dot(x, self.W1) + self.b1  # Linear transform
        self.A1 = self.relu(self.Z1)  # Activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Output layer
        self.A2 = self.softmax(self.Z2)  # Softmax for classification
        return self.A2

    def backward(self, X, y_true):
        m = X.shape[0]

        # Computing gradients (By taking derivative)
        dZ2 = self.A2 - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivation(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Updating weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, x_train, y_train, steps=100):
        for step in range(steps):
            # Forward Propagation
            y_pred = self.forward(x_train)

            # Compute Loss
            loss = self.cross_entropy_loss(y_train, y_pred)

            # Backward Propagation
            self.backward(x_train, y_train)

            if step % 10 == 0:
                print(f"Step: {step} | Loss: {loss:.4f}")

    def predict(self, x_test):
        y_pred = self.forward(x_test)
        return np.argmax(y_pred, axis=1)  # Converting softmax probabilities into class labels

