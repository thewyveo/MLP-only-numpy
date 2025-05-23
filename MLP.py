import numpy as np

#np.random.seed(42)

class MLP:
    def __init__( self, input_size, hidden_sizes, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.output_size = output_size

        # Xavier initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [
        np.random.randn(layer_sizes[i], layer_sizes[i + 1])
        * np.sqrt(1.0 / layer_sizes[i])  # Use 1.0 for sigmoid
        for i in range(len(layer_sizes) - 1)
            ]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.sigmoid(X.dot(w) + b)
            self.activations.append(X)
        return X

    def backward(self, X, y, output):
        """Backward pass to update weights"""
        deltas = [(y - output) * self.sigmoid_derivative(output)]

        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append(
                deltas[-1].dot(self.weights[i].T)
                * self.sigmoid_derivative(self.activations[i])
            )
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * self.learning_rate
            self.biases[i] += (
                np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate
            )


    def train(self, X, y, epochs=100, batch_size=16, patience=5):
        """Train the network with mini-batch gradient descent and early stopping"""
        best_accuracy = 0
        patience_counter = 0
        best_weights = None
        best_biases = None
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            # Calculate metrics and check for early stopping
            metrics = self.evaluate(X, y)

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                self.weights = best_weights
                self.biases = best_biases
                break

            loss = np.mean(np.square(y_batch - output))            
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the model's performance using accuracy, precision, recall, and F1-score.
        """
        predictions = self.predict(X).flatten()
        y = y.flatten()

        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == 0))
        TN = np.sum((predictions == 0) & (y == 0))
        FN = np.sum((predictions == 0) & (y == 1))

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }


def preprocess_data(data, train_ratio=0.8):
    """
    Preprocesses mushroom data with proper one-hot encoding and normalization.
    Splits data into train/test sets properly before applying normalization.
    
    Args:
        data: String containing the mushroom dataset
        train_ratio: Proportion of data to use for training (default: 0.8)
        
    Returns:
        X_train, y_train, X_test, y_test, X, y: Processed datasets
    """
    encoding_dicts = {
        "class": {"e": 0, "p": 1},
        "cap-shape": {"b": 0, "c": 1, "x": 2, "f": 3, "k": 4, "s": 5},
        "cap-surface": {"f": 0, "g": 1, "y": 2, "s": 3},
        "cap-color": {"n": 0, "b": 1, "c": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9,},
        "bruises": {"t": 0, "f": 1},
        "odor": {"a": 0, "l": 1, "c": 2, "y": 3, "f": 4, "m": 5, "n": 6, "p": 7, "s": 8,},
        "gill-attachment": {"a": 0, "d": 1, "f": 2, "n": 3},
        "gill-spacing": {"c": 0, "w": 1, "d": 2},
        "gill-size": {"b": 0, "n": 1},
        "gill-color": {"k": 0, "n": 1, "b": 2, "h": 3, "g": 4, "r": 5, "o": 6, "p": 7, "u": 8, "e": 9, "w": 10, "y": 11,},
        "stalk-shape": {"e": 0, "t": 1},
        "stalk-surface-above-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-surface-below-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-color-above-ring": {"n": 0, "b": 1, "c": 2, "g": 3,"o": 4, "p": 5, "e": 6, "w": 7, "y": 8,},
        "stalk-color-below-ring": {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8,},
        "veil-type": {"p": 0, "u": 1},
        "veil-color": {"n": 0, "o": 1, "w": 2, "y": 3},
        "ring-number": {"n": 0, "o": 1, "t": 2},
        "ring-type": {"c": 0, "e": 1, "f": 2, "l": 3, "n": 4, "p": 5, "s": 6, "z": 7},
        "spore-print-color": {"k": 0, "n": 1, "b": 2, "h": 3, "r": 4, "o": 5, "u": 6, "w": 7, "y": 8,},
        "population": {"a": 0, "c": 1, "n": 2, "s": 3, "v": 4, "y": 5},
        "habitat": {"g": 0, "l": 1, "m": 2, "p": 3, "u": 4, "w": 5, "d": 6},
    }

    X_data = []
    y_data = []

    for line in data.strip().split("\n"):
        features = line.strip().split(",")
        y_data.append(1 if features[0] == "p" else 0)

        encoded_features = []
        for i, feature in enumerate(features[1:]):
            feature_key = list(encoding_dicts.keys())[i]
            one_hot = [0] * len(encoding_dicts[feature_key])
            index = encoding_dicts[feature_key].get(feature, -1)
            if index != -1:
                one_hot[index] = 1
            encoded_features.extend(one_hot)

        X_data.append(encoded_features)

    # Convert to numpy arrays
    X = np.array(X_data, dtype=float)
    y = np.array(y_data, dtype=float).reshape(-1, 1)

    # First split the data (before normalization to prevent data leakage)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Now normalize using only training data statistics
    # For one-hot encoded features, you might consider two approaches:
    
    # Option 1: Skip normalization for binary features
    # This keeps one-hot encoded features as 0s and 1s
    # You'd need to identify which columns are one-hot encodings
    
    # Option 2: Normalize everything (used here)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Handle features with zero or near-zero variance
    # Replace small std values with 1 to avoid division issues
    std[std < 1e-5] = 1
    
    # Apply normalization using training set statistics
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    
    # You could experiment with both normalized and raw versions
    # to see which performs better for your model
    
    return X_train_normalized, y_train, X_test_normalized, y_test, X, y

    # Alternative return if you want to experiment with unnormalized data:
    # return X_train, y_train, X_test, y_test, X, y


def run_mushroom_classification(data, hidden_sizes=[32, 16], epochs=10, batch_size=16):
    """Run the mushroom classification model with batch training"""
    X_train, y_train, X_test, y_test, _, _ = preprocess_data(data)
    input_size = X_train.shape[1]

    model = MLP(input_size, hidden_sizes)
    print("Training model...")
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    return model


if __name__ == "__main__":
    with open("processed-mushroom.data", "r") as file:
        data = file.read()
    run_mushroom_classification(data)