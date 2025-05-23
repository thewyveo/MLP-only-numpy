import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Xavier initialization for better gradient flow
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def forward(self, X):
        """Forward pass through the network"""
        self.layer1 = X.dot(self.weights1) + self.bias1
        self.activation1 = self.sigmoid(self.layer1)
        self.layer2 = self.activation1.dot(self.weights2) + self.bias2
        self.activation2 = self.sigmoid(self.layer2)
        return self.activation2

    def backward(self, X, y, output):
        """Backward pass to update weights"""
        # Calculate gradients
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(self.activation2)

        # Hidden layer gradients
        self.hidden_error = self.output_delta.dot(self.weights2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.activation1)

        # Update weights and biases
        self.weights2 += self.activation1.T.dot(self.output_delta) * self.learning_rate
        self.bias2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights1 += X.T.dot(self.hidden_delta) * self.learning_rate
        self.bias1 += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000, verbose=True):
        """Train the network"""
        best_accuracy = 0
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, y, output)

            # Dynamic learning rate adjustment
            if epoch % 1000 == 0:
                self.learning_rate = max(self.learning_rate * 0.95, 1e-6) # Reduce learning rate over time

            # Calculate and print metrics
            if verbose:  # Increased frequency of reporting
                metrics = self.evaluate(X, y)
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

                # Early stopping if accuracy is not improving
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']

    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the model's performance using accuracy, precision, recall, and F1-score.

        Parameters:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): True labels.
        """
        # Make predictions
        predictions = self.predict(X)

        # Flatten predictions and true labels
        predictions = predictions.flatten()
        y = y.flatten()

        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == 0))
        TN = np.sum((predictions == 0) & (y == 0))
        FN = np.sum((predictions == 0) & (y == 1))

        accuracy = (TP + TN) / (TP + FP + TN + FN)

        # Handle division by zero for precision, recall, and F1-score
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }


def preprocess_data(data, train_ratio=0.8):
    """Preprocess mushroom data and split into train/test sets"""
    encoding_dicts = {
        'class': {'e': 0, 'p': 1},
        'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
        'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
        'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6},
        'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
        'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
        'gill-size': {'b': 0, 'n': 1}
    }

    # Parse data lines
    X_data = []
    y_data = []
    valid_lines = 0

    for line in data.strip().split('\n'):
        features = line.strip().split(',')

        # Extract class (target)
        target_class = encoding_dicts['class'][features[0]]

        # Extract selected features
        selected_features = [features[5], features[19], features[21], features[20], features[1], features[8]]
        feature_vector = []
        for i, feature in enumerate(selected_features):
            feature_key = list(encoding_dicts.keys())[i + 1]  # +1 to skip 'class'
            try:
                # Get the encoded value for the feature
                encoded_value = encoding_dicts[feature_key][feature]
                feature_vector.append(encoded_value)
            except KeyError:
                # Handle unknown feature values
                print(f"Warning: Unknown value '{feature}' for feature '{feature_key}'")
                feature_vector.append(0)  # Default to 0 for unknown values

        X_data.append(feature_vector)
        y_data.append(target_class)
        valid_lines += 1

    print(f"Number of valid lines processed: {valid_lines}")

    # Convert to numpy arrays
    X = np.array(X_data, dtype=float)
    y = np.array(y_data, dtype=float).reshape(-1, 1)

    # Check if X or y is empty
    if X.size == 0 or y.size == 0:
        raise ValueError("No valid data to process. Please check the input data.")

    # Normalize data (important for neural networks)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # Split data into training and testing sets
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test, X, y

def run_mushroom_classification(data, hidden_size=4, learning_rate=0.01, epochs=15):
    """Run the mushroom classification model"""
    # Preprocess data
    X_train, y_train, X_test, y_test, _, _ = preprocess_data(data)

    # Check if X_train or X_test is empty
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Training or testing data is empty. Please check the preprocessing step.")

    # Get input size from data
    input_size = X_train.shape[1]
    output_size = 1  # Binary classification

    # Initialize and train the model
    model = MLP(input_size, hidden_size, output_size, learning_rate)
    print("Training model...")
    model.train(X_train, y_train, epochs=epochs)

    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    return model


if __name__ == "__main__":
    with open('processed-mushroom.data', 'r') as f:
        mushroom_data = f.read()

    model = run_mushroom_classification(mushroom_data)