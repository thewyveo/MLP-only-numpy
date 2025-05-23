import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes Adam optimizer
        
        params:
            learning_rate: step size for updates
            beta1: exponential decay rate for first moment estimates
            beta2: exponential decay rate for second moment estimates
            epsilon: small constant to prevent division by zero
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None  # first moment estimate for weights
        self.v_weights = None  # second moment estimate for weights
        self.m_biases = None   # first moment estimate for biases
        self.v_biases = None   # second moment estimate for biases
        self.t = 0             # time step
        
    def initialize(self, weights, biases):
        """Initialize moment estimates based on model's weights and biases"""
        self.m_weights = [np.zeros_like(w) for w in weights]
        self.v_weights = [np.zeros_like(w) for w in weights]
        self.m_biases = [np.zeros_like(b) for b in biases]
        self.v_biases = [np.zeros_like(b) for b in biases]
        
    def update(self, weights, biases, weight_gradients, bias_gradients):
        """
        Update weights and biases using Adam optimization
        
        Args:
            weights: Current weights
            biases: Current biases
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
            
        Returns:
            Updated weights and biases
        """
        if self.m_weights is None:
            self.initialize(weights, biases)
            
        self.t += 1
        
        # Update for each layer
        for i in range(len(weights)):
            # Update for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * np.square(weight_gradients[i])
            
            # Bias correction
            m_hat = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_weights[i] / (1 - self.beta2 ** self.t)
            
            # Update weights
            weights[i] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Update for biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * np.square(bias_gradients[i])
            
            # Bias correction
            m_hat = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update biases
            biases[i] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        return weights, biases
    
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size=1, optimizer=None, learning_rate=0.01, gdescent='minibatch', **optimizer_params):
        self.learning_rate = learning_rate
        self.output_size = output_size

        # Xavier initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) 
            * np.sqrt(1.0 / layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]
        
        # Set optimizer
        if optimizer is not None:
            self.optimizer_name = optimizer.lower()
            if self.optimizer_name == 'adam':
                self.optimizer = AdamOptimizer(learning_rate=learning_rate, **optimizer_params)
        else:
            # Default to SGD
            self.optimizer = None
            self.optimizer_name = None

        self.gdescent = gdescent

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
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights)):
            weight_gradients.append(self.activations[i].T.dot(deltas[i]))
            bias_gradients.append(np.sum(deltas[i], axis=0, keepdims=True))

        if self.optimizer_name == 'adam':
            self.weights, self.biases = self.optimizer.update(
                self.weights, self.biases, weight_gradients, bias_gradients
            )
        else:
            # Standard GD
            for i in range(len(self.weights)):
                self.weights[i] += weight_gradients[i] * self.learning_rate
                self.biases[i] += bias_gradients[i] * self.learning_rate

    def train(self, X, y, epochs=100, batch_size=16, patience=5):
        """Train the network with/without mini-batch gradient descent and with early stopping"""
        best_accuracy = 0
        patience_counter = 0
        best_weights = None
        best_biases = None
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            if self.gdescent == 'minibatch':    # Mini-batch gradient descent
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]

                    output = self.forward(X_batch)
                    self.backward(X_batch, y_batch, output)
            
            else:   # Stochastic gradient descent
                for i in range(n_samples):
                    X_sample = X_shuffled[i:i+1]
                    y_sample = y_shuffled[i:i+1]
                    
                    output = self.forward(X_sample)
                    self.backward(X_sample, y_sample, output)

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
            
            if self.gdescent == 'minibatch':
                loss = np.mean(np.square(y_batch - output))            
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

            else:
                loss = np.mean(np.square(y_sample - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def predict(self, X, verbose=False):
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
    
def preprocess_data(data, train_ratio=0.8, return_all=False):
    """
    Preprocesses mushroom data with proper one-hot encoding and normalization.
    
    Args:
        data: String containing the mushroom dataset
        train_ratio: Proportion of data to use for training (default: 0.8)
        
    Returns:
        X_train, y_train, X_test, y_test, X, y: Processed datasets
    """
    encoding_dicts = {
        "class": {"e": 0, "p": 1}, #target label
        "cap-shape": {"b": 0, "c": 1, "x": 2, "f": 3, "k": 4, "s": 5},
        "cap-surface": {"f": 0, "g": 1, "y": 2, "s": 3},
        "cap-color": {"n": 0, "b": 1, "c": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9},
        "bruises": {"t": 0, "f": 1},
        "odor": {"a": 0, "l": 1, "c": 2, "y": 3, "f": 4, "m": 5, "n": 6, "p": 7, "s": 8},
        "gill-attachment": {"a": 0, "d": 1, "f": 2, "n": 3},
        "gill-spacing": {"c": 0, "w": 1, "d": 2},
        "gill-size": {"b": 0, "n": 1},
        "gill-color": {"k": 0, "n": 1, "b": 2, "h": 3, "g": 4, "r": 5, "o": 6, "p": 7, "u": 8, "e": 9, "w": 10, "y": 11},
        "stalk-shape": {"e": 0, "t": 1},
        "stalk-surface-above-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-surface-below-ring": {"f": 0, "y": 1, "k": 2, "s": 3},
        "stalk-color-above-ring": {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},
        "stalk-color-below-ring": {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},
        "veil-type": {"p": 0, "u": 1},
        "veil-color": {"n": 0, "o": 1, "w": 2, "y": 3},
        "ring-number": {"n": 0, "o": 1, "t": 2},
        "ring-type": {"c": 0, "e": 1, "f": 2, "l": 3, "n": 4, "p": 5, "s": 6, "z": 7},
        "spore-print-color": {"k": 0, "n": 1, "b": 2, "h": 3, "r": 4, "o": 5, "u": 6, "w": 7, "y": 8},
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
            feature_key = list(encoding_dicts.keys())[i + 1]  # Skip the class key
            one_hot = [0] * len(encoding_dicts[feature_key])
            index = encoding_dicts[feature_key].get(feature, -1)
            if index != -1:
                one_hot[index] = 1
            encoded_features.extend(one_hot)

        X_data.append(encoded_features)

    # Convert to numpy arrays
    X = np.array(X_data, dtype=float)
    y = np.array(y_data, dtype=float).reshape(-1, 1)

    if return_all:
        return X, y

    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    cross_val_size = int(n_samples * (1 - train_ratio) / 2)
    train_indices = indices[:train_size]
    cross_indices = indices[train_size : train_size + cross_val_size]
    test_indices = indices[train_size + cross_val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_cross_val, y_cross_val = X[cross_indices], y[cross_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Normalize using training data statistics
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-5] = 1  # Prevent division by zero
    
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    X_cross_val_normalized = (X_cross_val - mean) / std
    
    return X_train_normalized, y_train, X_test_normalized, y_test, X_cross_val_normalized, y_cross_val, X, y

def run_mushroom_classification_with_cv(data, hidden_sizes_options=[[16], [32], [32, 16]], 
                                       learning_rate_options=[0.01, 0.001], 
                                       optimizer='adam', epochs=10, batch_size=16, gdescent='minibatch', **optimizer_params):
    """Run the mushroom classification model with proper cross-validation"""
    X_train, y_train, X_test, y_test, X_cross_val, y_cross_val, _, _ = preprocess_data(data)
    input_size = X_train.shape[1]
    
    # Store results for each hyperparameter combination
    results = []
    
    # Grid search over hyperparameters
    for hidden_sizes in hidden_sizes_options:
        for lr in learning_rate_options:
            print(f"\nTrying hidden_sizes={hidden_sizes}, learning_rate={lr}")
            
            # Train the model with current hyperparameters
            model = MLP(input_size, hidden_sizes, optimizer=optimizer, 
                        learning_rate=lr, gdescent=gdescent, **optimizer_params)
            model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
            
            # Evaluate on cross-validation set
            cv_metrics = model.evaluate(X_cross_val, y_cross_val)
            print(f"Cross-validation accuracy: {cv_metrics['accuracy']:.4f}")
            
            # Store results
            results.append({
                'hidden_sizes': hidden_sizes,
                'learning_rate': lr,
                'model': model,
                'cv_accuracy': cv_metrics['accuracy']
            })
    
    # Find best model based on cross-validation accuracy
    best_result = max(results, key=lambda x: x['cv_accuracy'])
    best_model = best_result['model']
    
    print("\nBest hyperparameters:")
    print(f"Hidden sizes: {best_result['hidden_sizes']}")
    print(f"Learning rate: {best_result['learning_rate']}")
    
    # Evaluate best model on test set (final evaluation)
    test_metrics = best_model.evaluate(X_test, y_test)
    print("\nTest Set Results with best model:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    
    return best_model

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics including error measurements.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Classification metrics
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Error metrics
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mse": mse,
        "rmse": rmse
    }

def create_baseline_model(X_train, y_train, X_test, y_test):
    """
    Creates and evaluates a simple baseline model using majority class prediction.
    """
    # Find the majority class in the training set
    majority_class = round(np.mean(y_train))
    
    # Predict the majority class for all test samples
    predictions = np.full(y_test.shape, majority_class)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    print("\nBaseline Model Results (Majority Class):")
    print(f"Majority Class: {majority_class}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    return metrics

if __name__ == "__main__":
    with open("processed-mushroom.data", "r") as file:
        data = file.read()
     
    X_train, y_train, X_test, y_test, X_cross_val, y_cross_val, _, _ = preprocess_data(data)
    create_baseline_model(X_train, y_train, X_test, y_test)

    # Example usage with Adam optimizer
    run_mushroom_classification_with_cv(
        data, 
        optimizer='adam', 
        gdescent='minibatch'
    )