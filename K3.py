import numpy as np      # the ONLY library we used in this implementation is numpy.

class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=False):
        """
        Initializes Adam optimizer
        
        Args:
            learning_rate: step size for updates
            beta1: exponential decay rate for first moment estimates (we dont change this from the default)
            beta2: exponential decay rate for second moment estimates (we dont change this from the default)
            epsilon: small constant to prevent division by zero
        """
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0             # time step
        self.m_weights = None  # moment initialization for weights & biases
        self.v_weights = None
        self.m_biases = None
        self.v_biases= None

        if self.verbose:
            print(f"AdaM: beta1 [{self.beta1}] - beta2 [{self.beta2}] - eps [{self.epsilon}]")
        
    def initialize(self, weights, biases):
        """
        Initializes moment estimates based on model's weights and biases.
        NOTE: This is different from the first __init__ method, because there we actually "initialize"
        the values by setting them to None, and here we actually set (initialize!) the values to the
        correct shape and size.

        Args:
            weights: list of weight matrices for each layer in the model (list of numpy.ndarray)
            biases: list of bias vectors for each layer in the model. (list of numpy.ndarray)
        """

        self.m_weights = [np.zeros_like(w) for w in weights]    # first moment estimate (weights)
        self.v_weights = [np.zeros_like(w) for w in weights]    # second moment estimate (weights)
        self.m_biases = [np.zeros_like(b) for b in biases]      # first moment estimate (biases)
        self.v_biases = [np.zeros_like(b) for b in biases]      # second moment estimate (biases)
        
    def update(self, weights, biases, weight_gradients, bias_gradients):
        """
        Updates weights and biases using Adam optimization
        
        Args:
            weights: list of weight matrices for each layer in the model (list of numpy.ndarray)
            biases: list of bias vectors for each layer in the model. (list of numpy.ndarray)
            weight_gradients: list of gradients for weights (list of numpy.ndarray)
            bias_gradients: list of gradients for biases (list of numpy.ndarray)

        Returns:
            updated_weights: list of updated weight matrices (list of numpy.ndarray)
            updated_biases: list of updated bias vectors (list of numpy.ndarray)
        """

        if self.m_weights is None:   
            self.initialize(weights, biases)
            
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        if self.verbose:
            print(f"AdaM: lr-time-adj [{lr_t:.8f}]")
        
        updated_weights = []
        updated_biases = []
        
        # iterate over layers
        for i in range(len(weights)):
            # update weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * np.square(weight_gradients[i])
            if self.verbose:
                print(f"AdaM: m-weights µ: [{np.mean(self.m_weights[i]):.8f}] v-weights µ: [{np.mean(self.v_weights[i]):.8f}]")

            
            # update weights with combined step
            updated_w = weights[i] - lr_t * self.m_weights[i] / (np.sqrt(self.v_weights[i]) + self.epsilon)
            updated_weights.append(updated_w)
            
            # update biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * np.square(bias_gradients[i])
            if self.verbose:
                print(f"AdaM: m-biases µ: [{np.mean(self.m_biases[i]):.8f}] v-biases µ: [{np.mean(self.v_biases[i]):.8f}]")

            
            # update biases with combined step
            updated_b = biases[i] - lr_t * self.m_biases[i] / (np.sqrt(self.v_biases[i]) + self.epsilon)
            updated_biases.append(updated_b)
            
        return updated_weights, updated_biases
    
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size=1, optimizer=None, learning_rate=0.01, gdescent='minibatch', metric_for_selection='accuracy', verbose=False, **optimizer_params):
        """
        Initializes the Multi-Layer Perceptron model
        
        Args:
            input_size: number of features in the input data
            hidden_sizes: list of hidden layer sizes
            output_size: number of output classes (in our case, its always 1 because this is a binary classification task)
            optimizer: optimization algorithm ('adam' or None)
            learning_rate: step size for updates
            gdescent: type of gradient descent ('minibatch' or 'stochastic')
            metric_for_selection: metric to use for model selection ('accuracy', 'f1_score', etc.)
            optimizer_params: additional parameters for the optimizer
        """
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.verbose = verbose

        # Xavier initialization (modified to be a middleground between Xavier and He initialization, to be applicable to ReLU and Sigmoid)
        # The logic here is that normal Xavier initialization is defined as (1 / nin+nout),
        # and He initialization is defined as (2 / nin).
        # our initialization formula, (2 / (nin + nout)), is a middleground between the two.
        # this ensures that the weights are not too small (like Xavier) or too large (like He) - and that our activations are in the sweet spot.
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        if self.verbose:
            print(f"MLP: layer-sizes [{len(layer_sizes)}]")
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) 
            * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]
        
        # optimizer setup
        if optimizer is not None:
            self.optimizer_name = optimizer.lower()
            if self.optimizer_name == 'adam':
                self.optimizer = AdamOptimizer(learning_rate=learning_rate, verbose=True, **optimizer_params)
        else:
            # no optimizer option, this uses either stochastic GD or minibatch GD.
            self.optimizer = None
            self.optimizer_name = None

        # gradient descent type
        self.gdescent = gdescent

        # metric for selection from validation set
        self.metric_for_selection = metric_for_selection


    # Sigmoid activation is used for the output layer, as it is a binary classification problem.
    # Sigmoid also has the benefit of fitting values between 0 and 1, letting us calculate probabilities.
    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-np.clip(x, -50, 50)))
        #if self.verbose:
            #print(f"MLP: activate-sigmoid [{output[0:10]}]")
        """The sigmoid activation function maps any real-valued number into the range (0, 1). This produces probabilities."""
        return output

    def sigmoid_derivative(self, sigmoid_output):
        output = sigmoid_output * (1 - sigmoid_output)
        #if self.verbose:
            #print(f"MLP: derivate-sigmoid [{output[0:10]}]")
        """Derivative of the sigmoid function, used for backpropagation."""
        return output
        
    # ReLU activation is used for hidden layers, maintaining efficient gradient propagation.
    def relu(self, x):
        output = np.maximum(0, x)
        #if self.verbose:
            #print(f"MLP: activate-relu [{output[0:10]}]")
        """The ReLU activation function returns the input directly if it is positive; otherwise, it returns zero."""
        return output

    def relu_derivative(self, x):
        output = (x > 0).astype(float)
        #if self.verbose:
            #print(f"MLP: derivate-relu [{output[0:10]}]")
        """Derivative of the ReLU function, used for backpropagation."""
        return output

    # The loss function was chosen to be BCE as it is used for binary classification problems, and works with sigmoid well.
    # The initial loss function was set to be MSE, but was changed to BCE to better fit the problem as it provides
    # stronger gradients and better convergence.   
    def binary_cross_entropy(self, y_true, y_pred):
        """
        Binary cross-entropy loss function
        
        Args:
            y_true: true labels
            y_pred: predicted labels
            
        Returns:
            binary cross-entropy loss between true and predicted labels
        """

        epsilon = 1e-15     # to prevent log(0) = -inf
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if self.verbose:
            print("MLP: loss-binary-cross-entropy [", -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), "]")

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward(self, X):
        """
        Forward pass with ReLU for hidden layers and sigmoid for output
        
        Args:
            X: input data
            
        Returns:
            output: output of the model (predictions)
        """
        self.layer_inputs = []  # store initial inputs w/out activation
        self.activations = [X]
        
        # hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # output layer with sigmoid
        z_out = self.activations[-1].dot(self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(z_out)
        output = self.sigmoid(z_out)
        self.activations.append(output)

        
        return output

    def backward(self, X, y, output):
        """
        Backward pass with proper derivatives based on activations
        
        Args:
            X: input data
            y: true labels
            output: predicted labels
        """
        m = X.shape[0]  # batch size (depending on GD type, this can be 1 or batch_size)
        
        # compute gradient of BCE w.r.t. output
        if self.output_size == 1:  # Binary classification
            delta_output = output - y
        else:
            raise NotImplementedError("Only binary classification is supported")
            # for multi-class, would need softmax+cross-entropy handling
            # we don't use multi-class in this model, therefore this is a placeholder and we just leave it unimplemented.
        
        deltas = [delta_output]


        # backpropagate through hidden layers (backwards, so starting from last hidden layer)
        for i in range(len(self.weights) - 1, 0, -1):
            # compute delta for current layer via delta = delta(l+1) * weights(l+1)^T * relu_derivative(l)
            delta = deltas[-1].dot(self.weights[i].T) * self.relu_derivative(self.layer_inputs[i-1])
            deltas.append(delta)
        
        # reverse deltas to match layer order
        deltas.reverse()
        
        # initialize lists for gradients
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights)):
            # gradient calculation with normalization by batch size
            weight_grad = self.activations[i].T.dot(deltas[i]) / m
            bias_grad = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)
        
        # update weights and biases with adam if toggled
        if self.optimizer_name == 'adam':
            if self.verbose:
                print("[OPTIMIZATION := AdaM] processing...")
            self.weights, self.biases = self.optimizer.update(
                self.weights, self.biases, weight_gradients, bias_gradients
            )
        else:
            # standard gradient descent otherwise (either minibatch or stochastic)
            for i in range(len(self.weights)):
                if self.verbose:
                    print("[OPTIMIZATION := Gradient_Descent] processing...")
                self.weights[i] -= weight_gradients[i] * self.learning_rate
                self.biases[i] -= bias_gradients[i] * self.learning_rate

    def train(self, X, y, epochs=100, batch_size=16, patience=5):
        """
        Model training with early stopping
        
        Args:
            X: input data
            y: true labels
            epochs: maximum number of training epochs
            batch_size: batch size (for mini-batch gradient descent)
            patience: number of epochs to wait before early stopping
        """
        n_samples = X.shape[0]  # number of samples
        patience_counter = 0    # patience counter for early stopping
        best_metric = 0       # initializations
        best_weights = None
        best_biases = None
        
        # training phase
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            if self.gdescent == 'minibatch':  
                if self.verbose:
                    print("[OPTIMIZATION := Gradient_Descent (BATCHED)] processing...")  # minibatch gradient descent
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    
                    output = self.forward(X_batch)  #forward
                    batch_loss = self.binary_cross_entropy(y_batch, output) #loss
                    epoch_losses.append(batch_loss) #append loss
                    self.backward(X_batch, y_batch, output) #backward
            else:                               # stochastic gradient descent
                if self.verbose:
                    print("[OPTIMIZATION := Gradient_Descent (STOCHASTIC)] processing...")
                for i in range(n_samples):
                    X_sample = X_shuffled[i:i+1]
                    y_sample = y_shuffled[i:i+1]
                    
                    output = self.forward(X_sample) #same as above, but for each sample instead of batch
                    sample_loss = self.binary_cross_entropy(y_sample, output)
                    epoch_losses.append(sample_loss)
                    self.backward(X_sample, y_sample, output)
            
            # calculate metrics
            metrics = self.evaluate(X, y)
            
            # check for early stopping depending on whether metric keeps improving
            if metrics[self.metric_for_selection] > best_metric:
                best_metric = metrics[self.metric_for_selection]
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0    #if so, patience_counter is repeatedly set to 0, and the loop continues
                if self.verbose:
                    print("Ptc: 0 - TERMINATING...")
            else:
                patience_counter += 1   #if improvement stops improving, patience_counter is incremented by 1
                if self.verbose:
                    print("Ptc: ", patience_counter)
            if patience_counter >= patience:    #when this counter hits its threshold, the loop breaks
                self.weights = best_weights     #and the best weights and biases are kept
                self.biases = best_biases
                break
    
    def predict(self, X):
        """
        Classification prediction of the model
        
        Args:
            X: input data
        Returns:
            prediction of the model, as either a 0 or 1 (rather than True/False)
        """
        output = self.forward(X)
        output2 = (output > 0.5).astype(int)
        if self.verbose:
            print(f"MLP: compute-predict [{output2}]")
        return output2

    def evaluate(self, X, y):
        """
        Evaluation of the model's performance (using various metrics; like precision, recall, and so on).

        Args:
            X: input data
            y: true labels

        Returns:
            a dictionary of all metrics
        """
        predictions = self.predict(X).flatten() #flatten to match y's shape
        y = y.flatten() #flatten to match predictions' shape, as they need to be the same shape for comparison

        # calculate all metrics
        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == 0))
        TN = np.sum((predictions == 0) & (y == 0))
        FN = np.sum((predictions == 0) & (y == 1))
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return_statement = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "TP": TP, "FP": FP, 
                "TN": TN, "FN": FN
            }
        }

        if self.verbose:
            print(f"MLP: compute-evaluate [{return_statement['confusion_matrix']}]")

        return return_statement
        
    def get_feature_importance(self, X, y):
        """
        Calculates feature importance by measuring how much each feature
        affects the prediction when perturbed

        Args:
            X: input data
            y: true labels
        
        Returns:
            importances: a list of feature importances
        """

        base_preds = self.predict(X)                            # model's predictions on the original data
        base_acc = np.mean(base_preds.flatten() == y.flatten()) # accuracy of the model on the original data
        
        importances = []
        
        for i in range(X.shape[1]):     # iterate over each feature, X.shape[1] is the number of features
            X_shuffled = X.copy()       # create a copy
            X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])  # random shuffle on the i-th feature (random permutation)
            
            preds_shuffled = self.predict(X_shuffled)   # model's predictions on the dataset with the i-th feature shuffled
            acc_shuffled = np.mean(preds_shuffled.flatten() == y.flatten()) # and its accuracy
            
            importance = base_acc - acc_shuffled    # importance is calculated as the decrease in accuracy when feature is shuffled
            importances.append(importance)          # append to the list

        if self.verbose:
            print(f"MLP: compute-feature-imp [{importances}]")
            
        return importances
    
def preprocess_data(data, train_ratio=0.8, verbose=False):
    """
    Preprocesseses the diabetes dataset.

    Args:
        data: dataset
        train_ratio: proportion of data to use for training (default is 0.8, validation and test sets are split equally)

    Returns:
        X_train: training features
        y_train: training labels
        X_test: test features
        y_test: test labels
        X_val: validation features
        y_val: validation labels
        X: all features
        y: all labels
    """
    print("Preprocessing...")

    # define encoding for categorical features (in this case, only gender)
    encoding_dicts = {
        "gender": {"Female": 0, "Male": 1},
    }
    
    X_data = []
    y_data = []
    
    # iterate through each line of the dataset
    for line in data.strip().split("\n")[1:]:
        features = line.strip().split(",")
        
        y_data.append(int(features[-1]))    # extract target value (diabetes --> 0 or 1)
        
        encoded_features = []   # initialize encoded features list
        
        # gender encoding (categorical)
        gender = features[0]  # (gender is the first feature, hence 0)
        encoded_features.append(encoding_dicts["gender"].get(gender, -1))  # encode gender
        # NOTE: we do not apply one-hot encoding to gender, as the values are only in binary (0 or 1) and
        # since we're working with a NN, it can handle binary values well - therefore even though the
        # gender is encoded "numerically" it is treated as a categorical feature.
        # (i.e. even though it is not [1, 0] for 0 and [0, 1] for 1; it is still categorical)
        
        # numerical features are: age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level
        numerical_features = features[1:-1]  # all features except gender and diabetes target
        encoded_features.extend([float(x) for x in numerical_features])  # convert to float
        
        X_data.append(encoded_features) # and append to the X_data list

    # convert to numpy arrays
    X = np.array(X_data, dtype=float)
    y = np.array(y_data, dtype=float).reshape(-1, 1)
    
    # normalize only the numerical data, excluding the first categorical feature
    numerical_indices = list(range(1, X.shape[1]))  # indexes for numerical features, we skip the first feature (gender)
    mean = np.mean(X[:, numerical_indices], axis=0) # mean and std are calculated only on the numerical features with indexing
    std = np.std(X[:, numerical_indices], axis=0)
    if verbose:
        print(f"Pre: µ [{mean}] - σ [{std}]")
    std[std < 1e-5] = 1  # prevent division by zero
    
    # normalize the data
    X_normalized = X.copy()
    X_normalized[:, numerical_indices] = (X[:, numerical_indices] - mean) / std
    
    # set data split sizes
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)         # the training set size is set to be train_ratio
    val_size = int(n_samples * (1 - train_ratio) / 2) # the / 2 is to split the remaining data into validation and test sets
    train_indices = indices[:train_size]              # e.g. if train_ratio = 0.8, then the rest is split into another 0.1 and 0.1
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    if verbose:
        print("TRAIN: ", f"{train_size * 100:.0f}%", "TEST: ", f"{val_size * 100:.0f}%", "VAL: ", f"{val_size * 100:.0f}%")
    
    # execute the split
    X_train, y_train = X_normalized[train_indices], y[train_indices]
    X_val, y_val = X_normalized[val_indices], y[val_indices]
    X_test, y_test = X_normalized[test_indices], y[test_indices]
    
    return X_train, y_train, X_test, y_test, X_val, y_val, X, y

def run_classification(data, 
                      hidden_sizes_options=[[16], [32], [64], [32, 16], [64, 32], [64, 32, 16]],
                      learning_rate_options=[0.01, 0.005, 0.001],
                      optimizer='adam',
                      epochs=50,
                      batch_size=32,
                      gdescent='minibatch',
                      patience=5,
                      random_seed=None,
                      train_ratio = 0.8,
                      metric_for_selection='accuracy',
                      verbose= False,
                      **optimizer_params):
    """
    Run diabetes classification with hyperparameter tuning
    
    Args:
        data: string containing the dataset
        hidden_sizes_options: list of hidden layer configurations to try
        learning_rate_options: list of learning rates to try
        optimizer: optimization algorithm ('adam' or None, default is 'adam')
        epochs: maximum number of training epochs (default is 50)
        batch_size: batch size (for mini-batch gradient descent, default is 32')
        gdescent: type of gradient descent ('minibatch' or 'stochastic', default is 'minibatch')
        patience: number of epochs to wait before early stopping (default is 5)
        train_ratio: proportion of data to use for training (default is 0.8, validation and test sets are split equally)
        random_seed: random seed for reproducibility (default is None)
        metric_for_selection: metric to use for model selection ('accuracy', 'f1_score', etc.; default is 'accuracy')
        optimizer_params: additional parameters for the optimizer
        
    Returns:
        best_model: the best performing model
        all_results: complete results for all hyperparameter combinations (grid-search)
    """

    # set seed if provided in params, otherwise defaults to None
    if random_seed is not None:
        np.random.seed(random_seed)
        if verbose:
            print(f"SEED: {random_seed}")
    
    # preprocess data
    X_train, y_train, X_test, y_test, X_val, y_val, X_orig, y_orig = preprocess_data(data, train_ratio=train_ratio)
    input_size = X_train.shape[1]
    
    # store results for each hyperparameter combination
    results = []
    
    if verbose:
        print("Starting hyperparameter selection via grid-search, via a validation set...")
        print(f"Using {metric_for_selection} for model selection...")
    
    total_combinations = len(hidden_sizes_options) * len(learning_rate_options) # total number of combinations
    iterations = 0                                                              # to keep track of the progress
    # grid-search over hyperparameters
    for hidden_sizes in hidden_sizes_options:
        for lr in learning_rate_options:
            
            # define a model with current hyperparameters
            model = MLP(input_size, 
                      hidden_sizes, 
                      optimizer=optimizer, 
                      learning_rate=lr, 
                      gdescent=gdescent,
                      metric_for_selection = metric_for_selection, 
                      verbose= True,
                      **optimizer_params)
            
            # train with early stopping for each model
            model.train(X_train, y_train, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       patience=patience)
            
            # evaluate on validation set
            validation_metrics = model.evaluate(X_val, y_val)
            
            results.append({
                'hidden_sizes': hidden_sizes,
                'learning_rate': lr,
                'model': model,
                'validation_metrics': validation_metrics
            })

            iterations += 1
            if iterations % 3 == 0:
                print(f"    Completed {iterations}/{total_combinations} iterations")
    print("Hyperparameter selection via grid-search complete.")
    
    # find best model based on selected metric
    best_result = max(results, key=lambda x: x['validation_metrics'][metric_for_selection])
    best_model = best_result['model']
    
    print("Best hyperparameters:")
    print(f"    Hidden sizes: {best_result['hidden_sizes']}")
    print(f"    Learning rate: {best_result['learning_rate']}")
    print(f"    Validation {metric_for_selection}: {best_result['validation_metrics'][metric_for_selection]:.4f}")
    
    # evaluate best model on test set (final evaluation)
    test_metrics = best_model.evaluate(X_test, y_test)
    print("\nTest Set results with best model:")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall: {test_metrics['recall']:.4f}")
    print(f"    F1-Score: {test_metrics['f1_score']:.4f}")
    
    # get feature importance (if applicable)
    try:
        importances = best_model.get_feature_importance(X_train, y_train)   # call the get_feature_importance method on the best model
        features = ["Gender", "Age", "Hypertension", "Heart Disease", "BMI", "HbA1c", "Blood Glucose"]
        
        print("\nFeature Importance:")
        # iterating over tuples of (feature, importance) that are zipped together, and sorting by importance
        for i, (feature, importance) in enumerate(sorted(zip(features, importances), key=lambda x: abs(x[1]), reverse=True)):
            print(f"    {i+1}. {feature}: {importance:.4f}")
    except:
        print("\nFeature importance calculation not available")
    print()
    
    return best_model, results  # this return is a placeholder, we don't actually use it for anything.
                                # however it could possibly be used to store the model, or to analyze results for further insight.

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics including error measurements.
    NOTE: This function is a copy of the evaluate method in the MLP class, but is standalone, for the reason that
    it can be used to evaluate any model's predictions, not just the MLP model. Since we create the baseline model
    outside of our actual main MLP class, this function is used to evaluate the baseline model's predictions.
    This is applicable, since the baseline model only predicts the majority class, and does not any methods from the MLP class.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        
    Returns:
        a dictionary of all metrics
    """
    y_true = y_true.flatten() # flatten to match y_pred's shape
    y_pred = y_pred.flatten() # flatten to match y_true's shape, as they need to be the same shape for comparison
    
    # calculate all metrics
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

def create_baseline_model(y_train, y_test):
    """
    Creates and evaluates a simple baseline model using majority class prediction.

    Args:
        y_train: training labels
        y_test: test labels

    Returns:
        metrics: a dictionary of all metrics for the baseline model
    """
    # find the majority class in the training set (count the number of 1s and 0s for the target label)
    majority_class = round(np.mean(y_train))
    
    # predict the majority class for all instances in the test set
    predictions = np.full(y_test.shape, majority_class)
    
    # calculate metrics
    
    print("\nBaseline Model Results (Majority Class):")
    print(f"    Majority Class: {majority_class}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1-Score: {metrics['f1_score']:.4f}")
    print()
    
    return metrics

def full_run():
    with open("diabetes-processed.csv", "r") as file:
        data = file.read()

    X_train, y_train, X_test, y_test, X_val, y_val, _, _ = preprocess_data(data, verbose=True)
    create_baseline_model(y_train, y_test)

    model_AdaM_Mini, results_AdaM_Mini = run_classification(data, optimizer='AdaM', gdescent='minibatch', random_seed=None, metric_for_selection='accuracy', verbose=True)
    model_AdaM_SGD, results_AdaM_SGD = run_classification(data, optimizer='AdaM', gdescent='sgd', random_seed=None, metric_for_selection='accuracy', verbose=True)
    model_Mini, results_Mini = run_classification(data, optimizer='None', gdescent='minibatch', random_seed=None, metric_for_selection='accuracy', verbose=True)
    model_SGD, results_SGD = run_classification(data, optimizer='None', gdescent='sgd', random_seed=None, metric_for_selection='accuracy', verbose=True)

if __name__ == "__main__":
    full_run()