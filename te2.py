import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare data
data = pd.read_csv('processed-mushroom.data', header=None, delimiter=',')

# Define column names
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
    'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]
data.columns = column_names

# Select only 'class' and 'odor' columns
selected_data = data[['class', 'odor']]

# Encode the target variable (class)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(selected_data['class'])

# Use OneHotEncoder for the 'odor' feature
odor_values = selected_data[['odor']]
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(odor_values)

# Get the odor categories for later reference
odor_categories = encoder.categories_[0]

# Split the data (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))