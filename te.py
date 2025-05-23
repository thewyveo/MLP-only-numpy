import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer

# Load and prepare data
data = pd.read_csv('processed-mushroom.data', header=None)

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

# Select only 'class', 'odor', and 'habitat' columns
selected_data = data[['class', 'odor', 'habitat']]

# Encode the target variable (class)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(selected_data['class'])

# Use OneHotEncoder for features to preserve category information
preprocessor = ColumnTransformer(
    transformers=[
        ('odor', OneHotEncoder(sparse_output=False), ['odor']),
        ('habitat', OneHotEncoder(sparse_output=False), ['habitat'])
    ],
    remainder='drop'  # Drop any unused columns
)

# Apply the transformations
X = preprocessor.fit_transform(selected_data[['odor', 'habitat']])

# Get feature names for later reference
feature_names = preprocessor.get_feature_names_out()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print results
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))