import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("diabetes-processed.csv")

label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])  # Female=0, Male=1

X = df.drop(columns=['diabetes'])  # Features
y = df['diabetes']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# to use Adam optimizer change solver='adam'
# change hidden_layer_sizes=(8, 4) to use 8 and 4 neurons in the hidden layer
# see scikit MLPClassifier documentation for other default values of the parameters
mlp = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.01, solver='sgd', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))