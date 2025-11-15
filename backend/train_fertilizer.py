import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load CSV (given your folder structure)
df = pd.read_csv("data/Crop_recommendation_with_fertilizer_packages_full.csv")

# Select input features
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

print(df.columns)

# Target column
y = df["Fertilizer_Package"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# Save model in /models folder
with open("models/fertilizer_recommendation.pkl", "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)

print("Model saved in models/fertilizer_recommendation.pkl")
