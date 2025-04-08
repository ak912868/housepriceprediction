import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv("data/house_data.csv")

# Drop missing values if any (none in your case)
df = df.dropna()

# Encode categorical features
label_encoders = {}
categorical_cols = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'] + categorical_cols
X = df[feature_cols]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Create output directories if not exist
os.makedirs("model", exist_ok=True)

# Save model
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved successfully.")
