import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv(r"/Users/sravya/Desktop/MLOPS pipeline/Data/data.csv")

# Features and target
X = df[["area", "bedrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model (optional but typical for backend use)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)