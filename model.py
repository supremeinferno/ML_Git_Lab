import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Area": [800, 1000, 1200, 1500],
    "Price": [40, 50, 60, 75]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")
