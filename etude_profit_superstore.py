import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Chargement des données
df = pd.read_csv("Sample - Superstore.csv", encoding='ISO-8859-1')

# 2. Nettoyage : suppression des colonnes inutiles
df.drop(columns=["Row ID", "Order ID", "Customer ID", "Customer Name", "Country", "City",
                 "State", "Postal Code", "Product ID", "Product Name", "Order Date", "Ship Date"], inplace=True)

df.info()

# 3. Visualisation de la distribution du profit
plt.figure(figsize=(8, 5))
sns.histplot(df['Profit'], bins=50, kde=True)
plt.title("Distribution du Profit (Avant traitement)")
plt.show()

# 4. Boîte à moustaches pour détecter les outliers
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Profit')
plt.title("Boîte à moustaches du Profit (Avant traitement)")
plt.show()

print("Statistiques descriptives de Profit avant traitement :")
print(df["Profit"].describe())

# 5. Suppression des valeurs extrêmes (5e à 95e percentile)
q_low = df["Profit"].quantile(0.05)
q_high = df["Profit"].quantile(0.95)
df = df[(df["Profit"] >= q_low) & (df["Profit"] <= q_high)]

print("\nStatistiques après suppression des outliers :")
print(df["Profit"].describe())

# 6. Nouvelle visualisation
plt.figure(figsize=(8, 5))
sns.histplot(df['Profit'], bins=50, kde=True)
plt.title("Distribution du Profit (Après suppression des outliers)")
plt.show()

# 7. Préparation des données
X = df.drop(columns=["Profit"])
y = df["Profit"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# 8. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Modélisation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    print("RMSE:", np.sqrt(mse))
    print("R² Score:", r2_score(y_test, y_pred))

    # Visualisation des prédictions pour Random Forest
    if name == "Random Forest":
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Valeurs réelles (Profit)")
        plt.ylabel("Valeurs prédites")
        plt.title("Prédictions vs Réel (Random Forest)")
        plt.grid(True)
        plt.show()