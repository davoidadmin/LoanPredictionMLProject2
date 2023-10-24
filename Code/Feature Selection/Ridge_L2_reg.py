import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carica il dataset e codifica le variabili categoriche
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in df.columns:
    if df[feature].dtype == "object":
        df[feature] = pd.Categorical(df[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = df.drop(["Id", "Risk_Flag"], axis=1)
y = df["Risk_Flag"]

# Suddividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizza i dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crea un modello di regressione Ridge e adattalo ai dati
alpha = 1  # Puoi regolare alpha a tuo piacimento
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_scaled, y_train)

# Valuta il modello sul test set
y_pred = ridge_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualizza i coefficienti della regressione Ridge
coefficients = ridge_model.coef_
feature_names = X.columns

# Rappresentazione grafica dei coefficienti
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xticks(rotation=90)
plt.title("Coefficients of Ridge Regression")
plt.xlabel("Features")
plt.ylabel("Coefficient Values")
plt.show()
