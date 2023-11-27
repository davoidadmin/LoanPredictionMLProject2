import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import shap

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Itera su tutte le feature
for feature in df.columns:
    # Controlla il tipo di dati della feature
    if df[feature].dtype == "object":
        # La feature è di tipo categorico/stringa
        # Converti i valori in numeri
        df[feature] = pd.Categorical(df[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = df.drop(["Id", "Risk_Flag"], axis=1)  # Elimina variabile target
y = df["Risk_Flag"]

# Cattura i nomi delle caratteristiche
feature_names = X.columns

# Crea un modello per la selezione delle caratteristiche
model = LogisticRegression()  # Puoi sostituire con il tuo modello di machine learning preferito

# Crea un selettore RFE
rfe = RFE(model, n_features_to_select=11)

# Adatta il selettore RFE ai dati
rfe.fit(X, y)

# Adatta il modello di regressione logistica alle caratteristiche selezionate
model.fit(X[X.columns[rfe.support_]], y)

# Calcola l'importanza dei coefficienti
coeff_importance = abs(model.coef_[0])

# Calcola gli Shapley values
# Crea un oggetto LinearExplainer di SHAP utilizzando il modello addestrato
explainer = shap.LinearExplainer(model, X[X.columns[rfe.support_]])
shap_values = explainer.shap_values(X[X.columns[rfe.support_]])
shap_importance = abs(shap_values).mean(axis=0)

# Stampa le caratteristiche selezionate per ciascun metodo
print("Caratteristiche selezionate tramite RFE:", X.columns[rfe.support_])
print("Importanza dei coefficienti:", coeff_importance)
print("Importanza degli Shapley values:", shap_importance)

# Salvataggio degli indici di importanza in un DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns[rfe.support_],
    'Coefficient_Importance': coeff_importance,
    'Shapley_Importance': shap_importance
})

# Stampa la tabella e salva in un file CSV
print("\nTabella di Importanza delle Caratteristiche:")
print(feature_importance_df)
feature_importance_df.to_csv('feature_importance.csv', index=False)

# Plot dei coefficienti
plt.figure(figsize=(10, 6))
sns.barplot(x=coeff_importance, y=feature_names[rfe.support_])
plt.title("Importanza dei Coefficienti")
plt.xlabel("Importanza")
plt.ylabel("Caratteristiche")
plt.show()

# Plot degli Shapley values
plt.figure(figsize=(10, 6))
sns.barplot(x=shap_importance, y=feature_names[rfe.support_])
plt.title("Importanza degli Shapley Values")
plt.xlabel("Importanza")
plt.ylabel("Caratteristiche")
plt.show()
