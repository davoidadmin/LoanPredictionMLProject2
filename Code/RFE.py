import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression  # Puoi sostituire con il tuo modello di machine learning preferito
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
X = df.drop("Risk_Flag", axis=1)  # Elimina variabile target
y = df["Risk_Flag"]

# Crea un modello per la selezione delle caratteristiche
model = LogisticRegression()  # Puoi sostituire con il tuo modello di machine learning preferito

# Crea un selettore RFE
rfe = RFE(model, n_features_to_select = None)

# Adatta il selettore RFE ai dati
rfe.fit(X, y)

# Ottieni le caratteristiche selezionate
selected_features = X.columns[rfe.support_]

# Adatta il modello di regressione logistica alle caratteristiche selezionate
model.fit(X[selected_features], y)

# Calcola l'importanza dei coefficienti
coeff_importance = abs(rfe.estimator_.coef_)[0]

# Calcola l'importanza degli errori
#error_importance = rfe.estimator_.predict_proba(X)[:, 1]

# Definisci la funzione di previsione per SHAP
def predict_with_model(X):
    return model.predict_proba(X)[:, 1]

# Calcola gli Shapley values
# Crea un oggetto LinearExplainer di SHAP
explainer = shap.Explainer(predict_with_model, X)
shap_values = explainer.shap_values(X)
shap_importance = abs(shap_values).mean(axis=0)

# Stampa le caratteristiche selezionate per ciascun metodo
print("Caratteristiche selezionate tramite RFE:", selected_features)
print("Importanza dei coefficienti:", coeff_importance)
print("Importanza degli Shapley values:", shap_importance)