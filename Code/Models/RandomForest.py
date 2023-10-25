import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in df.columns:
    if df[feature].dtype == "object":
        df[feature] = pd.Categorical(df[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = df.drop(["Id", "Risk_Flag"], axis=1)
y = df["Risk_Flag"]

# Dividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modello di Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Puoi personalizzare il numero di alberi (n_estimators) e altri parametri

# Adatta il modello ai dati di addestramento
model.fit(X_train, y_train)

# Effettua previsioni sul test set
y_pred = model.predict(X_test)

# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza del modello: {accuracy:.2f}')

# Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice di Confusione:")
print(conf_matrix)

# Visualizza un report di classificazione
class_report = classification_report(y_test, y_pred)
print("Report di Classificazione:")
print(class_report)

# Visualizzazione dell'importanza delle feature
feature_importance = model.feature_importances_
feature_names = X.columns

# Grafico per visualizzare l'importanza delle feature
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Importanza delle Feature')
plt.title('Importanza delle Feature nel Random Forest')
plt.gca().invert_yaxis()
plt.show()
