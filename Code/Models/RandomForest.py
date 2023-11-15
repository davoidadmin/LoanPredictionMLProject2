import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importa il dataset
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in dataset.columns:
    if dataset[feature].dtype == "object":
        dataset[feature] = pd.Categorical(dataset[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = dataset.drop(["Id", "Risk_Flag"], axis=1)
y = dataset["Risk_Flag"]

# Dividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modello Random Forest con i parametri specificati
class_weights = {0: 1, 1: 3}
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=False,
    class_weight=class_weights,
    random_state=42,
    verbose=True,
)

# Adatta il modello ai dati di addestramento
model.fit(X_train, y_train)

# Effettua previsioni sul test set
y_pred = model.predict(X_test)

# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

# Stampa la matrice di confusione
print(f"Matrice di Confusione:")
print(conf_matrix)

# Calcola il report di classificazione
class_report = classification_report(y_test, y_pred)

# Stampa il report di classificazione
print(f"Report di Classificazione:")
print(class_report)



