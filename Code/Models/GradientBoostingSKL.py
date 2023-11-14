import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Crea un modello di Gradient Boosting
model = GradientBoostingClassifier()

# Definisci la griglia degli iperparametri da esplorare
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2]
    # Altri parametri da includere nella griglia
}

# Crea l'oggetto GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Esegui la ricerca della griglia sull'intero spazio degli iperparametri
grid_search.fit(X_train, y_train)

# Visualizza i migliori iperparametri
print("Migliori iperparametri:", grid_search.best_params_)

# Valuta le prestazioni del modello con i migliori iperparametri sul test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

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


