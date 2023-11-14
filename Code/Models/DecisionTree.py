import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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

# Crea un modello Decision Tree
model = DecisionTreeClassifier()

# Definisci la griglia degli iperparametri da esplorare
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
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

