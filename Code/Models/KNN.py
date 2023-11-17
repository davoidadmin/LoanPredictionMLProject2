import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importa il dataset originale
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in dataset.columns:
    if dataset[feature].dtype == "object":
        dataset[feature] = pd.Categorical(dataset[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = dataset.drop(["Id", "Risk_Flag"], axis=1)
y = dataset["Risk_Flag"]

# Dividi il dataset in training set e test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Dividi il set temporaneo in set di validazione e test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crea un modello KNN
knn = KNeighborsClassifier()

# Addestra il modello KNN con i pesi personalizzati
knn.fit(X_train, y_train)

# Definisci la griglia degli iperparametri da esplorare
param_grid = {
    'n_neighbors': [3, 6, 9],
    'weights': ['distance'],
    'p': [2],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
}

# Crea l'oggetto GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=3)

# Esegui la ricerca della griglia sull'intero spazio degli iperparametri
grid_search.fit(X_train, y_train)

# Visualizza i migliori iperparametri
print("Migliori iperparametri:", grid_search.best_params_)

# Valuta le prestazioni del modello con i migliori iperparametri sul set di validazione
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred)
print("Accuratezza sul set di validazione:", accuracy_val)

# Valuta le prestazioni del modello con i migliori iperparametri sul test set
y_pred_test = best_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuratezza sul test set:", accuracy_test)




