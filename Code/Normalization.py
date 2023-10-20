import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Importa il dataset
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Elimina la colonna "Id"
dataset = dataset.drop("Id", axis=1)

# Codifica le variabili categoriche in variabili binarie (one-hot encoding)
categorical_columns = ["Married/Single"]  # Aggiungi altre colonne categoriche se necessario
dataset = pd.get_dummies(dataset, columns=categorical_columns)

# Crea un dizionario per memorizzare le variabili da normalizzare
variables_to_normalize = {
    column: type(dataset[column][0]) for column in dataset.columns
    if not dataset[column].dtype.name.startswith("object") and not dataset[column].dtype.name == "category"
}

# Applica la normalizzazione z-score
dataset_zscore = dataset.copy()
for column, type_ in variables_to_normalize.items():
    if type_ in [np.float64, np.int64]:
        dataset_zscore[column] = (dataset_zscore[column] - dataset_zscore[column].mean()) / dataset_zscore[column].std()

# Applica la normalizzazione min-max
dataset_minmax = dataset.copy()
for column, type_ in variables_to_normalize.items():
    if type_ in [np.float64, np.int64]:
        dataset_minmax[column] = (dataset_minmax[column] - dataset_minmax[column].min()) / (dataset_minmax[column].max() - dataset_minmax[column].min())

# Dividi il dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset["Risk_Flag"], test_size=0.25)
X_train_zscore, X_test_zscore, y_train_zscore, y_test_zscore = train_test_split(dataset_zscore, dataset_zscore["Risk_Flag"], test_size=0.25)
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(dataset_minmax, dataset_minmax["Risk_Flag"], test_size=0.25)

# Crea un modello di classificazione
model = LogisticRegression()

# Adatta il modello al dataset di training (z-score)
model.fit(X_train_zscore.drop("Risk_Flag", axis=1), y_train_zscore)

# Valuta il modello sul dataset di test (z-score)
accuracy_zscore = model.score(X_test_zscore.drop("Risk_Flag", axis=1), y_test_zscore)

# Adatta il modello al dataset di training (min-max)
model.fit(X_train_minmax.drop("Risk_Flag", axis=1), y_train_minmax)

# Valuta il modello sul dataset di test (min-max)
accuracy_minmax = model.score(X_test_minmax.drop("Risk_Flag", axis=1), y_test_minmax)

# Stampa le accuratezze
print("Accuratezza con normalizzazione z-score:", accuracy_zscore)
print("Accuratezza con normalizzazione min-max:", accuracy_minmax)
