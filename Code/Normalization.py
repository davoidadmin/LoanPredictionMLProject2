import numpy as np
import pandas as pd

# Importa il dataset
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Mantieni le colonne specificate
columns_to_keep = ["Id", "Profession", "CITY", "STATE", "Risk_Flag"]
original_dataset = dataset[columns_to_keep]

# Rimuovi le colonne specificate dal dataset per la normalizzazione
columns_to_normalize = dataset.columns.difference(columns_to_keep)
dataset_for_normalization = dataset[columns_to_normalize]

# Crea un dizionario per memorizzare le variabili da normalizzare
variables_to_normalize = {
    column: type(dataset_for_normalization[column][0]) for column in dataset_for_normalization.columns
    if not dataset_for_normalization[column].dtype.name.startswith("object")
}

# Applica la normalizzazione z-score
dataset_zscore = dataset_for_normalization.copy()
for column, type_ in variables_to_normalize.items():
    if type_ in [np.float64, np.int64]:
        dataset_zscore[column] = (dataset_zscore[column] - dataset_zscore[column].mean()) / dataset_zscore[column].std()

# Applica la normalizzazione min-max
dataset_minmax = dataset_for_normalization.copy()
for column, type_ in variables_to_normalize.items():
    if type_ in [np.float64, np.int64]:
        dataset_minmax[column] = (dataset_minmax[column] - dataset_minmax[column].min()) / (dataset_minmax[column].max() - dataset_minmax[column].min())

# Crea il nuovo dataset normalizzato con le colonne nell'ordine desiderato
columns_ordered = dataset.columns.tolist()  # Ordina le colonne come desiderato
normalized_dataset_zscore = pd.concat([original_dataset, dataset_zscore], axis=1)
normalized_dataset_zscore = normalized_dataset_zscore[columns_ordered]

normalized_dataset_minmax = pd.concat([original_dataset, dataset_minmax], axis=1)
normalized_dataset_minmax = normalized_dataset_minmax[columns_ordered]

# Salva i dataset normalizzati in file separati
normalized_dataset_zscore.to_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_zscore.csv", index=False)
normalized_dataset_minmax.to_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_minmax.csv", index=False)
