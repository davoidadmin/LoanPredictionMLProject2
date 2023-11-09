import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Importa i dataset
original_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in original_dataset.columns:
    if original_dataset[feature].dtype == "object":
        original_dataset[feature] = pd.Categorical(original_dataset[feature]).codes

z_score_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_zscore.csv")
for feature in z_score_dataset.columns:
    if z_score_dataset[feature].dtype == "object":
        z_score_dataset[feature] = pd.Categorical(z_score_dataset[feature]).codes

minmax_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_minmax.csv")
for feature in minmax_dataset.columns:
    if minmax_dataset[feature].dtype == "object":
        minmax_dataset[feature] = pd.Categorical(minmax_dataset[feature]).codes

# Elenco dei nomi dei dataset
dataset_names = ["Original", "Z-Score Normalized", "Min-Max Normalized"]

# Elenco dei dataset
datasets = [original_dataset, z_score_dataset, minmax_dataset]

accuracies = []

# Ciclo attraverso i dataset
for dataset_name, dataset in zip(dataset_names, datasets):
    # Dividi il dataset in variabili indipendenti (X) e variabile target (y)
    X = dataset.drop(["Id", "Risk_Flag", "CITY", "STATE"], axis=1)
    y = dataset["Risk_Flag"]

    # Dividi il dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea un modello di Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Puoi personalizzare il numero di alberi (n_estimators) e altri parametri

    # Adatta il modello ai dati di addestramento
    model.fit(X_train, y_train)

    # Effettua previsioni sul test set
    y_pred = model.predict(X_test)

    # Calcola l'accuratezza del modello
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calcola la matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Stampa la matrice di confusione
    print(f"Matrice di Confusione per {dataset_name}:")
    print(conf_matrix)

    # Calcola il report di classificazione
    class_report = classification_report(y_test, y_pred)

    # Stampa il report di classificazione
    print(f"Report di Classificazione per {dataset_name}:")
    print(class_report)

    # Calcola l'importanza delle feature
    feature_importance = model.feature_importances_
    feature_names = X.columns

    # Grafico per visualizzare l'importanza delle feature
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel('Importanza delle Feature')
    plt.title(f'Importanza delle Feature nel Decision Tree ({dataset_name})')
    plt.gca().invert_yaxis()
    plt.show()

    # Grafico per visualizzare l'albero decisionale
    plt.figure(figsize=(12, 6))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f'Albero Decisionale nel Decision Tree ({dataset_name})')
    plt.show()

# Crea un grafico a barre per confrontare le accuratezze
plt.figure(figsize=(8, 6))
plt.bar(dataset_names, accuracies, color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('Accuratezza')
plt.title('Confronto delle Accuratezze tra i Diversi Datasets')
plt.show()
