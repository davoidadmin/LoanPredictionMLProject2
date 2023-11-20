import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Importa il dataset originale
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in dataset.columns:
    if dataset[feature].dtype == "object":
        dataset[feature] = pd.Categorical(dataset[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = dataset.drop(["Id", "Risk_Flag"], axis=1)
y = dataset["Risk_Flag"]

# Dividi il dataset in training set e validation set
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# Crea un modello KNN con i parametri specificati
model = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')

# Adatta il modello ai dati di addestramento
model.fit(X_train, y_train)

# Effettua previsioni sul validation set
y_pred = model.predict(X_validation)

# Calcola l'accuratezza del modello sul validation set
accuracy = accuracy_score(y_validation, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_validation, y_pred)

# Stampa la matrice di confusione
print(f"Matrice di Confusione:")
print(conf_matrix)

# Calcola il report di classificazione
class_report = classification_report(y_validation, y_pred, output_dict=True)

# Stampa il report di classificazione
print(f"Report di Classificazione:")
print(class_report)

# Estrai i valori dal classification report
precision_0 = class_report['0']['precision']
precision_1 = class_report['1']['precision']
recall_0 = class_report['0']['recall']
recall_1 = class_report['1']['recall']
f1_score_0 = class_report['0']['f1-score']
f1_score_1 = class_report['1']['f1-score']
support_0 = class_report['0']['support']
support_1 = class_report['1']['support']

# Salva i risultati nel file CSV
results_dict = {
    'Model': ['KNN'],
    'Accuracy': [accuracy],
    'Confusion_Matrix_TP': [conf_matrix[0, 0]],
    'Confusion_Matrix_FP': [conf_matrix[0, 1]],
    'Confusion_Matrix_FN': [conf_matrix[1, 0]],
    'Confusion_Matrix_TN': [conf_matrix[1, 1]],
    'Precision_0': [precision_0],
    'Precision_1': [precision_1],
    'Recall_0': [recall_0],
    'Recall_1': [recall_1],
    'F1_Score_0': [f1_score_0],
    'F1_Score_1': [f1_score_1],
    'Support_0': [support_0],
    'Support_1': [support_1]
}

# Aggiungi i risultati al DataFrame
results_df = pd.DataFrame(results_dict)

# Se il file non esiste, aggiungi l'intestazione
if not os.path.isfile('results.csv'):
    results_df.to_csv('results.csv', index=False)
else:
    # Se il file esiste, aggiungi i risultati in append
    results_df.to_csv('results.csv', mode='a', header=False, index=False)

# Calcola la curva ROC e l'area sotto la curva
y_scores = model.predict_proba(X_validation)[:, 1]
fpr, tpr, _ = roc_curve(y_validation, y_scores)
roc_auc = auc(fpr, tpr)

# Visualizza la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

