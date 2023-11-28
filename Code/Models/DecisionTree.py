import os
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Importa il dataset
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in dataset.columns:
    if dataset[feature].dtype == "object":
        dataset[feature] = pd.Categorical(dataset[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = dataset.drop(["Id", "Risk_Flag"], axis=1)
y = dataset["Risk_Flag"]

# Dividi il dataset in training set, validation set e test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4
)

# Adatta il modello ai dati di addestramento
model.fit(X_train, y_train)

# Effettua previsioni sul validation set
y_val_pred = model.predict(X_val)

# Calcola l'accuratezza del modello sul validation set
accuracy_val = accuracy_score(y_val, y_val_pred)

# Calcola la matrice di confusione sul validation set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)

# Stampa la matrice di confusione sul validation set
print(f"Matrice di Confusione (Validation Set):")
print(conf_matrix_val)

# Calcola il report di classificazione sul validation set
class_report = classification_report(y_val, y_val_pred, output_dict=True)

# Stampa il report di classificazione approssimato
print(f"Report di Classificazione:")
print(f"Accuracy: {round(accuracy_val, 2)}")
print("Precision_0: {:.2f}".format(class_report['0']['precision']))
print("Precision_1: {:.2f}".format(class_report['1']['precision']))
print("Recall_0: {:.2f}".format(class_report['0']['recall']))
print("Recall_1: {:.2f}".format(class_report['1']['recall']))
print("F1_Score_0: {:.2f}".format(class_report['0']['f1-score']))
print("F1_Score_1: {:.2f}".format(class_report['1']['f1-score']))
print("Support_0: {:.2f}".format(class_report['0']['support']))
print("Support_1: {:.2f}".format(class_report['1']['support']))

# Salva i risultati nel file CSV
results_dict = {
    'Model': ['DT'],
    'Accuracy': [round(accuracy_val, 2)],
    'Confusion_Matrix_TP': [conf_matrix_val[0, 0]],
    'Confusion_Matrix_FP': [conf_matrix_val[0, 1]],
    'Confusion_Matrix_FN': [conf_matrix_val[1, 0]],
    'Confusion_Matrix_TN': [conf_matrix_val[1, 1]],
    'Precision_0': [round(class_report['0']['precision'], 2)],
    'Precision_1': [round(class_report['1']['precision'], 2)],
    'Recall_0': [round(class_report['0']['recall'], 2)],
    'Recall_1': [round(class_report['1']['recall'], 2)],
    'F1_Score_0': [round(class_report['0']['f1-score'], 2)],
    'F1_Score_1': [round(class_report['1']['f1-score'], 2)],
    'Support_0': [round(class_report['0']['support'], 2)],
    'Support_1': [round(class_report['1']['support'], 2)]
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
y_scores = model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_scores)
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
