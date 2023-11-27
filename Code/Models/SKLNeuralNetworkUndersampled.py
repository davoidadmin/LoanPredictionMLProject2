import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

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

# Applica l'undersampling alla classe sovrarappresentata ('Risk_Flag'=0) sia sul training che sul validation set
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
X_val_resampled, y_val_resampled = undersampler.fit_resample(X_val, y_val)

# Standardizza le features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_val_resampled = scaler.transform(X_val_resampled)
X_test = scaler.transform(X_test)

# Definisci il modello di rete neurale con i parametri specificati
model = MLPClassifier(
    alpha=0.01,
    hidden_layer_sizes=(128, 64, 32),
    max_iter=200,
    random_state=42,
    activation='tanh',
    batch_size=64,
    learning_rate='adaptive',
    solver='lbfgs'
)

# Adatta il modello ai dati di addestramento
model.fit(X_train_resampled, y_train_resampled)

# Effettua previsioni sul validation set
y_pred = model.predict(X_val_resampled)

# Calcola l'accuratezza del modello sul validation set
accuracy = accuracy_score(y_val_resampled, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_val_resampled, y_pred)

# Stampa la matrice di confusione
print(f"Matrice di Confusione:")
print(conf_matrix)

# Calcola il report di classificazione
class_report = classification_report(y_val_resampled, y_pred, output_dict=True)

# Stampa il report di classificazione approssimato
print(f"Report di Classificazione:")
print(f"Accuracy: {round(accuracy, 2)}")
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
    'Model': ['MPL_UnderSampled'],
    'Accuracy': [round(accuracy, 2)],
    'Confusion_Matrix_TP': [conf_matrix[0, 0]],
    'Confusion_Matrix_FP': [conf_matrix[0, 1]],
    'Confusion_Matrix_FN': [conf_matrix[1, 0]],
    'Confusion_Matrix_TN': [conf_matrix[1, 1]],
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
y_scores = model.predict_proba(X_val_resampled)[:, 1]
fpr, tpr, _ = roc_curve(y_val_resampled, y_scores)
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
