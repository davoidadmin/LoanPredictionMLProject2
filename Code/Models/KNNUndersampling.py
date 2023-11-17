import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

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

# Applica l'undersampling alla classe sovrarappresentata ('Risk_Flag'=0)
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Crea un modello KNN con i parametri specificati
model = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')

# Adatta il modello ai dati di addestramento undersampled
model.fit(X_resampled, y_resampled)

# Effettua previsioni sul test set
y_pred = model.predict(X_temp)

# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_temp, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_temp, y_pred)

# Stampa la matrice di confusione
print(f"Matrice di Confusione:")
print(conf_matrix)

# Calcola il report di classificazione
class_report = classification_report(y_temp, y_pred)

# Stampa il report di classificazione
print(f"Report di Classificazione:")
print(class_report)
