# Importa le librerie necessarie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in df.columns:
    if df[feature].dtype == "object":
        df[feature] = pd.Categorical(df[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = df.drop(["Id", "Risk_Flag"], axis=1)
y = df["Risk_Flag"]


# Dividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modello di Support Vector Machine
# Puoi personalizzare il tipo di kernel (linear, rbf, poly) e il parametro C (0.01, 0.1, 1, 10)
model = SVC(kernel='poly', C=10)

# Adatta il modello ai dati di addestramento
model.fit(X_train, y_train)

# Effettua previsioni sul test set
y_pred = model.predict(X_test)

# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza del modello: {accuracy:.2f}')

# Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice di Confusione:")
print(conf_matrix)

# Visualizza un report di classificazione
class_report = classification_report(y_test, y_pred)
print("Report di Classificazione:")
print(class_report)

# Grafico per visualizzare i dati
plt.scatter(X_test[y_pred == 0]['Feature1'], X_test[y_pred == 0]['Feature2'], label='Classe 0', c='b')
plt.scatter(X_test[y_pred == 1]['Feature1'], X_test[y_pred == 1]['Feature2'], label='Classe 1', c='r')
plt.legend()
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Risultati della SVM')
plt.show()
