import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Itera su tutte le feature
for feature in df.columns:
     # Controlla il tipo di dati della feature
     if df[feature].dtype == "object":
         # La feature è di tipo categorico/stringa
         # Converti i valori in numeri
         df[feature] = pd.Categorical(df[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = df.drop(["Id", "Risk_Flag"], axis=1)  # Elimina variabile target
y = df["Risk_Flag"]

# Esegui la PCA
pca = PCA(n_components=None)

# Adatta il modello ai dati
pca.fit(X)

# Calcola l'importanza delle variabili
explained_variance_ratio = pca.explained_variance_ratio_

# Stampa le variabili selezionate
print("Variance Ratio delle Caratteristiche :", explained_variance_ratio)

# Plot dei coefficienti
plt.figure(figsize=(10, 6))
sns.barplot(x=explained_variance_ratio, y=X.columns)
plt.title("Variance Ratio delle Caratteristiche")
plt.xlabel("Variance Ratio")
plt.ylabel("Caratteristiche")
plt.show()