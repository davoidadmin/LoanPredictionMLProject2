import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# CALCOLO INDICI DI CORRELAZIONE E VISUALIZZAZIONE MATRICE DI CORRELAZIONE
# Itera su tutte le feature
for feature in df.columns:
     # Controlla il tipo di dati della feature
     if df[feature].dtype == "object":
         # La feature è di tipo categorico/stringa
         # Converti i valori in numeri
         df[feature] = pd.Categorical(df[feature]).codes
# Calcola la matrice di correlazione
corr_matrix = df.corr()

# Visualizza la matrice di correlazione
print(corr_matrix)
#
# # Visualizza la matrice di correlazione fra le feature di tutto il dataset tramite un grafico a matrice di calore
sns.heatmap(corr_matrix, cmap="RdBu", vmax=1, vmin=-1, annot=True, fmt=".2f", linewidths=0.2, square=True,
            xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
#
# # Aggiungi un titolo al grafico
plt.title("Matrice di correlazione")
#
# # Modifica la dimensione del grafico
plt.figure(figsize=(10, 10))
plt.show()