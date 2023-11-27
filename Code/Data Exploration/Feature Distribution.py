import pandas as pd
import matplotlib.pyplot as plt

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

#   VISUALIZZAZIONE DISTRIBUZIONE SINGOLE FEATURE (NB: NON CICLARE, STAMPA OGNI FEATURE CON GRAFICO AD HOC)
#  Itera su tutte le feature
for feature in df.columns:
     # Controlla il tipo di dati della feature
     if df[feature].dtype == "object":
         # La feature è di tipo categorico/stringa
         # Visualizza la distribuzione dei valori
         plt.hist(df[feature])
         plt.xlabel(feature)
         plt.ylabel("Frequenza")
         plt.show()
     else:
          # La feature è di tipo numerico
          # Visualizza un grafico con media, deviazione standard, minimo e massimo
          if df[feature].name == "Risk_Flag":
              plt.hist(df[feature])
              plt.xlabel(feature)
              plt.ylabel("Frequenza")
              plt.show()
          else:
              plt.boxplot(df[feature])
              plt.xlabel(feature)
              plt.ylabel("Valore")
              plt.show()