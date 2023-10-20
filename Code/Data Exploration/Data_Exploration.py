import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\loan_data_1.csv")

# Stampa la dimensione del dataset
print("Dimensione del dataset:", df.shape)

# Stampa il numero di feature
print("Numero di feature:", df.shape[1])

# Stampa il tipo di dati di ciascuna feature
 for feature in df.columns:
     print("Tipo di dati di", feature, ":", df[feature].dtype)




# 2. FEATURE CATEGORICHE CON POCHI VALORI
# 3. FEATURE CATEGORICHE CON MOLTI VALORI


# # Valutare la presenza di valori anomali (test di Grubbs)
# outliers = df["target"].loc[df["target"].between(df["target"].quantile(0.05), df["target"].quantile(0.95))]
# print(outliers)