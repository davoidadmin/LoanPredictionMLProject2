import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\loan_data_1.csv")

# 1.2: Calcola il range di valori della variabile "Age"
# Imposta manualmente le soglie per le categorie
age_thresholds = [20, 39, 59, 79]

# Definisci i nomi delle categorie
age_categories = ["Young", "Middle-aged", "Elderly"]

# Crea le categorie e specifica l'ordine
category_order = ["Young", "Middle-aged", "Elderly"]
df["Age_category"] = pd.cut(df["Age"], bins=age_thresholds, labels=age_categories, ordered=True)

# Converti la variabile target a stringa
df["Risk_Flag"] = df["Risk_Flag"].astype(str)

# Crea due dataset separati per "Risk_Flag = 0" e "Risk_Flag = 1"
df_risk_0 = df[df["Risk_Flag"] == "0"]
df_risk_1 = df[df["Risk_Flag"] == "1"]

# Calcola gli istogrammi separati per le due categorie di "Risk_Flag"
hist_risk_0 = df_risk_0["Age_category"].value_counts().reindex(category_order, fill_value=0)
print(hist_risk_0)
hist_risk_1 = df_risk_1["Age_category"].value_counts().reindex(category_order, fill_value=0)
print(hist_risk_1)

# Crea un istogramma sovrapposto
width = 0.35  # Larghezza delle barre

x = range(len(age_thresholds) - 1)
plt.bar(x, hist_risk_0, width, label="Risk_Flag = 0")
plt.bar([i + width for i in x], hist_risk_1, width, label="Risk_Flag = 1")

# Etichette degli assi
plt.xlabel("Categorie di età")
plt.ylabel("Numero di osservazioni")

# Etichette sull'asse x
plt.xticks([i + width / 2 for i in x], category_order)

# Legenda
plt.legend()

# Mostra l'istogramma sovrapposto
plt.show()