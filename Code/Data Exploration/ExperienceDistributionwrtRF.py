import pandas as pd
import matplotlib.pyplot as plt

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# 1.3: Calcola il range di valori della variabile "Experience"
Experience_range = df["Experience"].max() - df["Experience"].min()

# Calcola i valori di soglia per le tre categorie
low_threshold = Experience_range * 0.25
medium_threshold = Experience_range * 0.75

# Convert bin edges to strings
bin_edges = [df["Experience"].min(), low_threshold, medium_threshold, df["Experience"].max()]

# Crea le categorie
df["Experience_category"] = pd.cut(df["Experience"], bins=bin_edges, labels=["Low", "Medium", "High"])

# Specifica l'ordine delle categorie
category_order = ["Medium", "Low", "High"]
df["Experience_category"] = df["Experience_category"].cat.reorder_categories(category_order, ordered=True)

# Converti la variabile target a stringa
df["Risk_Flag"] = df["Risk_Flag"].astype(str)

# Ordina le categorie nel DataFrame
df["Experience_category"] = df["Experience_category"].cat.as_ordered()

# Crea due dataset separati per "Risk_Flag = 0" e "Risk_Flag = 1"
df_risk_0 = df[df["Risk_Flag"] == "0"]
df_risk_1 = df[df["Risk_Flag"] == "1"]

# Calcola gli istogrammi separati per le due categorie di "Risk_Flag"
hist_risk_0 = df_risk_0["Experience_category"].value_counts()
print(hist_risk_0)
hist_risk_1 = df_risk_1["Experience_category"].value_counts()
print(hist_risk_1)

# Crea un istogramma sovrapposto
width = 0.35  # Larghezza delle barre

x = range(len(bin_edges) - 1)
plt.bar(x, hist_risk_0, width, label="Risk_Flag = 0")
plt.bar([i + width for i in x], hist_risk_1, width, label="Risk_Flag = 1")

# Etichette degli assi
plt.xlabel("Categorie di Esperienza")
plt.ylabel("Numero di osservazioni")

# Etichette sull'asse x
plt.xticks([i + width / 2 for i in x], category_order)  # Usa l'ordine specificato

# Legenda
plt.legend()

# Mostra l'istogramma sovrapposto
plt.show()