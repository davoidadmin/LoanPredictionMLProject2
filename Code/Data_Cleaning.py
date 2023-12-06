import pandas as pd

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Controlla i valori mancanti per tutte le feature
missing_values = df.isnull().sum()
missing_values = missing_values.reset_index(name="Numero di valori mancanti")

# Controlla i valori duplicati
duplicates = df[df.duplicated()]

# Controlla i valori errati e anomali
erroneous_values = []

# Controlla colonne numeriche
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# Se "Risk_Flag" è presente nelle colonne numeriche, escludilo
if "Risk_Flag" in numeric_columns:
    numeric_columns = numeric_columns.drop("Risk_Flag")
numeric_erroneous_values = [{"Feature": col, "Numero di valori errati o anomali": (df[col] < 0).sum()} for col in numeric_columns]
erroneous_values.extend(numeric_erroneous_values)

# Controlla la colonna "Married/Single"
marital_status_values = ["married", "single"]
erroneous_values.append({"Feature": "Married/Single", "Numero di valori errati o anomali": ((df["Married/Single"] != "married") & (df["Married/Single"] != "single")).sum()})

# Controlla la colonna "House_Ownership"
house_ownership_values = ["rented", "owned", "norent_noown"]
erroneous_values.append({"Feature": "House_Ownership", "Numero di valori errati o anomali": (df[~df["House_Ownership"].isin(house_ownership_values)]).shape[0]})

# Controlla la colonna "Car_Ownership"
car_ownership_values = ["yes", "no"]
erroneous_values.append({"Feature": "Car_Ownership", "Numero di valori errati o anomali": (df[~df["Car_Ownership"].isin(car_ownership_values)]).shape[0]})

# Controlla la colonna "Risk_Flag"
risk_flag_values = [0, 1]
erroneous_values.append({"Feature": "Risk_Flag", "Numero di valori errati o anomali": (df[~df["Risk_Flag"].isin(risk_flag_values)]).shape[0]})

# Crea un DataFrame dai risultati
erroneous_df = pd.DataFrame(erroneous_values)

# Stampa i risultati
print("Valori Mancanti:")
print(missing_values)

print("\nValori Duplicati:")
print(duplicates)

print("\nValori Errati o Anomali:")
print(erroneous_df)



# Correzione della colonna 'CITY'
df['CITY'] = df['CITY'].str.extract(r'([a-zA-Z_ ]+)', expand=False).str.strip()
print("\nDataFrame con correzione:")
print(df['CITY'])


# Crea un dizionario per memorizzare le righe che violano i vincoli
violations = {}

# Itera su tutte le righe del dataset
for i in range(df.shape[0]):
    # Controlla che l'esperienza non sia maggiore dell'età
    if df.loc[i, "Experience"] > df.loc[i, "Age"]:
        violations[i] = "Experience > Age"

    # Controlla che l'esperienza non sia maggiore dell'anzianità lavorativa attuale
    if df.loc[i, "Experience"] < df.loc[i, "CURRENT_JOB_YRS"]:
        violations[i] = "Experience < CURRENT_JOB_YRS"

# Converti il dizionario in un dataframe
violations = pd.DataFrame.from_dict(violations, orient="index", columns=["Violation"])

# Visualizza il dataframe
print(violations)

df.to_csv("loan_data_1.csv")