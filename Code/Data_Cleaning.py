import pandas as pd

# Importa il dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Conta i valori mancanti per tutte le feature
#missing_values = df.isnull().sum()

# Crea una tabella con i risultati
#missing_values = missing_values.reset_index(name="Numero di valori mancanti")

# Stampa la tabella
#print(missing_values)

# NB : Check the statistical summary of numerical columns!!!

# # Crea un dizionario per memorizzare i conteggi
# profession_counts = {}
#
# # Itera su tutti i valori della feature "Profession"
# for profession in df["Profession"].unique():
#     # Conta le righe con il valore specificato
#     profession_counts[profession] = df[df["Profession"] == profession].shape[0]
#
# # Converti il dizionario in un dataframe
# profession_counts = pd.DataFrame.from_dict(profession_counts, orient="index", columns=["Count"])
#
# # Conta il numero di voci nel dizionario
# number_of_entries = len(profession_counts)

# # Visualizza il dataframe
# print(profession_counts)
#
# # Stampa il numero di voci
# print(number_of_entries)

# Correzione della colonna 'CITY'
# df['CITY'] = df['CITY'].str.extract(r'([a-zA-Z_ ]+)', expand=False).str.strip()
# print("\nDataFrame con correzione:")
# print(df['CITY'])


# # Crea un dizionario per memorizzare le righe che violano i vincoli
# violations = {}
#
# # Itera su tutte le righe del dataset
# for i in range(df.shape[0]):
#     # Controlla che l'esperienza non sia maggiore dell'età
#     if df.loc[i, "Experience"] > df.loc[i, "Age"]:
#         violations[i] = "Experience > Age"
#
#     # Controlla che l'esperienza non sia maggiore dell'anzianità lavorativa attuale
#     if df.loc[i, "Experience"] < df.loc[i, "CURRENT_JOB_YRS"]:
#         violations[i] = "Experience < CURRENT_JOB_YRS"

# # Converti il dizionario in un dataframe
# violations = pd.DataFrame.from_dict(violations, orient="index", columns=["Violation"])
#
# # Visualizza il dataframe
# print(violations)

# # Elimina la feature "Id"
# df = df.drop('Id', axis=1)
#
# # Salva il nuovo dataset
# df.to_csv("loan_data_1.csv")