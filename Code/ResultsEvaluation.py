import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file CSV
df = pd.read_csv('C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\Code\\Models\\results.csv')

# Visualizza la tabella con gli indici di tutti i modelli
print("Tabella con indici di tutti i modelli:")
print(df)

# Visualizza un grafico ad istogramma per confrontare le accuracy dei vari modelli
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=df)
plt.title('Confronto Accuracy dei Modelli')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare le precision dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Precision_0', data=df, color='blue', label='Precision_0')
sns.barplot(x='Model', y='Precision_1', data=df, color='orange', label='Precision_1')
plt.title('Confronto Precision dei Modelli')
plt.ylabel('Precision')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare le recall dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Recall_0', data=df, color='blue', label='Recall_0')
sns.barplot(x='Model', y='Recall_1', data=df, color='orange', label='Recall_1')
plt.title('Confronto Recall dei Modelli')
plt.ylabel('Recall')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare gli F1-score dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1_Score_0', data=df, color='blue', label='F1_Score_0')
sns.barplot(x='Model', y='F1_Score_1', data=df, color='orange', label='F1_Score_1')
plt.title('Confronto F1-score dei Modelli')
plt.ylabel('F1-score')
plt.legend()
plt.xticks(rotation=45)
plt.show()

