import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file CSV
df = pd.read_csv('C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\Code\\Models\\results.csv')

# Aggiungi una colonna per indicare se il modello è undersampled o no
df['Undersampled'] = df['Model'].str.contains('UnderSampled')

# Visualizza la tabella con gli indici di tutti i modelli
print("Tabella con indici di tutti i modelli:")
print(df)

# Visualizza un grafico ad istogramma per confrontare le accuracy dei vari modelli
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', hue='Undersampled', data=df)
plt.title('Confronto Accuracy dei Modelli')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare le precision dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Precision_0', hue='Undersampled', data=df, palette='Blues')
sns.barplot(x='Model', y='Precision_1', hue='Undersampled', data=df, palette='Oranges')
plt.title('Confronto Precision dei Modelli')
plt.ylabel('Precision')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare le recall dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Recall_0', hue='Undersampled', data=df, palette='Blues')
sns.barplot(x='Model', y='Recall_1', hue='Undersampled', data=df, palette='Oranges')
plt.title('Confronto Recall dei Modelli')
plt.ylabel('Recall')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualizza un grafico ad istogramma per confrontare gli F1-score dei vari modelli
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1_Score_0', hue='Undersampled', data=df, palette='Blues')
sns.barplot(x='Model', y='F1_Score_1', hue='Undersampled', data=df, palette='Oranges')
plt.title('Confronto F1-score dei Modelli')
plt.ylabel('F1-score')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualizza grafici ad istogramma per confrontare i valori della matrice di confusione di ogni modello
confusion_matrix_cols = ['Confusion_Matrix_TP', 'Confusion_Matrix_FP', 'Confusion_Matrix_FN', 'Confusion_Matrix_TN']
for col in confusion_matrix_cols:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=col, hue='Undersampled', data=df)
    plt.title(f'Confronto {col} dei Modelli')
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.show()



