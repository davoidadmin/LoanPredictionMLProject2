import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Carica il modello
model = joblib.load('C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\Code\\Models\\knn.joblib')

# Carica i dati di convalida
validation_data = joblib.load('validation_data.joblib')
X_validation = validation_data['X_validation']
y_validation = validation_data['y_validation']

# Ottieni i punteggi di decisione
y_scores = model.predict_proba(X_validation)[:, 1]

# Calcola la curva ROC
fpr, tpr, thresholds = roc_curve(y_validation, y_scores)
roc_auc = auc(fpr, tpr)

# Visualizza la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()




