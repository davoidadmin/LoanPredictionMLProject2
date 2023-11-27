import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense

# Importa il dataset
dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in dataset.columns:
    if dataset[feature].dtype == "object":
        dataset[feature] = pd.Categorical(dataset[feature]).codes

# Dividi il dataset in variabili indipendenti (X) e variabile target (y)
X = dataset.drop(["Id", "Risk_Flag"], axis=1)
y = dataset["Risk_Flag"]

# Dividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calcola i pesi delle classi
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Standardizza le features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funzione per creare il modello Keras
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Imposta il peso della classe minoritaria (1) al doppio di quello della classe maggioritaria (0)
    class_weight = {0: 1, 1: 2}

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], class_weight=class_weight)
    return model

# Crea il modello Keras
model = create_model()

# Allenamento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, class_weight=class_weights_dict)

# Valutazione del modello sul test set
y_pred = model.predict_classes(X_test)
y_pred = y_pred.flatten()

# Calcola l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

# Stampa la matrice di confusione
print(f"Matrice di Confusione:")
print(conf_matrix)

# Calcola il report di classificazione
class_report = classification_report(y_test, y_pred)

# Stampa il report di classificazione
print(f"Report di Classificazione:")
print(class_report)


