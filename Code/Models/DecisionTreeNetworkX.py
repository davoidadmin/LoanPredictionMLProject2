import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Importa il dataset
original_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")
for feature in original_dataset.columns:
    if original_dataset[feature].dtype == "object":
        original_dataset[feature] = pd.Categorical(original_dataset[feature]).codes

z_score_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_zscore.csv")
for feature in z_score_dataset.columns:
    if z_score_dataset[feature].dtype == "object":
        z_score_dataset[feature] = pd.Categorical(z_score_dataset[feature]).codes

minmax_dataset = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data_minmax.csv")
for feature in minmax_dataset.columns:
    if minmax_dataset[feature].dtype == "object":
        minmax_dataset[feature] = pd.Categorical(minmax_dataset[feature]).codes

# Elenco dei nomi dei dataset
dataset_names = ["Original", "Z-Score Normalized", "Min-Max Normalized"]

# Elenco dei dataset
datasets = [original_dataset, z_score_dataset, minmax_dataset]

# Ciclo attraverso i dataset
for dataset_name, dataset in zip(dataset_names, datasets):
    # Dividi il dataset in variabili indipendenti (X) e variabile target (y)
    X = dataset.drop(["Id", "Risk_Flag", "CITY", "STATE"], axis=1)
    y = dataset["Risk_Flag"]

    # Dividi il dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea un modello di Decision Tree
    model = DecisionTreeClassifier()

    # Adatta il modello ai dati di addestramento
    model.fit(X_train, y_train)

    # Creare un grafico NetworkX per visualizzare l'albero decisionale
    G = nx.DiGraph()

    def build_tree(graph, parent_name, tree, feature_names, class_names):
        # Aggiungi il nodo corrente al grafico
        graph.add_node(parent_name, label=str(tree.value))

        # Estrai le informazioni sull'albero
        left_child = tree.children_left[parent_name]
        right_child = tree.children_right[parent_name]

        # Aggiungi rami sinistro e destro ricorsivamente
        if left_child != right_child:  # non è un nodo foglia
            feature_name = feature_names[tree.feature[parent_name]]
            threshold = tree.threshold[parent_name]
            graph.add_edge(parent_name, left_child, label=f'{feature_name} <= {threshold}')
            build_tree(graph, left_child, tree, feature_names, class_names)

            graph.add_edge(parent_name, right_child, label=f'{feature_name} > {threshold}')
            build_tree(graph, right_child, tree, feature_names, class_names)
        else:  # è un nodo foglia
            class_index = tree.value[parent_name].argmax()
            class_label = class_names[class_index]
            graph.add_edge(parent_name, f'{class_label}')

    build_tree(G, 0, model.tree_, X.columns, ['0', '1'])

    # Visualizza il grafico NetworkX
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, font_size=8, node_size=700, node_color='skyblue', font_color='black', font_weight='bold', edge_color='gray', linewidths=0.5)
    plt.title(f'Albero Decisionale nel Decision Tree ({dataset_name})')
    plt.show()


