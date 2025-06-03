
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Criar diretório de imagens, se não existir
os.makedirs("img", exist_ok=True)

# Carregar dados
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Dicionário com os modelos
modelos = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "knn": KNeighborsClassifier(),
    "svm": SVC(),
    "neural_network": MLPClassifier(max_iter=500)
}

# Arquivo de saída de texto
with open("comparativo_modelos.txt", "w") as f:
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Salvar imagem da matriz de confusão
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusão - {nome.replace('_', ' ').title()}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        img_path = f"img/{nome}_matrizconfusao.png"
        plt.savefig(img_path)
        plt.close()

        # Escrever resultados no arquivo
        f.write(f"{'='*60}\n")
        f.write(f"Modelo: {nome.replace('_', ' ').title()}\n")
        f.write(f"{'-'*60}\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(cm) + "\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\n")
