import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar dados
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Dicionário com os modelos
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Árvore de Decisão": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Rede Neural (MLP)": MLPClassifier(max_iter=500)
}

# Abrir arquivo de saída
with open("comparativo_modelos.txt", "w") as f:
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        f.write(f"{'='*60}\n")
        f.write(f"Modelo: {nome}\n")
        f.write(f"{'-'*60}\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\n")
