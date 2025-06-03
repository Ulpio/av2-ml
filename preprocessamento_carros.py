import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_excel("p40.xlsx", sheet_name="2005 CAR DATA")

# Remover colunas extras além da 12ª (as últimas têm metadados irrelevantes)
df = df.iloc[:, :12]

# Remover linhas com valores ausentes
df = df.dropna()

# Visualizar informações básicas
print("Resumo das colunas:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# Codificação de variáveis categóricas
df_encoded = pd.get_dummies(df, columns=["Make", "Model", "Trim", "Type"], drop_first=True)

# Separar variáveis independentes (X) e alvo (y)
X = df_encoded.drop("Leather", axis=1)
y = df_encoded["Leather"]

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar os dados (normalização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter os arrays escalados de volta para DataFrames
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Salvar os conjuntos processados em CSVs
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPré-processamento concluído com sucesso.")
