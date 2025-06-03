
# 📊 Classificação de Dados de Carros Usados (2005)

Este projeto aplica diversos algoritmos de aprendizado de máquina supervisionado com o objetivo de prever se um carro possui **banco de couro** (`Leather`) a partir de suas características técnicas e de mercado.

---

## 📁 Estrutura do Projeto

- `preprocessamento_carros.py`: Realiza a limpeza, codificação e separação dos dados em treino/teste.
- `regressao_logistica.py`: Regressão logística.
- `arvore_decisao.py`: Árvore de decisão.
- `rand_forest.py`: Random Forest.
- `knn.py`: K-Nearest Neighbors.
- `svm.py`: Support Vector Machine.
- `rede_neural.py`: Rede Neural (MLPClassifier).
- `comparador_final.py`: Executa todos os modelos e gera um relatório comparativo.
- `comparativo_modelos.txt`: Resultado consolidado das avaliações.

---

## 🧠 Explicação dos Modelos

### 1. Regressão Logística
Modelo linear para classificação. Estima a probabilidade de um evento (ex: carro ter couro) com base em uma função logística (sigmoide).

- Ideal para relações lineares.
- Rápido e interpretável.

### 2. Árvore de Decisão
Estrutura hierárquica que divide os dados em decisões binárias até alcançar uma classificação.

- Fácil de interpretar.
- Pode overfitar se não for podada.

### 3. Random Forest
Conjunto de árvores de decisão treinadas com amostras diferentes. O resultado é a média (ou voto) das previsões de cada árvore.

- Reduz overfitting.
- Ótimo para dados tabulares com variáveis categóricas.

### 4. K-Nearest Neighbors (KNN)
Classifica um item com base nos rótulos dos seus vizinhos mais próximos no espaço de características.

- Simples e eficaz.
- Requer normalização para boas performances.

### 5. Support Vector Machine (SVM)
Encontra o melhor hiperplano que separa as classes com a maior margem possível.

- Pode ser poderoso com kernels não-lineares.
- Escala mal com grandes datasets.

### 6. Rede Neural (MLP)
Perceptron multicamadas capaz de capturar padrões não-lineares complexos.

- Requer tuning de parâmetros.
- Pode demorar mais para treinar.

---

## ▶️ Como Executar

1. Execute o pré-processamento:
```bash
python preprocessamento_carros.py
```

2. Execute os modelos individualmente, por exemplo:
```bash
python logistic_regression.py
```

3. Ou execute o comparador completo:
```bash
python comparador_final.py
```

4. Verifique os resultados em `comparativo_modelos.txt`.

---

## 📝 Observações

- A variável alvo é binária: 1 (com couro) ou 0 (sem couro).
- As variáveis categóricas foram codificadas com `get_dummies`.
- A divisão dos dados foi feita em 80% para treino e 20% para teste.

---

## 📬 Contato

Para dúvidas ou sugestões, entre em contato com o responsável pelo projeto.

