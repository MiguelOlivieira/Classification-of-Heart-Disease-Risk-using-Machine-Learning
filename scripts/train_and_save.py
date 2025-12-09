import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

def train_and_save():
    # Nota: Se der erro de leitura, verifique se o arquivo está na mesma pasta onde você roda o comando
    csv_path = "heart.xls"  
    model_path = "app_backend/model/heart.pkl"

    # Criar pasta para salvar o modelo, se não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Lendo dados de: {csv_path}...")
    try:
        
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Não achei o arquivo '{csv_path}'.")
        print("Verifique se você está rodando o terminal na pasta raiz do projeto.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    # 2. Separar as colunas (X) e o alvo (y)
    # Garante que usamos a coluna 'target' como objetivo
    if 'target' not in df.columns:
        print("ERRO: O arquivo não tem a coluna 'target'. Verifique os dados.")
        return

    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Treinar a Árvore de Decisão (Decision Tree)
    print("Treinando o modelo de Árvore de Decisão...")
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X, y)

    # 4. Salvar o modelo pronto
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Sucesso! Modelo salvo em: {model_path}")

# Rodar o treinamento e salvar o modelo quando este script for executado diretamente
if __name__ == "__main__":
    train_and_save()