import joblib
import numpy as np
import os

def load_model(path: str):
    """Carrega o artefato .pkl do disco com tratamento de erro."""
    if not os.path.exists(path):
        print(f"[ERRO] Arquivo de modelo não encontrado em: {path}")
        return None
        
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[ERRO] Falha ao deserializar modelo: {e}")
        return None

def predict_instance(model, features: list):
    """
    Executa a predição para uma única instância.
    Retorna: (Label legível, Confiança, Classe bruta)
    """
    input_vector = np.array([features])

    try:
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector)[0]

        # Mapeamento de classe: 1 = Doença, 0 = Saudável
        if prediction == 1:
            label = "Alto Risco de Doença Cardíaca"
            conf = probs[1]
        else:
            label = "Baixo Risco (Saudável)"
            conf = probs[0]
            
        return label, conf, prediction

    except Exception as e:
        print(f"[ERRO] Falha na inferência: {e}")
        return "Erro no Processamento", 0.0, -1