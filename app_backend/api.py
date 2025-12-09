from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
from fastapi.middleware.cors import CORSMiddleware
from .model_util import load_model, predict_instance
import os

# Cria o app FastAPI
app = FastAPI(title="API de Risco Cardíaco")

# Permite que o Streamlit converse com esse servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definição dos dados que vêm do Site
class ClinicalUserInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Resposta que vai voltar para o Site
class HeartRiskPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

# Carrega o modelo assim que o servidor liga
MODEL_PATH = "app_backend/model/arvore_decisao_classificador_model.pkl"
model = load_model(MODEL_PATH)

# Rota de predição
@app.post("/riskpredict", response_model=HeartRiskPredictionResponse)
def riskpredict(data: ClinicalUserInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo de IA não carregado (arquivo .pkl faltando)")

    # Monta a lista na ordem EXATA que foi usada no treinamento (heart.csv)
    # ATENÇÃO: Se mudar a ordem aqui, a IA erra tudo!
    x = [
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,      # <--- Corrigido: O original esquecia esse campo
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]
    
    # Chama a função de previsão
    pred_class, confidence, _ = predict_instance(model, x)
    
    # Retorna a resposta
    return {
        "predicted_class": pred_class,
        "confidence": confidence
    }