from fastapi import FastAPI
from pydantic import BaseModel, Field, conint #a biblioteca Field permite reconhecer o que vem do frontend
from typing import List, Literal 
from fastapi.middleware.cors import CORSMiddleware #Cors para permitir 
from .model_util import load_model, predict_instance 

app = FastAPI(title ="Heart Disease Risk Predictor API")


#permissoes para acessar a porta do front (streamlit)
app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
)

#Entrada com os dados dos usuario
class ClinicalUserInput(BaseModel):
    age: int = Field(..., gt=0) #idade
    sex: Literal[0, 1] #sexo, deve ser recebido como um inteiro
    cp:  Literal[0, 4] #chest pain type (são 4 valores, de 0 a 4)
    trestbps:  int = Field(..., gt=0) #resting blood pressure (pressão arterial em repouso)
    chol:  int = Field(..., gt=0) #serum cholestoral in mg/dl (colesterol sérico em mg/dl)
    fbs:  Literal[0, 1]#fasting blood sugar > 120 mg/dl (fasting blood sugar > 120 mg/dl)
    restecg: Literal[0, 2]#resting electrocardiographic results (values 0,1,2) (resultados do eletrocardiograma em repouso)
    thalach: int = Field(..., gt=0) #maximum heart rate achieved (frequência cardíaca máxima atingida)
    exang: Literal[0, 1] #exercise induced angina (angina induzida por exercício)
    oldpeak: float = Field(..., ge=0) #ST depression induced by exercise relative to rest (depressão do segmento ST induzida pelo exercício em relação ao repouso)
    slope: Literal[0, 2]#the slope of the peak exercise ST segment (inclinação do segmento ST no pico do exercício)
    ca: Literal[0, 3] #number of major vessels (0-3) colored by flourosopy (número de vasos principais (0–3) coloridos por fluoroscopia)
    thal: Literal[0, 3]  #thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (thal: 0 = normal; 1 = defeito fixo; 2 = defeito reversível)


#Saída
class HeartRiskPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float #grau de confiança que pode ir de 0 a 1
    probabilites: List[float] #Lista de probabilidade 
    
#TODO: Inserir o nome do arquivo modelo
MODEL_PATH = "app.backend/model/nome_do_modelo.pkl"
model = load_model(MODEL_PATH)  #TODO: fazer o metodo load_model (em model_util.py)

@app.get("/")
def read_root():
    return {"message": "Classification of Heart Disease Risk API. POST to"}

@app.post("/riskpredict", response_model=HeartRiskPredictionResponse)
def riskpredict(data: ClinicalUserInput):
    x = [
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal,
    ]
    pred_class, confidence, probs = predict_instance(model, x)
    return HeartRiskPredictionResponse(
        predicted_class= pred_class,
        confidence=round(confidence, 4),
        probabilites=[round(float(p), 4) for p in probs.tolist()]
    )
    