#ALGUNS DADOS QUE DEVEM EXISTIR NO INPUT DO USUARIO:

#   age: int, #idade
#   sex: Literal[0, 1] #sexo -> recebe apenas [0 ou 1]
#   cp: #chest pain type (são 4 valores, de 0 a 4) -> recebe apenas [0, 1, 2 ou 4]
#  trestbps:  int,  #resting blood pressure (pressão arterial em repouso) 
#  chol:  int, #serum cholestoral in mg/dl (colesterol sérico em mg/dl)
#  fbs: #fasting blood sugar > 120 mg/dl (fasting blood sugar > 120 mg/dl) -> recebe apenas [0, 1]
# restecg: #resting electrocardiographic results (values 0,1,2) (resultados do eletrocardiograma em repouso) -> recebe apenas [0, 1 ou 2]
#  thalach: int,  #maximum heart rate achieved (frequência cardíaca máxima atingida)
#  exang:  #exercise induced angina (angina induzida por exercício) -> recebe apenas [0, 1]
# oldpeak: float , #ST depression induced by exercise relative to rest (depressão do segmento ST induzida pelo exercício em relação ao repouso)
# slope: the slope of the peak exercise ST segment (inclinação do segmento ST no pico do exercício) -> recebe apenas [0, 1 ou 2]
# ca: number of major vessels (0-3) colored by flourosopy (número de vasos principais (0–3) coloridos por fluoroscopia) -> recebe apenas [0, 1, 2 ou 3]
#  thal: thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (thal: 0 = normal; 1 = defeito fixo; 2 = defeito reversível) -> recebe apenas [0, 1, 2 ou 3]
#
#TODOS OS DADOS ACIMA FORAM RETIRADOS DO CSV DO NOSSO MODELO, SÃO ESSES OS DADOS Q A I.A ESPERA NO INPUT
#CASO DUVIDAS SOBRE OS DADOS ACIMA, CONSULTEM EM: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download
#OU https://colab.research.google.com/drive/1Yq_Cbc8-KGwZbna0ByTsFqSQiTOK3oxG?usp=sharing (vai precisar configurar primeiro)


import streamlit as st
import requests

# --- Configuração da Página ---
st.set_page_config(
    page_title="Diagnóstico Cardíaco IA",
    page_icon="💓",
    layout="wide"
)

# --- Barra Lateral (Configurações) ---
with st.sidebar:
    st.header("⚙️ Configurações")
    # Permite mudar a URL se você fizer deploy na nuvem depois
    api_url = st.text_input("URL da API do Modelo", value="http://localhost:8000/riskpredict")
    st.info("Certifique-se de que o arquivo 'api.py' (backend) esteja rodando.")
    st.write("---")
    st.markdown("Desenvolvido para auxílio médico.")

# --- Título e Cabeçalho ---
st.title("💓 Diagnóstico Assistido por Inteligência Artificial")
st.markdown("""
**Instruções:** Preencha os dados clínicos abaixo. O sistema utilizará um modelo de Machine Learning 
para estimar a probabilidade de risco cardíaco.
""")
st.divider()

# --- Formulário de Entrada ---
with st.form("ficha_medica"):
    st.subheader("📋 Dados Pessoais e Sintomas")
    
    col1, col2, col3 = st.columns(3)
    
    # Dados Pessoais
    with col1:
        age = st.number_input("Idade", min_value=1, max_value=120, value=45, step=1)
        
        sex_display = st.radio("Sexo Biológico", ["Masculino", "Feminino"], horizontal=True)
        sex = 1 if sex_display == "Masculino" else 0

    # Dados Clínicos
    with col2:
        cp_labels = {
            0: "Angina Típica (Dor forte/aperto)",
            1: "Angina Atípica (Desconforto)",
            2: "Dor não-cardíaca (Outra origem)",
            3: "Assintomático (Sem dor)"
        }
        cp = st.selectbox(
            "Tipo de dor no peito:", 
            options=[0, 1, 2, 3], 
            format_func=lambda x: cp_labels[x]
        )
    # Exang (Angina induzida por exercício)
    with col3:
        exang_display = st.radio("Sente dor ao fazer esforço físico?", ["Não", "Sim"], horizontal=True)
        exang = 1 if exang_display == "Sim" else 0

    st.write("") # Espaçamento
    st.subheader("🩺 Sinais Vitais e Exames")
    col4, col5 = st.columns(2)

    # Dados Vitais e Exames
    with col4:
        trestbps = st.slider("Pressão Arterial em Repouso (mmHg)", 90, 200, 120, help="Pressão sistólica (valor maior).")
        chol = st.slider("Colesterol Total (mg/dl)", 100, 600, 200)
        
        # FBS (Glicemia de jejum)
        fbs_display = st.checkbox("Glicemia de jejum > 120 mg/dl? (Diabetes/Pré)")
        fbs = 1 if fbs_display else 0

    with col5:
        # RestECG
        restecg_labels = {
            0: "Normal",
            1: "Anormalidades de onda ST-T",
            2: "Hipertrofia Ventricular Provável"
        }
        restecg = st.selectbox(
            "Eletrocardiograma em Repouso:", 
            options=[0, 1, 2], 
            format_func=lambda x: restecg_labels[x]
        )
        
        thalach = st.number_input("Frequência Cardíaca Máxima (bpm)", 60, 220, 150)

    st.write("")
    st.subheader("🔬 Indicadores Avançados (Laudo Médico)")
    
    # Expander aberto por padrão para facilitar a visualização
    with st.container():
        col6, col7 = st.columns(2)
        with col6:
            oldpeak = st.number_input("Depressão ST (Oldpeak)", 0.0, 10.0, 0.0, step=0.1, help="Depressão do segmento ST induzida pelo exercício.")
            
            slope_labels = {0: "Inclinando p/ Cima (Upsloping)", 1: "Plano (Flat)", 2: "Inclinando p/ Baixo (Downsloping)"}
            slope = st.selectbox("Inclinação do Segmento ST (Slope):", [0, 1, 2], format_func=lambda x: slope_labels[x])
            
        with col7:
            # CORREÇÃO 1: Removida a opção 4 que geralmente é NaN no dataset original
            ca = st.selectbox("Vasos principais coloridos na Fluoroscopia (0-3):", [0, 1, 2, 3], help="Quanto maior o número, melhor a circulação visível.")
            
            # CORREÇÃO 2: Removida a opção 0 (Erro/Nulo) para evitar envio de dados sujos
            thal_labels = {
                1: "Normal", 
                2: "Defeito Fixo (Fixed Defect)", 
                3: "Defeito Reversível (Reversable Defect)"
            }
            # Se seu modelo foi treinado onde 0 era algo válido, adicione o 0 na lista abaixo. 
            # Mas geralmente em produção removemos o 0.
            thal = st.selectbox("Talassemia (Thal):", options=[1, 2, 3], format_func=lambda x: thal_labels[x])

    st.write("---")
    # Botão de Envio
    submit = st.form_submit_button("🔍 PROCESSAR DIAGNÓSTICO", type="primary", use_container_width=True)

# --- Lógica de Envio e Exibição ---
if submit:
    # Monta o JSON igual ao Pydantic do backend
    payload = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": int(thalach),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal)
    }

    # Chama a API
    try:
        with st.spinner("Conectando à IA Médica..."):
            response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            resultado_texto = data['predicted_class']
            confianca = data['confidence'] * 100
            
            st.divider()
            st.subheader("📄 Resultado da Análise")
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                # Lógica de cor baseada no texto da resposta
                # Ajuste as strings abaixo conforme o que seu 'model_util.py' retorna
                if "Alto" in resultado_texto or "Doença" in resultado_texto or "Risco" in resultado_texto and "Baixo" not in resultado_texto:
                    st.error(f"### {resultado_texto}")
                    st.markdown("⚠️ **Atenção:** Recomenda-se avaliação clínica detalhada.")
                else:
                    st.success(f"### {resultado_texto}")
                    st.markdown("✅ **Status:** Indicadores dentro do padrão de baixo risco.")
            
            with col_b:
                st.write("Probabilidade calculada pelo modelo:")
                st.progress(confianca / 100)
                st.caption(f"Confiança da IA: {confianca:.2f}%")
                
            # JSON Debug (Opcional, bom para desenvolvedores verem o que foi enviado)
            with st.expander("Ver dados técnicos enviados"):
                st.json(payload)
                
        else:
            st.error(f"Erro na API: {response.status_code}")
            st.write(response.text)
    
    # Tratamento de erros de conexão      
    except requests.exceptions.ConnectionError:
        st.error("❌ Não foi possível conectar ao servidor.")
        st.warning("Dica: Verifique se você rodou o comando `uvicorn app_backend.api:app --reload` no terminal.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")