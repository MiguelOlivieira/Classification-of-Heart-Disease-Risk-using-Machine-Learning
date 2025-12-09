## Introdução 🚀

**Nome do projeto:** **Classification-of-Heart-Disease-Risk-using-Machine-Learning**

**Equipe / Alunos envolvidos:**
Clebson Alexandre, Nicolas Klayvert, Diego Luiz, Sérgio Roberto, Leonardo Antônio, José Miguel.

**Contexto da atividade:**
Projeto acadêmico da disciplina — usando o conjunto de dados sorteado, a equipe construiu uma aplicação completa (Front-End, Back-End e modelo de IA) para resolver um problema de **classificação supervisionada**.

**Objetivo principal / desafio:**
Prever o **risco de doença cardíaca** a partir dos dados de pacientes — integrando todo o pipeline: dados, modelo de ML, backend e frontend. ❤️‍🩹

**Motivação:**
Aprender na prática como aplicar Machine Learning + engenharia de software, entregando uma aplicação funcional ao invés de apenas um script ou notebook.

---

## Principais Funcionalidades do Projeto

* 📊 **Pré-processamento de dados** — limpeza, normalização e divisão em treino/teste.
* 🧠 **Treinamento de modelo supervisionado** — classificação do risco cardíaco.
* 🔗 **API Backend com FastAPI** — recebe dados e retorna a predição do modelo treinado.
* 🖥️ **Frontend em Streamlit** — formulário amigável para o usuário inserir seus dados.
* 🔁 **Pipeline completo** — dataset → ML → API → interface → resultado final.

---

## Tecnologias Utilizadas

### 🔧 Backend / IA

* **FastAPI**
* **scikit-learn**
* **pandas / numpy**
* **joblib**
* **pydantic**
* **python-multipart**
* **python-dotenv**

### 🖥️ Frontend

* **Streamlit**
* **Requests** (consulta a API)

### ⚙️ Ambiente

* **uvicorn** para rodar a API
* **requirements.txt** para instalar dependências

---

## 🗂️ Requirements.txt

Aqui estão todos os pacotes necessários (como você pediu):

```
fastapi
uvicorn[standard]
scikit-learn
pandas
numpy
python-multipart
joblib
pydantic
streamlit
requests
python-dotenv
```

---

# ▶️ Como Rodar o Projeto (Passo a Passo)

## 1️⃣ Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

## 2️⃣ Criar e ativar o ambiente virtual

### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# 🧠 4️⃣ Treinar o modelo (se houver script)

Se seu projeto tem um arquivo tipo `train.py`:

```bash
python train.py
```

Isso deve gerar algo como:
📌 `modelo_treinado.joblib`

---

# 🚀 5️⃣ Rodar o **BACKEND (FastAPI)**

Dentro da pasta do backend:

```bash
uvicorn main:app --reload
```

A API ficará disponível em:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

E a documentação automática da API (Swagger):
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

# 🖥️ 6️⃣ Rodar o **FRONTEND (Streamlit)**

Na pasta do frontend:

```bash
streamlit run app.py
```

A interface abrirá no navegador automaticamente.

---

# 📸 Capturas de Tela

<img src="https://drive.google.com/uc?export=view&id=12D-1ncQthp0xOMNk_DtZShIiJ98fTXn1" width="500px" />


---

# 👥 Colaboradores 🕴️

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/leonard0antonio">
        <img src="https://github.com/leonard0antonio.png" width="100px;" />
        <br /><sub><b>Leonardo Antonio</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ClebsAlexandre">
        <img src="https://github.com/ClebsAlexandre.png" width="100px;" />
        <br /><sub><b>Clebson Alexandre</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/nicolasklayvert">
        <img src="https://github.com/nicolasklayvert.png" width="100px;" />
        <br /><sub><b>Nicolas Klayvert</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/DiegoL13">
        <img src="https://github.com/DiegoL13.png" width="100px;" />
        <br /><sub><b>Diego Luiz</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SergioRoberto-DEV">
        <img src="https://github.com/SergioRoberto-DEV.png" width="100px;" />
        <br /><sub><b>Sérgio Roberto</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MiguelOlivieira">
        <img src="https://github.com/MiguelOlivieira.png" width="100px;" />
        <br /><sub><b>José Miguel</b></sub>
      </a>
    </td>
  </tr>
</table>
