# 📘 Medication Adherence Backend — Minimal Version
Backend em **FastAPI + MongoDB** para ingestão de gestos classificados no celular e execução do pós-processamento.

---

## 📁 Estrutura do Projeto

```
med-adherence-backend/
├── app/
│   ├── main.py
│   └── config.py
├── README.md
└── requirements.txt  (ou pyproject.toml se usar Poetry)
```

---

## 🚀 1. Pré-requisitos

Antes de iniciar, instale:

### **MongoDB**
Local ou remoto (Atlas).  
Rodando localmente no Linux:

```bash
sudo systemctl start mongod
```

Verifique se está funcionando:

```bash
mongo --eval 'db.runCommand({ ping: 1 })'
```

### **Python 3.10+**

Recomendo virtualenv ou Conda.

---

## 📦 2. Instalação das dependências

Se estiver usando `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ou instalação manual:

```bash
pip install fastapi uvicorn motor pydantic[dotenv]
```

---

## ⚙️ 3. Configuração do MongoDB

Por padrão, o backend usa:

- URI: `mongodb://localhost:27017`
- Banco: `med_adherence`

Você pode mudar isso criando um arquivo `.env` na raiz:

```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=med_adherence
```

O arquivo `app/config.py` carrega essas variáveis automaticamente.

---

## ▶️ 4. Rodando o Backend

Execute o servidor FastAPI com:

```bash
uvicorn app.main:app --reload
```


---

## 🗄️ 5. Collections criadas automaticamente no MongoDB

Quando o backend recebe dados, ele cria essas coleções:

- `sessions`
- `gesture_events`
- `medication_intake_events` (quando o pós-processamento for implementado)

Você pode inspecionar no MongoDB:

```bash
mongosh
use med_adherence
db.sessions.find()
db.gesture_events.find()
```

---


## 📚 6. Documentação automática

O FastAPI gera documentação automática:

### 🎨 Swagger UI  
```
http://127.0.0.1:8000/docs
```

### 📘 ReDoc  
```
http://127.0.0.1:8000/redoc
```

