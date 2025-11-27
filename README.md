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

A API ficará acessível em:

```
http://127.0.0.1:8000
```

### Endpoints importantes

| Método | Rota | Descrição |
|-------|------|-----------|
| GET | `/health` | Verifica conexão com o MongoDB |
| POST | `/api/v1/sessions/{session_id}/gestures` | Ingestão de lote de gestos |

---

## 🧪 5. Testando o Endpoint Principal

Use um cliente HTTP como:

- Insomnia  
- Postman  
- Thunder Client  
- cURL  

### Exemplo usando `curl`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/sessions/S_20251126_001/gestures" \
  -H "Content-Type: application/json" \
  -d '{
        "patient_id": "P001",
        "session_id": "S_20251126_001",
        "phone_id": "PHONE_GALAXY_S23",
        "model_version": "tflite_v1.0",
        "start_time": "2025-11-26T19:00:00Z",
        "end_time": "2025-11-26T19:05:00Z",
        "gestures": [
          { "timestamp": "2025-11-26T19:00:01.200Z", "window_id": 1, "label": "G1", "confidence": 0.92 },
          { "timestamp": "2025-11-26T19:00:02.200Z", "window_id": 2, "label": "G1", "confidence": 0.90 },
          { "timestamp": "2025-11-26T19:00:03.200Z", "window_id": 3, "label": "G1", "confidence": 0.88 }
        ]
      }'
```

### Resposta esperada

```json
{
  "session_id": "S_20251126_001",
  "inserted_gestures": 3,
  "post_processing_triggered": true
}
```

---

## 🗄️ 6. Collections criadas automaticamente no MongoDB

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

## 🧩 7. Sobre o Pós-Processamento

O backend já contém um **stub**:

```python
async def run_post_processing(session_id: str):
    print(f"[POST-PROCESSING] Triggered for session_id={session_id}")
```

Futuramente será substituído por:

- Leitura dos `gesture_events`
- Execução da lógica simbólica/fuzzy
- Criação de `medication_intake_events`
- Atualização do `status` da sessão para `"processed"`

---

## 📚 8. Documentação automática

O FastAPI gera documentação automática:

### 🎨 Swagger UI  
```
http://127.0.0.1:8000/docs
```

### 📘 ReDoc  
```
http://127.0.0.1:8000/redoc
```

---

## 🧱 9. Roadmap dos próximos passos

- [ ] Implementar o pós-processamento real  
- [ ] Adicionar endpoints para o médico/paciente  
- [ ] Criar coleção `prescriptions`  
- [ ] Criar autenticação JWT  
- [ ] Adicionar blockchain (registro de ingestões e prescrições)

---
