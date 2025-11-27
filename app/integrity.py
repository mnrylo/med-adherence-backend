# app/integrity.py

import json
import hashlib
from datetime import datetime, date
from typing import Any, Dict


def compute_data_hash(payload: Dict[str, Any]) -> str:
    """
    Gera um hash SHA-256 determinístico de um payload JSON.

    Importante:
    - Não deve conter tipos não serializáveis (ObjectId, datetime, etc),
      então usamos um serializer customizado.
    """
    def default_serializer(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # qualquer outro tipo estranho vira string
        return str(obj)

    raw = json.dumps(payload, sort_keys=True, default=default_serializer).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def append_to_ledger(
    db,
    entry_type: str,
    ref_id: str,
    patient_id: str,
    payload: Dict[str, Any],
    data_hash: str,
):
    """
    Adiciona uma entrada ao ledger local (blockchain simplificada).

    - entry_type: 'prescription' ou 'intake_event'
    - ref_id: id lógico da entidade (ex: prescription_id, intake_event_id)
    - patient_id: id do paciente
    - payload: snapshot do documento
    - data_hash: hash de integridade do documento
    """
    now = datetime.utcnow()

    # Recuperar último bloco para pegar index e prev_hash
    last = await db.ledger_entries.find_one({}, sort=[("index", -1)])

    if last is None:
        prev_hash = "GENESIS"
        next_index = 1
    else:
        prev_hash = last.get("chain_hash", "GENESIS")
        next_index = int(last.get("index", 0)) + 1

    # Montar string base para o hash encadeado
    chain_payload = {
        "index": next_index,
        "entry_type": entry_type,
        "ref_id": ref_id,
        "patient_id": patient_id,
        "timestamp": now.isoformat(),
        "prev_hash": prev_hash,
        "data_hash": data_hash,
    }
    chain_hash = compute_data_hash(chain_payload)

    ledger_doc = {
        "index": next_index,
        "entry_type": entry_type,
        "ref_id": ref_id,
        "patient_id": patient_id,
        "timestamp": now,
        "payload": payload,
        "prev_hash": prev_hash,
        "data_hash": data_hash,
        "chain_hash": chain_hash,
        "created_at": now,
    }

    await db.ledger_entries.insert_one(ledger_doc)
    print(
        f"[LEDGER] Appended {entry_type} ref_id={ref_id} index={next_index} "
        f"chain_hash={chain_hash[:16]}..."
    )


async def setup_integrity_indexes(db):
    """
    Cria índices necessários para a coleção de ledger.
    Chamar dentro do startup_event do FastAPI.
    """
    await db.ledger_entries.create_index("patient_id")
    await db.ledger_entries.create_index("entry_type")
    await db.ledger_entries.create_index("index", unique=True)
