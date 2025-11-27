# app/main.py
from bson import ObjectId
from datetime import datetime, time, date, timedelta
from typing import List, Optional

import json
import hashlib

from app.base_model_classes import *
from app.integrity import compute_data_hash, append_to_ledger, setup_integrity_indexes

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from motor.motor_asyncio import AsyncIOMotorClient

from .config import settings

# -------------------------
# PillIntakeEvaluator + regras
# -------------------------

gesture_confidence = {
    ("G1", "G2", "G3", "G4"): 1.0,
    ("G1", "G2", "G4"): 0.8,
    ("G1", "G2", "G3"): 0.6,
    ("G1", "G2"): 0.5,
    ("G1", "G3"): 0.3,
    ("G1", "G4"): 0.2,
}


class PillIntakeEvaluator:
    def __init__(self, gesture_confidence: dict, trigger: str = "G1"):
        self.rules = {tuple(k): v for k, v in gesture_confidence.items()}
        self.trigger = trigger

    def evaluate_with_trigger(self, gesture_sequence, window_size: int = 10):
        """
        Procura o gesto gatilho (self.trigger) e avalia as regras
        na janela subsequente de tamanho window_size.

        gesture_sequence : list[str]  - ex: ["G1", "G1", "T", "G2", ...]
        window_size      : int        - número de gestos considerados após o gatilho
        """
        results = []

        for i, g in enumerate(gesture_sequence):
            if g == self.trigger:
                # subsequência a partir do gatilho
                subseq = gesture_sequence[i : i + window_size]

                # Remove G5 e T da subsequência para avaliação das regras
                filtered = [x for x in subseq if x != "G5" and x != "T"]
                unique = set(filtered)

                # Encontra a regra com maior confiança que seja subset dos gestos presentes
                max_conf, matched = 0.0, None
                for rule, conf in self.rules.items():
                    if all(r in unique for r in rule):
                        if conf > max_conf:
                            max_conf, matched = conf, rule

                results.append(
                    {
                        "trigger_index": i,
                        "subsequence": filtered,
                        "confidence": max_conf,
                        "matched_rule": matched,
                        "window_size": window_size,
                    }
                )

        return results



# -------------------------
# FastAPI app + Mongo client
# -------------------------

app = FastAPI(title="Medication Adherence Backend (Minimal)")

mongo_client: AsyncIOMotorClient | None = None


@app.on_event("startup")
async def startup_event():
    global mongo_client
    mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    # You can also create indexes here if you want
    db = mongo_client[settings.mongodb_db]
    await db.sessions.create_index("patient_id")
    await db.sessions.create_index("start_time")
    await db.gesture_events.create_index("session_id")
    await db.gesture_events.create_index([("patient_id", 1), ("timestamp", 1)])
    await db.medication_intake_events.create_index(
        [("patient_id", 1), ("intake_time", 1)]
    )
    await db.prescriptions.create_index([("patient_id", 1), ("status", 1)])
    await setup_integrity_indexes(db)


@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client is not None:
        mongo_client.close()


def get_db():
    global mongo_client
    if mongo_client is None:
        raise RuntimeError("Mongo client is not initialized")
    return mongo_client[settings.mongodb_db]

def map_prescription_doc(doc: dict) -> PrescriptionOut:
    return PrescriptionOut(
        prescription_id=str(doc.get("_id")),
        patient_id=doc.get("patient_id"),
        medication_id=doc.get("medication_id"),
        medication_name=doc.get("medication_name"),
        dose=doc.get("dose"),
        route=doc.get("route"),
        frequency=doc.get("frequency"),
        scheduled_times=doc.get("scheduled_times") or [],
        start_date=doc.get("start_date"),
        end_date=doc.get("end_date"),
        status=doc.get("status", "active"),
        notes=doc.get("notes"),
        created_by=doc.get("created_by"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


def map_intake_event_doc(doc: dict) -> IntakeEventOut:
    return IntakeEventOut(
        event_id=str(doc.get("_id")),
        session_id=doc.get("session_id"),
        patient_id=doc.get("patient_id"),
        prescription_id=doc.get("prescription_id"),
        medication_id=doc.get("medication_id"),
        intake_time=doc.get("intake_time"),
        confidence=float(doc.get("confidence", 0.0) or 0.0),
        detected_by=doc.get("detected_by", ""),
        status=doc.get("status", "auto_detected"),
        source_gestures=doc.get("source_gestures") or {},
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


def map_session_doc(doc: dict) -> SessionOut:
    return SessionOut(
        session_id=str(doc.get("_id")),
        patient_id=doc.get("patient_id"),
        watch_id=doc.get("watch_id"),
        phone_id=doc.get("phone_id"),
        start_time=doc.get("start_time"),
        end_time=doc.get("end_time"),
        status=doc.get("status"),
        model_version=doc.get("model_version"),
        notes=doc.get("notes"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


async def compute_adherence_summary(
    db,
    patient_id: str,
    window_days: int = 7,
) -> PatientAdherenceSummary:
    """
    Calcula a adesão de um paciente em uma janela de N dias (default 7).

    Estratégia:
    - Janela: [hoje - (window_days-1), hoje]
    - Considera prescrições ativas do paciente cujo intervalo [start_date, end_date]
      intersecta essa janela.
    - Para cada prescrição:
        * expected_doses por dia = len(scheduled_times) para cada dia em que a prescrição está ativa
        * detected_doses por dia = número de intake_events dessa prescrição naquela data
    """

    today = datetime.utcnow().date()
    window_end = today
    window_start = today - timedelta(days=window_days - 1)

    # 1) Buscar prescrições ativas do paciente
    cursor = db.prescriptions.find({"patient_id": patient_id, "status": "active"})
    pres_docs = await cursor.to_list(length=None)

    # filtrar apenas prescrições cujo intervalo intersecta a janela
    prescriptions = []
    for p in pres_docs:
        start_dt = p.get("start_date")
        end_dt = p.get("end_date")

        if isinstance(start_dt, datetime):
            p_start = start_dt.date()
        else:
            p_start = start_dt

        if isinstance(end_dt, datetime):
            p_end = end_dt.date()
        else:
            p_end = end_dt  # pode ser None

        # se end_date é None, consideramos "sem fim"
        if p_end is None:
            active_end = window_end
        else:
            active_end = p_end

        # interseção com a janela
        if p_start is None:
            continue

        # intervalo real de atividade considerando janela
        eff_start = max(p_start, window_start)
        eff_end = min(active_end, window_end)

        if eff_start > eff_end:
            # não intersecta janela
            continue

        p["_eff_start"] = eff_start
        p["_eff_end"] = eff_end
        prescriptions.append(p)

    # se não há prescrições, devolve vazio
    if not prescriptions:
        return PatientAdherenceSummary(
            patient_id=patient_id,
            window_start=window_start,
            window_end=window_end,
            prescriptions=[],
        )

    # 2) Buscar intake_events na janela (somente com prescription_id preenchido)
    window_start_dt = datetime.combine(window_start, time.min)
    window_end_dt = datetime.combine(window_end, time.max)

    cursor = db.medication_intake_events.find(
        {
            "patient_id": patient_id,
            "intake_time": {"$gte": window_start_dt, "$lte": window_end_dt},
            "prescription_id": {"$ne": None},
        }
    )
    intake_docs = await cursor.to_list(length=None)

    # agrupar intake_events por (prescription_id, data)
    intake_by_presc_date: dict[tuple[str, date], int] = {}
    for ev in intake_docs:
        pid = ev.get("prescription_id")
        if not pid:
            continue
        itime = ev.get("intake_time")
        if not isinstance(itime, datetime):
            continue
        d = itime.date()
        key = (pid, d)
        intake_by_presc_date[key] = intake_by_presc_date.get(key, 0) + 1

    # 3) Construir resumo por prescrição
    pres_summaries: List[PrescriptionAdherenceSummary] = []

    # lista das datas da janela para iterar (garante ordem)
    all_days = [
        window_start + timedelta(days=i) for i in range(window_days)
    ]

    for p in prescriptions:
        pid = p.get("_id")
        medication_id = p.get("medication_id")
        medication_name = p.get("medication_name")
        eff_start: date = p["_eff_start"]
        eff_end: date = p["_eff_end"]

        sched_times = p.get("scheduled_times") or []
        doses_per_day = len(sched_times)

        daily_items: List[DailyAdherenceItem] = []
        total_expected = 0
        total_detected = 0

        for d in all_days:
            # só conta o dia se estiver dentro do intervalo efetivo da prescrição
            if d < eff_start or d > eff_end:
                daily_items.append(
                    DailyAdherenceItem(
                        date=d,
                        expected_doses=0,
                        detected_doses=0,
                        adherence=None,
                    )
                )
                continue

            expected = doses_per_day
            detected = intake_by_presc_date.get((pid, d), 0)

            total_expected += expected
            total_detected += detected

            adh = (detected / expected) if expected > 0 else None

            daily_items.append(
                DailyAdherenceItem(
                    date=d,
                    expected_doses=expected,
                    detected_doses=detected,
                    adherence=adh,
                )
            )

        overall_adh = (
            (total_detected / total_expected) if total_expected > 0 else None
        )

        # converter start/end para date para a resposta
        start_dt = p.get("start_date")
        end_dt = p.get("end_date")
        if isinstance(start_dt, datetime):
            start_date_val = start_dt.date()
        else:
            start_date_val = start_dt

        if isinstance(end_dt, datetime):
            end_date_val = end_dt.date()
        else:
            end_date_val = end_dt

        pres_summaries.append(
            PrescriptionAdherenceSummary(
                prescription_id=str(pid),
                medication_id=medication_id,
                medication_name=medication_name,
                start_date=start_date_val,
                end_date=end_date_val,
                total_expected_doses=total_expected,
                total_detected_doses=total_detected,
                adherence=overall_adh,
                daily=daily_items,
            )
        )

    return PatientAdherenceSummary(
        patient_id=patient_id,
        window_start=window_start,
        window_end=window_end,
        prescriptions=pres_summaries,
    )


# -------------------------
# Healthcheck
# -------------------------

@app.get("/health")
async def health():
    db = get_db()
    # Just a simple ping; will raise if disconnected
    await db.command("ping")
    return {"status": "ok"}


# -------------------------
# Core endpoint:
#   POST /api/v1/sessions/{session_id}/gestures
# -------------------------

@app.post(
    "/api/v1/sessions/{session_id}/gestures",
    response_model=IngestResponse,
    summary="Ingest gesture events for a session",
)
async def ingest_session_gestures(
    session_id: str = Path(..., description="Session ID in path"),
    payload: SessionPayload = ...,
):
    """
    Ingests a batch of gesture classification results for a given session.

    1. Validates that path session_id == payload.session_id.
    2. Upserts the Session document in the 'sessions' collection.
    3. Inserts GestureEvent documents into 'gesture_events'.
    4. Triggers post-processing (placeholder) to create MedicationIntakeEvents.
    """

    if payload.session_id != session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id in path does not match session_id in payload",
        )

    db = get_db()

    # 1) Upsert Session
    now = datetime.utcnow()
    session_doc = {
        "_id": payload.session_id,
        "patient_id": payload.patient_id,
        "phone_id": payload.phone_id,
        "watch_id": None,  # you can fill this later when you have it
        "start_time": payload.start_time,
        "end_time": payload.end_time,
        "status": "finished",  # backend may later change to 'processed'
        "model_version": payload.model_version,
        "notes": None,
        "created_at": now,
        "updated_at": now,
    }

    # Use upsert to create or update the session
    await db.sessions.update_one(
        {"_id": payload.session_id},
        {
            "$setOnInsert": {
                "created_at": now,
            },
            "$set": {
                "patient_id": payload.patient_id,
                "phone_id": payload.phone_id,
                "start_time": payload.start_time,
                "end_time": payload.end_time,
                "status": "finished",
                "model_version": payload.model_version,
                "updated_at": now,
            },
        },
        upsert=True,
    )

    # 2) Insert GestureEvents in bulk
    gestures_docs = []
    for g in payload.gestures:
        gestures_docs.append(
            {
                # let Mongo generate _id automatically
                "session_id": payload.session_id,
                "patient_id": payload.patient_id,
                "timestamp": g.timestamp,
                "window_id": g.window_id,
                "label": g.label,
                "confidence": g.confidence,
                "model_version": payload.model_version,
                "created_at": now,
            }
        )

    inserted_gestures = 0
    if gestures_docs:
        result = await db.gesture_events.insert_many(gestures_docs)
        inserted_gestures = len(result.inserted_ids)

    # 3) Trigger post-processing (placeholder)
    await run_post_processing(session_id=payload.session_id)

    return IngestResponse(
        session_id=payload.session_id,
        inserted_gestures=inserted_gestures,
        post_processing_triggered=True,
    )


# -------------------------
# Prescription Endpoint
# POST
# -------------------------

@app.post(
    "/api/v1/patients/{patient_id}/prescriptions",
    summary="Create or update a prescription for a patient",
)
async def create_or_update_prescription(patient_id: str,payload: PrescriptionIn,):
    """
    Cria ou atualiza uma prescrição para o paciente.

    Usa prescription_id como _id no Mongo.
    """
    db = get_db()
    now = datetime.utcnow()

    if payload.prescription_id == "":
        raise HTTPException(status_code=400, detail="prescription_id must not be empty")

    # converter date -> datetime (meia-noite)
    start_dt = datetime.combine(payload.start_date, time.min)
    end_dt = (
        datetime.combine(payload.end_date, time.min)
        if payload.end_date is not None
        else None
    )

    # documento base (sem _id, created_at, data_hash ainda)
    base_doc = {
        "patient_id": patient_id,
        "medication_id": payload.medication_id,
        "medication_name": payload.medication_name,
        "dose": payload.dose,
        "route": payload.route,
        "frequency": payload.frequency,
        "scheduled_times": payload.scheduled_times,
        "start_date": start_dt,
        "end_date": end_dt,
        "status": payload.status,
        "notes": payload.notes,
        "created_by": payload.created_by,
    }

    # doc usado para hash: inclui prescription_id, mas ignora timestamps
    hash_payload = {
        "_id": payload.prescription_id,
        **base_doc,
    }
    data_hash = compute_data_hash(hash_payload)

    doc = {
        **base_doc,
        "updated_at": now,
        "data_hash": data_hash,
    }

    # upsert: cria se não existe, atualiza se já existir
    result = await db.prescriptions.update_one(
        {"_id": payload.prescription_id},
        {
            "$setOnInsert": {"created_at": now},
            "$set": doc,
        },
        upsert=True,
    )

    # --- Ledger local ---
    ledger_payload = {
        "_id": payload.prescription_id,
        **base_doc,
    }

    await append_to_ledger(
        db=db,
        entry_type="prescription",
        ref_id=payload.prescription_id,
        patient_id=patient_id,
        payload=ledger_payload,
        data_hash=data_hash,
    )

    return {
        "prescription_id": payload.prescription_id,
        "patient_id": patient_id,
        "status": "ok",
        "updated_at": now.isoformat(),
        "data_hash": data_hash,
    }


# ENDPOINT LISTAR PRESCRICOES

@app.get(
    "/api/v1/patients/{patient_id}/prescriptions",
    response_model=List[PrescriptionOut],
    summary="List prescriptions for a patient",
)
async def list_prescriptions_for_patient(patient_id: str):
    db = get_db()
    cursor = db.prescriptions.find({"patient_id": patient_id}).sort("created_at", 1)
    docs = await cursor.to_list(length=None)
    return [map_prescription_doc(d) for d in docs]


#ENDPOINT LIST INTAKE

@app.get(
    "/api/v1/patients/{patient_id}/intake-events",
    response_model=List[IntakeEventOut],
    summary="List medication intake events for a patient",
)
async def list_intake_events_for_patient(
    patient_id: str,
    limit: int = 100,
):
    db = get_db()
    cursor = (
        db.medication_intake_events
        .find({"patient_id": patient_id})
        .sort("intake_time", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=None)
    return [map_intake_event_doc(d) for d in docs]

#ENDPOINT DETALHES SECAO
@app.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionOut,
    summary="Get session details",
)
async def get_session(session_id: str):
    db = get_db()
    doc = await db.sessions.find_one({"_id": session_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return map_session_doc(doc)

@app.get(
    "/api/v1/sessions/{session_id}/intake-events",
    response_model=List[IntakeEventOut],
    summary="List intake events for a session",
)
async def list_intake_events_for_session(session_id: str):
    db = get_db()
    cursor = db.medication_intake_events.find({"session_id": session_id}).sort(
        "intake_time", 1
    )
    docs = await cursor.to_list(length=None)
    return [map_intake_event_doc(d) for d in docs]



@app.get(
    "/api/v1/patients/{patient_id}/adherence-summary",
    response_model=PatientAdherenceSummary,
    summary="Get adherence summary for a patient in the last N days",
)
async def get_patient_adherence_summary(
    patient_id: str,
    days: int = 7,
):
    """
    Retorna um resumo de adesão do paciente na janela dos últimos N dias (default 7).

    - window_start: hoje - (days-1)
    - window_end: hoje
    - Para cada prescrição ativa:
        * total_expected_doses
        * total_detected_doses
        * adherence
        * série diária (daily)
    """
    if days <= 0:
        raise HTTPException(status_code=400, detail="days must be >= 1")

    db = get_db()
    summary = await compute_adherence_summary(db, patient_id=patient_id, window_days=days)
    return summary


# -------------------------
# Post-processing stub
# -------------------------

async def run_post_processing(session_id: str):
    """
    Pós-processamento usando regras simbólicas de ingestão:

    - Carrega a sessão e os gesture_events associados.
    - Monta a sequência de labels de gestos (G1, G2, T, etc.).
    - Usa o PillIntakeEvaluator para encontrar candidatos a ingestão.
    - Para cada candidato com matched_rule != None e confidence > 0,
      cria um documento em 'medication_intake_events'.
    - Atualiza o status da sessão para 'processed'.
    """
    db = get_db()

    # 1) Carregar sessão
    session = await db.sessions.find_one({"_id": session_id})
    if not session:
        print(f"[POST-PROCESSING] Session not found: {session_id}")
        return

    patient_id = session["patient_id"]

    # 2) Carregar gesture_events da sessão, ordenados por timestamp
    cursor = db.gesture_events.find({"session_id": session_id}).sort("timestamp", 1)
    gestures = await cursor.to_list(length=None)

    if not gestures:
        print(f"[POST-PROCESSING] No gestures for session_id={session_id}")
        # Mesmo assim, marcamos como 'processed'
        await db.sessions.update_one(
            {"_id": session_id},
            {"$set": {"status": "processed", "updated_at": datetime.utcnow()}},
        )
        return

    # 3) Montar sequência de labels na ordem temporal
    gesture_sequence = [ (g.get("label") or "") for g in gestures ]

    # 4) Rodar o avaliador de ingestão
    evaluator = PillIntakeEvaluator(gesture_confidence=gesture_confidence, trigger="G1")
    # você pode ajustar window_size aqui se quiser
    eval_results = evaluator.evaluate_with_trigger(gesture_sequence, window_size=10)

    intake_events_docs = []
    now = datetime.utcnow()

    # 5) Remover candidatos redundantes (janelas que se sobrepõem)
    # Ordenamos por confiança decrescente para manter sempre o melhor dentro de uma região
    sorted_results = sorted(
        eval_results,
        key=lambda r: float(r.get("confidence", 0.0) or 0.0),
        reverse=True,
    )

    used_ranges: list[tuple[int, int]] = []  # lista de (start_idx, end_idx) já aceitos

    def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return not (a_end < b_start or b_end < a_start)

    # Converter resultados em documentos de MedicationIntakeEvent
    for res in sorted_results:
        matched_rule = res.get("matched_rule")
        conf = float(res.get("confidence", 0.0) or 0.0)
        window_size = res.get("window_size", 10)
        trigger_index = res.get("trigger_index", 0)

        # Se nenhuma regra foi casada, ignoramos
        if matched_rule is None or conf <= 0.0:
            continue

        # Índices da janela considerada (gatilho até gatilho+window_size-1)
        start_idx = trigger_index
        end_idx = min(trigger_index + window_size - 1, len(gestures) - 1)

        # Verificar se essa janela se sobrepõe a alguma já aceita
        has_overlap = any(
            overlaps(start_idx, end_idx, u_start, u_end)
            for (u_start, u_end) in used_ranges
        )
        if has_overlap:
            # Já temos um evento nesta região com confiança maior ou igual
            continue

        # Marcar essa janela como usada
        used_ranges.append((start_idx, end_idx))

        start_ts = gestures[start_idx]["timestamp"]
        end_ts = gestures[end_idx]["timestamp"]
        intake_time = start_ts + (end_ts - start_ts) / 2

        # Tentar associar com a melhor prescrição
        prescription_id, medication_id = await find_best_prescription_for_intake(
            db=db,
            patient_id=patient_id,
            intake_time=intake_time,
        )

        base_doc = {
            # deixamos o _id para o Mongo gerar
            "session_id": session_id,
            "patient_id": patient_id,
            "prescription_id": prescription_id,
            "medication_id": medication_id,
            "intake_time": intake_time,
            "confidence": conf,
            "detected_by": "pill_intake_rules_v1.0",
            "source_gestures": {
                "start_time": start_ts,
                "end_time": end_ts,
                "trigger_index": trigger_index,
                "window_size": window_size,
                "matched_rule": list(matched_rule),
                "window_ids": [
                    gestures[i]["window_id"]
                    for i in range(start_idx, end_idx + 1)
                    if "window_id" in gestures[i]
                ],
            },
            "status": "auto_detected",
            "created_at": now,
            "updated_at": now,
        }

        # payload usado para hash (sem created_at/updated_at, sem _id)
        hash_payload = base_doc.copy()
        data_hash = compute_data_hash(hash_payload)

        intake_doc = {
            **base_doc,
            "data_hash": data_hash,
            "created_at": now,
            "updated_at": now,
        }

        intake_events_docs.append(intake_doc)

    # 6) Inserir intake events, se houver
    if intake_events_docs:
        result = await db.medication_intake_events.insert_many(intake_events_docs)
        print(
            f"[POST-PROCESSING] Inserted {len(result.inserted_ids)} "
            f"intake event(s) for session_id={session_id}"
        )
        # log pré-blockchain para cada evento
        for _id, doc in zip(result.inserted_ids, intake_events_docs):
            ref_id = str(_id)
            patient_id_ev = doc.get("patient_id")
            data_hash_ev = doc.get("data_hash")

            ledger_payload = {
                "_id": ref_id,
                "session_id": doc.get("session_id"),
                "patient_id": patient_id_ev,
                "prescription_id": doc.get("prescription_id"),
                "medication_id": doc.get("medication_id"),
                "intake_time": doc.get("intake_time"),
                "confidence": doc.get("confidence"),
                "detected_by": doc.get("detected_by"),
                "status": doc.get("status"),
            }

            await append_to_ledger(
                db=db,
                entry_type="intake_event",
                ref_id=ref_id,
                patient_id=patient_id_ev,
                payload=ledger_payload,
                data_hash=data_hash_ev,
            )
    else:
        print(f"[POST-PROCESSING] No intake events detected for session_id={session_id}")

    # 7) Atualizar status da sessão para 'processed'
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"status": "processed", "updated_at": now}},
    )


async def find_best_prescription_for_intake(
    db,
    patient_id: str,
    intake_time: datetime,
) -> tuple[Optional[str], Optional[str]]:
    """
    Encontra a melhor prescrição ativa para o paciente naquele horário.

    Estratégia simples:
    - Busca prescrições com status 'active' do paciente.
    - Filtra por start_date <= intake_date <= end_date (se end_date não for None).
    - Para cada prescrição, calcula a menor diferença (em minutos) entre intake_time
      e cada horário em scheduled_times (HH:MM).
    - Escolhe a prescrição com menor diferença.
    - Se não encontrar nenhuma, retorna (None, None).
    """

    intake_date = intake_time.date()
    intake_tod = intake_time.time()

    # Carregar prescrições ativas
    cursor = db.prescriptions.find(
        {"patient_id": patient_id, "status": "active"}
    )
    prescriptions = await cursor.to_list(length=None)

    if not prescriptions:
        return None, None

    best_prescription = None
    best_delta_minutes = None

    for p in prescriptions:
        start_date = p.get("start_date")
        end_date = p.get("end_date")

        # converter datetime -> date se necessário
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if start_date and intake_date < start_date:
            continue
        if end_date and intake_date > end_date:
            continue

        times = p.get("scheduled_times") or []
        if not times:
            continue

        # calcula a menor diferença, em minutos, entre intake_time e os horários da prescrição
        for hhmm in times:
            try:
                hour, minute = map(int, hhmm.split(":"))
            except Exception:
                continue

            sched_dt = datetime.combine(intake_date, datetime.min.time()).replace(
                hour=hour, minute=minute
            )
            delta = abs((intake_time - sched_dt).total_seconds()) / 60.0  # minutos

            if (best_delta_minutes is None) or (delta < best_delta_minutes):
                best_delta_minutes = delta
                best_prescription = p

    if best_prescription is None:
        return None, None

    return best_prescription["_id"], best_prescription.get("medication_id")

# -------------------------
# Módulo de integridade / pré-blockchain
# -------------------------


def log_prescription_to_ledger(prescription_doc: dict):
    """
    Stub para futura integração com blockchain.

    Hoje:
    - Apenas imprime no log o _id e o data_hash.
    Futuro:
    - Enviar transação para blockchain com (prescription_id, data_hash, timestamp, etc.)
    """
    pid = prescription_doc.get("_id")
    data_hash = prescription_doc.get("data_hash")
    print(f"[LEDGER] Prescription hash={data_hash}")


def log_intake_event_to_ledger(intake_doc: dict):
    """
    Stub para futura integração com blockchain.

    Hoje:
    - Apenas imprime no log o _id (se existir) e o data_hash.
    Futuro:
    - Enviar transação para blockchain com (session_id, patient_id, prescription_id,
      intake_time, data_hash, etc.)
    """
    eid = intake_doc.get("_id")
    data_hash = intake_doc.get("data_hash")
    print(f"[LEDGER] IntakeEvent {eid} hash={data_hash}")