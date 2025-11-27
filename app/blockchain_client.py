import os
import json
from datetime import datetime
from typing import Optional

from web3 import Web3

# URL do nó Ethereum (Ganache/Hardhat, ou outro)
WEB3_HTTP_PROVIDER = os.getenv("WEB3_HTTP_PROVIDER", "http://127.0.0.1:8545")
LEDGER_CONTRACT_ADDRESS = os.getenv("LEDGER_CONTRACT_ADDRESS", "")
LEDGER_CONTRACT_ABI_PATH = os.getenv(
    "LEDGER_CONTRACT_ABI_PATH",
    "app/MedicationLedger.abi.json",
)
DEFAULT_SENDER = os.getenv("LEDGER_SENDER_ADDRESS", "")

_w3: Optional[Web3] = None
_contract = None


def _init_web3():
    global _w3, _contract
    if _w3 is not None and _contract is not None:
        return

    _w3 = Web3(Web3.HTTPProvider(WEB3_HTTP_PROVIDER))
    if not _w3.is_connected():
        print(f"[BLOCKCHAIN] Warning: cannot connect to {WEB3_HTTP_PROVIDER}")
        return

    if not LEDGER_CONTRACT_ADDRESS:
        print("[BLOCKCHAIN] Warning: LEDGER_CONTRACT_ADDRESS not set")
        return

    try:
        with open(LEDGER_CONTRACT_ABI_PATH, "r") as f:
            abi = json.load(f)
    except FileNotFoundError:
        print(f"[BLOCKCHAIN] Warning: ABI file not found at {LEDGER_CONTRACT_ABI_PATH}")
        return

    _contract = _w3.eth.contract(
        address=_w3.to_checksum_address(LEDGER_CONTRACT_ADDRESS),
        abi=abi,
    )


def _to_timestamp(dt: datetime) -> int:
    # segundos desde epoch
    return int(dt.timestamp())


def log_prescription_to_chain(prescription_doc: dict):
    """
    Envia o evento de prescrição para a blockchain.

    Se não conseguir conectar ou faltar config, apenas loga um aviso e retorna.
    """
    _init_web3()
    if _w3 is None or _contract is None:
        # fallback: sem falhar o backend
        print("[BLOCKCHAIN] log_prescription_to_chain: blockchain not configured")
        return

    if not DEFAULT_SENDER:
        print("[BLOCKCHAIN] Warning: LEDGER_SENDER_ADDRESS not set")
        return

    pid = str(prescription_doc.get("_id"))
    patient_id = prescription_doc.get("patient_id") or ""
    medication_id = prescription_doc.get("medication_id") or ""
    start_date = prescription_doc.get("start_date")
    end_date = prescription_doc.get("end_date")
    data_hash = prescription_doc.get("data_hash")

    if isinstance(start_date, datetime):
        start_ts = _to_timestamp(start_date)
    else:
        start_ts = 0

    if isinstance(end_date, datetime):
        end_ts = _to_timestamp(end_date)
    else:
        end_ts = 0

    # data_hash deve ser bytes32; se estiver como hex string, converte
    if isinstance(data_hash, str):
        # tenta interpretar como hex
        try:
            data_hash_bytes = bytes.fromhex(data_hash)
        except ValueError:
            # se não der, usa hash dos bytes da string
            data_hash_bytes = Web3.keccak(text=data_hash)
    else:
        # fallback
        data_hash_bytes = Web3.keccak(text=str(data_hash))

    tx = _contract.functions.logPrescription(
        pid,
        patient_id,
        medication_id,
        start_ts,
        end_ts,
        data_hash_bytes,
    ).build_transaction(
        {
            "from": DEFAULT_SENDER,
            "nonce": _w3.eth.get_transaction_count(DEFAULT_SENDER),
            "gas": 300_000,
            "gasPrice": _w3.eth.gas_price,
        }
    )

    signed = _w3.eth.account.sign_transaction(
        tx,
        private_key=os.getenv("LEDGER_SENDER_PRIVATE_KEY", ""),
    )
    tx_hash = _w3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"[BLOCKCHAIN] Prescription tx sent: {tx_hash.hex()}")


def log_intake_event_to_chain(intake_doc: dict):
    """
    Envia o evento de ingestão para a blockchain.
    """
    _init_web3()
    if _w3 is None or _contract is None:
        print("[BLOCKCHAIN] log_intake_event_to_chain: blockchain not configured")
        return

    if not DEFAULT_SENDER:
        print("[BLOCKCHAIN] Warning: LEDGER_SENDER_ADDRESS not set")
        return

    eid = str(intake_doc.get("_id"))
    session_id = intake_doc.get("session_id") or ""
    patient_id = intake_doc.get("patient_id") or ""
    prescription_id = intake_doc.get("prescription_id") or ""
    intake_time = intake_doc.get("intake_time")
    confidence = float(intake_doc.get("confidence", 0.0) or 0.0)
    data_hash = intake_doc.get("data_hash")

    if isinstance(intake_time, datetime):
        intake_ts = _to_timestamp(intake_time)
    else:
        intake_ts = 0

    # codifica confiança como inteiro multiplicado por 1000
    conf1000 = int(max(0.0, min(confidence, 1.0)) * 1000)

    if isinstance(data_hash, str):
        try:
            data_hash_bytes = bytes.fromhex(data_hash)
        except ValueError:
            data_hash_bytes = Web3.keccak(text=data_hash)
    else:
        data_hash_bytes = Web3.keccak(text=str(data_hash))

    tx = _contract.functions.logIntakeEvent(
        eid,
        session_id,
        patient_id,
        prescription_id,
        intake_ts,
        conf1000,
        data_hash_bytes,
    ).build_transaction(
        {
            "from": DEFAULT_SENDER,
            "nonce": _w3.eth.get_transaction_count(DEFAULT_SENDER),
            "gas": 300_000,
            "gasPrice": _w3.eth.gas_price,
        }
    )

    signed = _w3.eth.account.sign_transaction(
        tx,
        private_key=os.getenv("LEDGER_SENDER_PRIVATE_KEY", ""),
    )
    tx_hash = _w3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"[BLOCKCHAIN] IntakeEvent tx sent: {tx_hash.hex()}")
