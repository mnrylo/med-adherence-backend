from pydantic import BaseModel, Field
from datetime import datetime, time, date, timedelta
from typing import List, Optional

# -------------------------
# Pydantic models (API)
# -------------------------

class GestureIn(BaseModel):
    timestamp: datetime = Field(..., description="Center timestamp of the classified window")
    window_id: int = Field(..., ge=0, description="Sequential window index within the session")
    label: str = Field(..., description="Predicted gesture label, e.g., G1, G2, Idle, T")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional confidence score for this prediction"
    )

class PrescriptionIn(BaseModel):
    prescription_id: str = Field(..., description="Unique prescription identifier (will be used as _id)")
    medication_id: str = Field(..., description="Medication identifier (e.g., METFORMIN_500MG)")
    medication_name: str = Field(..., description="Medication common name (for display)")
    dose: str = Field(..., description="Dose description, e.g., '500 mg'")
    route: str = Field(..., description="Administration route, e.g., 'oral'")
    frequency: str = Field(..., description="Human-readable frequency, e.g., '2x/day'")
    scheduled_times: List[str] = Field(
        ..., description="List of times in HH:MM for expected doses, e.g., ['08:00','20:00']"
    )
    start_date: date = Field(..., description="Prescription start date")
    end_date: Optional[date] = Field(None, description="Prescription end date (optional)")
    status: str = Field("active", description="active, paused, finished, etc.")
    notes: Optional[str] = Field(None, description="Optional notes")
    created_by: Optional[str] = Field(None, description="Doctor / prescriber identifier")

class SessionPayload(BaseModel):
    patient_id: str = Field(..., description="Patient pseudonymous identifier")
    session_id: str = Field(..., description="Session identifier, should match path param")
    phone_id: str = Field(..., description="Identifier of the phone running the model")
    model_version: str = Field(..., description="Model version used for classification")
    start_time: datetime = Field(..., description="Session start time (ISO 8601)")
    end_time: datetime = Field(..., description="Session end time (ISO 8601)")
    gestures: List[GestureIn] = Field(
        ..., description="List of classified gesture events for this session"
    )

class IngestResponse(BaseModel):
    session_id: str
    inserted_gestures: int
    post_processing_triggered: bool

class PrescriptionOut(BaseModel):
    prescription_id: str
    patient_id: str
    medication_id: str
    medication_name: str
    dose: str
    route: str
    frequency: str
    scheduled_times: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    status: str
    notes: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class IntakeEventOut(BaseModel):
    event_id: str
    session_id: str
    patient_id: str
    prescription_id: Optional[str] = None
    medication_id: Optional[str] = None
    intake_time: datetime
    confidence: float
    detected_by: str
    status: str
    source_gestures: dict
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SessionOut(BaseModel):
    session_id: str
    patient_id: str
    watch_id: Optional[str] = None
    phone_id: Optional[str] = None
    start_time: datetime
    end_time: datetime
    status: str
    model_version: str
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DailyAdherenceItem(BaseModel):
    date: date
    expected_doses: int
    detected_doses: int
    adherence: Optional[float]  # None se expected == 0

class PrescriptionAdherenceSummary(BaseModel):
    prescription_id: str
    medication_id: Optional[str]
    medication_name: Optional[str]
    start_date: date
    end_date: Optional[date]
    total_expected_doses: int
    total_detected_doses: int
    adherence: Optional[float]
    daily: List[DailyAdherenceItem]

class PatientAdherenceSummary(BaseModel):
    patient_id: str
    window_start: date
    window_end: date
    prescriptions: List[PrescriptionAdherenceSummary]
