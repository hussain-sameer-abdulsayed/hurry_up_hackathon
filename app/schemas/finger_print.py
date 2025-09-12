from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class FingerPrintCreate(BaseModel):
    template_data: str
    minutiae_count: int
    quality_score: float
    filename: Optional[str] = None

class FingerPrintRead(BaseModel):
    id: str
    minutiae_count: int
    quality_score: float
    enrollment_timestamp: datetime
    filename: Optional[str]

class VerificationResponse(BaseModel):
    verified: bool
    matched_fingerprint_id: Optional[str] = None
    confidence_score: float
    message: str
