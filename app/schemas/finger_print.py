from enum import Enum
from typing import Optional
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime


class ResponseStatus(str, Enum):
    success = "success"
    unsuccess = "unsuccess"

    def __str__(self):
        return self.value


class FingerPrintCreate(BaseModel):
    template_data: str
    minutiae_count: int
    quality_score: float
    filename: Optional[str] = None

class FingerPrintRead(BaseModel):
    id: UUID
    minutiae_count: int
    quality_score: float
    enrollment_timestamp: datetime
    filename: Optional[str]

class VerificationResponse(BaseModel):
    verified: bool
    matched_fingerprint_id: Optional[str] = None
    confidence_score: float
    message: str


class CreationResponse(BaseModel):
    status: ResponseStatus
    fingerprint_id: Optional[str]
    minutiae_count: Optional[int]
    quality_score: Optional[float]
    enrollment_timestamp: Optional[datetime]


class VerifiyFingerResponse(BaseModel):
    verified: bool
    matched_fingerprint_id: Optional[str]
    confidence_score: Optional[str]
    message: str

    """
    "status": "success",
        "message": "Fingerprint saved successfully",
        "fingerprint_id": str(fingerprint.id),
        "minutiae_count": fingerprint.minutiae_count,
        "quality_score": fingerprint.quality_score,
        "enrollment_timestamp": fingerprint.created_at.isoformat()
    """

