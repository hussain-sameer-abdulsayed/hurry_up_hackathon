from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel


class FingerPrint(SQLModel, table=True):
    __tablename__ = "fingerprints"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    template_data: str
    minutiae_count: int
    quality_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    filename: str | None = None