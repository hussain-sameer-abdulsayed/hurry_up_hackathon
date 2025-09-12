from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.repos.fingerprint_repo import FingerPrintRepository
from app.services.fingerprint_processor import FingerprintProcessor
from app.schemas.finger_print import FingerPrintCreate

import uuid
from datetime import datetime

router = APIRouter(prefix="/fingerprints", tags=["Fingerprints"])

processor = FingerprintProcessor()

@router.post("/save")
async def save_fingerprint(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    img = processor.capture_from_upload(content)
    preprocessed = processor.preprocess_image(img)
    enhanced = processor.enhance_image(preprocessed)
    minutiae = processor.extract_minutiae_features(enhanced)
    template = processor.create_template(minutiae)

    repo = FingerPrintRepository(db)
    fingerprint = await repo.create(FingerPrintCreate(
        template_data=template,
        minutiae_count=len(minutiae),
        quality_score=processor.calculate_quality_score(minutiae),
        filename=file.filename
    ))

    return JSONResponse({
        "status": "success",
        "message": "Fingerprint saved successfully",
        "fingerprint_id": fingerprint.id,
        "minutiae_count": fingerprint.minutiae_count,
        "quality_score": fingerprint.quality_score,
        "enrollment_timestamp": fingerprint.enrollment_timestamp.isoformat()
    })

@router.get("/list")
async def list_fingerprints(db: AsyncSession = Depends(get_db)):
    repo = FingerPrintRepository(db)
    fingerprints = await repo.get_all()
    return fingerprints
