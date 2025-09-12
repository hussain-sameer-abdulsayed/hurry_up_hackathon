from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.finger_print import CreationResponse, FingerPrintRead, ResponseStatus, VerifiyFingerResponse
from app.services.fingerprint_service import FingerPrintService

router = APIRouter(responses= {404 : {"description":"Not found"}})

@router.post("/save", status_code= status.HTTP_201_CREATED)
async def save_fingerprint(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)) -> CreationResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    service = FingerPrintService(db)
    content = await file.read()
    fingerprint = await service.save_fingerprint(content, file.filename)

    return CreationResponse(
        status= ResponseStatus.success,
        message= "Fingerprint saved successfully",
        fingerprint_id= str(fingerprint.id),
        minutiae_count= fingerprint.minutiae_count,
        quality_score= fingerprint.quality_score,
        enrollment_timestamp= fingerprint.created_at
    )

@router.post("/check", status_code= status.HTTP_200_OK)
async def verify_fingerprint(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)) -> VerifiyFingerResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    service = FingerPrintService(db)
    content = await file.read()
    result = await service.verify_fingerprint(content)
    
    return result


@router.get("/list", status_code= status.HTTP_200_OK)
async def list_fingerprints(db: AsyncSession = Depends(get_db)) -> List[FingerPrintRead]:
    service = FingerPrintService(db)
    fingerprints = await service.list_fingerprints()
    
    return fingerprints

@router.delete("/clear", status_code= status.HTTP_204_NO_CONTENT)
async def clear_all_fingerprints(db: AsyncSession = Depends(get_db)) -> int:
    """Clear all fingerprints from database"""
    service = FingerPrintService(db)
    deleted_count = await service.clear_all_fingerprints()
    
    return deleted_count