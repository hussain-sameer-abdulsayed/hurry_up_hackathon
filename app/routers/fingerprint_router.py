from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.fingerprint_service import FingerPrintService

router = APIRouter(prefix="/fingerprints", tags=["Fingerprints"])

@router.post("/save")
async def save_fingerprint(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    service = FingerPrintService(db)
    content = await file.read()
    fingerprint = await service.save_fingerprint(content, file.filename)

    return JSONResponse({
        "status": "success",
        "message": "Fingerprint saved successfully",
        "fingerprint_id": str(fingerprint.id),
        "minutiae_count": fingerprint.minutiae_count,
        "quality_score": fingerprint.quality_score,
        "enrollment_timestamp": fingerprint.created_at.isoformat()
    })

@router.post("/check")
async def verify_fingerprint(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    service = FingerPrintService(db)
    content = await file.read()
    result = await service.verify_fingerprint(content)
    
    return JSONResponse(result)

@router.get("/list")
async def list_fingerprints(db: AsyncSession = Depends(get_db)):
    service = FingerPrintService(db)
    fingerprints = await service.list_fingerprints()
    
    return JSONResponse({
        "status": "success",
        "total_fingerprints": len(fingerprints),
        "fingerprints": [
            {
                "id": str(fp.id),
                "minutiae_count": fp.minutiae_count,
                "quality_score": fp.quality_score,
                "created_at": fp.created_at.isoformat()
            }
            for fp in fingerprints
        ]
    })