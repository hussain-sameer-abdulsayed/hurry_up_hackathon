from fastapi import APIRouter, FastAPI

from .fingerprint_router import router as fingerprint_router

api_router = APIRouter()


api_router.include_router(
   fingerprint_router,
   prefix="/fingerprints", 
   tags=["Fingerprints"]
)