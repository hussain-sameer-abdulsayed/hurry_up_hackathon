from fastapi import Depends, FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel
import uvicorn
from app.database import engine, get_db
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Import the fingerprint matcher
from fingerprint_matcher import FingerprintMatcher, load_image_from_bytes, load_image_from_path

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global fingerprint matcher instance
matcher = FingerprintMatcher()

# Thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    # Initialize fingerprint matcher
    logger.info("Initializing fingerprint matcher...")
    global matcher
    matcher = FingerprintMatcher(distance_threshold=0.7, geometric_threshold=5.0)
    logger.info("Fingerprint matcher initialized successfully")
    
    yield
    
    # Cleanup
    thread_pool.shutdown(wait=True)
    await engine.dispose()

app = FastAPI(
    title=os.getenv("APP_NAME", "Hurry Up Hackathon"),
    description="Best Team - Fingerprint Matching API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validation functions
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, BMP, etc.)"
        )
    
    # Check file size (10MB limit)
    if hasattr(file, 'size') and file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail="File size too large. Maximum 10MB allowed."
        )

async def run_cpu_bound_task(func, *args):
    """Run CPU-bound tasks in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args)

@app.get("/")
async def root():
    return {
        "Hurry Up Hackathon": "Best Team Ever! ok?",
        "message": "Fingerprint Matching API",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0",
        "environment": os.getenv("APP_ENV", "development"),
        "endpoints": {
            "health": "/health",
            "match_fingerprints": "/match-fingerprints",
            "match_database": "/match-database",
            "extract_features": "/extract-features"
        }
    }

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    return {
        "status": "healthy", 
        "database": "connected",
        "fingerprint_matcher": "ready"
    }

@app.post("/match-fingerprints")
async def match_two_fingerprints(
    sample_file: UploadFile,
    candidate_file: UploadFile,
    db: AsyncSession = Depends(get_db)
):
    """
    Compare two fingerprint images and return similarity metrics
    """
    try:
        # Validate both files
        validate_image_file(sample_file)
        validate_image_file(candidate_file)
        
        # Read file contents
        sample_bytes = await sample_file.read()
        candidate_bytes = await candidate_file.read()
        
        # Load images in thread pool
        sample_image = await run_cpu_bound_task(load_image_from_bytes, sample_bytes)
        candidate_image = await run_cpu_bound_task(load_image_from_bytes, candidate_bytes)
        
        if sample_image is None:
            raise HTTPException(status_code=400, detail="Could not load sample image")
        
        if candidate_image is None:
            raise HTTPException(status_code=400, detail="Could not load candidate image")
        
        # Perform matching in thread pool
        result = await run_cpu_bound_task(
            matcher.match_fingerprints, 
            sample_image, 
            candidate_image
        )
        
        # Add file information to result
        result.update({
            "sample_filename": sample_file.filename,
            "candidate_filename": candidate_file.filename,
            "sample_size": len(sample_bytes),
            "candidate_size": len(candidate_bytes)
        })
        
        logger.info(f"Fingerprint matching completed: {result.get('quality', 'UNKNOWN')} match")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fingerprint matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/match-database")
async def match_against_database(
    sample_file: UploadFile,
    database_path: Optional[str] = None,
    max_files: int = 1000,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """
    Match uploaded fingerprint against a database of fingerprints
    """
    try:
        # Validate file
        validate_image_file(sample_file)
        
        # Use default database path if not provided
        if not database_path:
            database_path = os.getenv("FINGERPRINT_DB_PATH", "SOCOFing/Real")
        
        # Check if database path exists
        if not os.path.exists(database_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Database directory not found: {database_path}"
            )
        
        # Read and load sample image
        sample_bytes = await sample_file.read()
        sample_image = await run_cpu_bound_task(load_image_from_bytes, sample_bytes)
        
        if sample_image is None:
            raise HTTPException(status_code=400, detail="Could not load sample image")
        
        # Progress tracking (optional - could be enhanced with WebSocket)
        progress_info = {"current": 0, "total": 0, "best_match": None}
        
        def progress_callback(progress, filename, best_match):
            progress_info.update({
                "current": int(progress),
                "current_file": filename,
                "best_match": best_match
            })
        
        # Perform database matching in thread pool
        result = await run_cpu_bound_task(
            matcher.match_against_database,
            sample_image,
            database_path,
            max_files,
            progress_callback
        )
        
        # Add request information
        result.update({
            "sample_filename": sample_file.filename,
            "sample_size": len(sample_bytes),
            "database_path": database_path,
            "max_files_processed": max_files
        })
        
        if result.get("best_match"):
            logger.info(f"Database search completed. Best match: {result['best_match']['filename']} "
                       f"(Quality: {result['best_match']['quality']})")
        else:
            logger.info("Database search completed. No suitable matches found.")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in database matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract-features")
async def extract_fingerprint_features(
    file: UploadFile,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract SIFT features from a fingerprint image
    """
    try:
        # Validate file
        validate_image_file(file)
        
        # Read and load image
        image_bytes = await file.read()
        image = await run_cpu_bound_task(load_image_from_bytes, image_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")
        
        # Extract features in thread pool
        keypoints, descriptors = await run_cpu_bound_task(
            matcher.extract_features, 
            image
        )
        
        if keypoints is None or descriptors is None:
            return JSONResponse(content={
                "success": False,
                "error": "Could not extract features from image",
                "filename": file.filename
            })
        
        result = {
            "success": True,
            "filename": file.filename,
            "file_size": len(image_bytes),
            "keypoints_count": len(keypoints),
            "descriptors_shape": descriptors.shape if descriptors is not None else None,
            "image_dimensions": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        }
        
        logger.info(f"Feature extraction completed: {result['keypoints_count']} keypoints found")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Legacy endpoint (keeping for compatibility)
@app.post("/")
async def upload_file(file: UploadFile):
    """Legacy file upload endpoint"""
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully. Use /match-fingerprints or /match-database for fingerprint matching."
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("APP_ENV") != "production"
    )
