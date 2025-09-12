

from fastapi import APIRouter, Depends, UploadFile, responses, status

from app.schemas.finger_print import VerificationResponse



router = APIRouter(
   prefix="/fingerprints",
   tags=["fingerprints"],
   responses={404:{"description":"Not found"}}
)


@router.post("/create", response_model= VerificationResponse, status_code= status.HTTP_201_CREATED)
async def create(file: UploadFile):
   if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process file
        content = await file.read()
        img = processor.capture_from_upload(content)
        preprocessed = processor.preprocess_image(img)
        enhanced = processor.enhance_image(preprocessed)
        minutiae = processor.extract_minutiae_features(enhanced)
        template = processor.create_template(minutiae)
        
        # Generate unique ID and save
        fingerprint_id = str(uuid.uuid4())
        processor.fingerprints_db[fingerprint_id] = {
            'template': template,
            'enrollment_timestamp': datetime.now().isoformat(),
            'minutiae_count': len(minutiae),
            'quality_score': processor.calculate_quality_score(minutiae),
            'filename': file.filename
        }