import cv2
import numpy as np
import base64
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.repos.fingerprint_repo import FingerPrintRepository
from app.schemas.finger_print import FingerPrintCreate, FingerPrintRead, VerifiyFingerResponse
from app.models.finger_print import FingerPrint


class FingerPrintService:
    def __init__(self, db: AsyncSession):
        self.repo = FingerPrintRepository(db)

    # ----------------------
    # Image Processing
    # ----------------------
    def capture_from_upload(self, file_bytes: bytes) -> np.ndarray:
        """Read uploaded image into grayscale OpenCV matrix"""
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file")
        return img

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize, normalize, and binarize"""
        img = cv2.resize(img, (300, 300))
        img = cv2.equalizeHist(img)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary

    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for ridge enhancement"""
        return cv2.GaussianBlur(img, (3, 3), 0)

    def extract_minutiae_features(self, img: np.ndarray) -> List[tuple]:
        """Dummy feature extraction → returns nonzero coordinates"""
        coords = np.column_stack(np.where(img > 0))
        return coords.tolist()

    def create_template(self, minutiae: List[tuple]) -> str:
        """Serialize minutiae into Base64 string"""
        return base64.b64encode(str(minutiae).encode()).decode()

    def calculate_quality_score(self, minutiae: List[tuple]) -> float:
        """Simple heuristic: more minutiae = better quality"""
        return round(min(len(minutiae) / 500, 1.0), 2)

    # ----------------------
    # Database Operations
    # ----------------------
    async def save_fingerprint(self, file_bytes: bytes, filename: Optional[str] = None) -> FingerPrint:
        """Process and save fingerprint"""
        img = self.capture_from_upload(file_bytes)
        preprocessed = self.preprocess_image(img)
        enhanced = self.enhance_image(preprocessed)
        minutiae = self.extract_minutiae_features(enhanced)
        template = self.create_template(minutiae)
        quality = self.calculate_quality_score(minutiae)

        fp = await self.repo.create(FingerPrintCreate(
            template_data=template,
            minutiae_count=len(minutiae),
            quality_score=quality,
            filename=filename
        ))
        return fp

    async def list_fingerprints(self) -> List[FingerPrintRead]:
        fps = await self.repo.get_all()
        return [FingerPrintRead.model_validate(fp) for fp in fps]

    async def verify_fingerprint(self, file_bytes: bytes, threshold: float = 0.8) -> VerifiyFingerResponse:
        """Compare uploaded fingerprint with stored ones"""
        img = self.capture_from_upload(file_bytes)
        preprocessed = self.preprocess_image(img)
        enhanced = self.enhance_image(preprocessed)
        minutiae = self.extract_minutiae_features(enhanced)
        template = self.create_template(minutiae)

        stored = await self.repo.get_all()
        for fp in stored:
            similarity = self.compare_templates(template, fp.template_data)
            if similarity >= threshold:
                return VerifiyFingerResponse(
                    verified= True,
                    matched_fingerprint_id= str(fp.id),
                    confidence_score= similarity,
                    message= "Fingerprint verified successfully"
                )

        return VerifiyFingerResponse(
            verified= False,
            confidence_score= 0.0,
            message= "No matching fingerprint found"
        )

    # ----------------------
    # Matching Algorithm
    # ----------------------
    def compare_templates(self, t1: str, t2: str) -> float:
        """Naive comparison: count overlap of encoded minutiae strings"""
        set1 = set(t1.split(","))
        set2 = set(t2.split(","))
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)


    async def clear_all_fingerprints(self) -> int:
        """Clear all fingerprints from database"""
        return await self.repo.delete_all()