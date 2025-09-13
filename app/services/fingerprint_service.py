import cv2
import numpy as np
import base64
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
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file")
        return img

    def preprocess_image_enhanced(self, img: np.ndarray, source_type: str = "unknown") -> np.ndarray:
        img = cv2.resize(img, (300, 300))
        img = cv2.equalizeHist(img)
        return img

    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (3, 3), 0)

    # ----------------------
    # Feature Extraction
    # ----------------------
    def extract_minutiae_features(self, img: np.ndarray) -> List[np.ndarray]:
        """Use ORB feature extractor to get keypoints & descriptors"""
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None:
            return []
        return descriptors.tolist()

    def create_template(self, minutiae: List[np.ndarray]) -> str:
        """Serialize descriptors as Base64"""
        if not minutiae:
            return base64.b64encode("EMPTY".encode()).decode()
        arr = np.array(minutiae, dtype=np.uint8)
        return base64.b64encode(arr.tobytes()).decode()

    def calculate_quality_score(self, minutiae: List[np.ndarray]) -> float:
        """Quality = ratio of features detected"""
        return round(min(len(minutiae) / 500, 1.0), 2)

    # ----------------------
    # Database Operations
    # ----------------------
    async def save_fingerprint(self, file_bytes: bytes, filename: Optional[str] = None) -> FingerPrint:
        img = self.capture_from_upload(file_bytes)
        preprocessed = self.preprocess_image_enhanced(img)
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

    async def list_fingerprints(self) -> List[FingerPrint]:
        return await self.repo.get_all()

    async def verify_fingerprint(self, file_bytes: bytes, threshold: float = 0.35) -> VerifiyFingerResponse:
        import time
        
        # Start timing
        start_time = time.time()
        
        img = self.capture_from_upload(file_bytes)
        preprocessed = self.preprocess_image_enhanced(img)
        enhanced = self.enhance_image(preprocessed)
        minutiae = self.extract_minutiae_features(enhanced)
        template = self.create_template(minutiae)

        stored = await self.repo.get_all()
        
        # Track best match even if below threshold
        best_match = None
        best_score = 0.0
        best_fp_id = None
        
        for fp in stored:
            similarity = self.compare_templates_improved(template, fp.template_data)
            if similarity > best_score:
                best_score = similarity
                best_match = fp
                best_fp_id = str(fp.id)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        
        # Check if best match meets threshold
        if best_score >= threshold and best_match:
            return VerifiyFingerResponse(
                verified=True,
                matched_fingerprint_id=best_fp_id,
                confidence_score=best_score,
                message=f"Match found with {best_score:.1%} confidence (processed in {processing_time}ms)"
            )
        elif best_match:
            # Return closest match even if below threshold
            return VerifiyFingerResponse(
                verified=False,
                matched_fingerprint_id=best_fp_id,
                confidence_score=best_score,
                message=f"No match above threshold. Closest match: {best_score:.1%} confidence (processed in {processing_time}ms)"
            )
        else:
            return VerifiyFingerResponse(
                verified=False,
                matched_fingerprint_id=None,
                confidence_score=0.0,
                message=f"No fingerprints in database (processed in {processing_time}ms)"
            )

    # ----------------------
    # Matching Algorithm
    # ----------------------
    def compare_templates_improved(self, t1: str, t2: str) -> float:
        """Compare ORB descriptors with BFMatcher"""
        try:
            if t1 == "EMPTY" or t2 == "EMPTY":
                return 0.0

            d1 = np.frombuffer(base64.b64decode(t1), dtype=np.uint8)
            d2 = np.frombuffer(base64.b64decode(t2), dtype=np.uint8)

            if d1.size == 0 or d2.size == 0:
                return 0.0

            # Reshape into descriptor vectors of ORB (32-length each)
            d1 = d1.reshape(-1, 32)
            d2 = d2.reshape(-1, 32)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(d1, d2)
            if not matches:
                return 0.0

            good_matches = [m for m in matches if m.distance < 50]  # lower distance = better match
            score = len(good_matches) / max(len(matches), 1)

            return round(score, 2)

        except Exception:
            return 0.0

    async def clear_all_fingerprints(self) -> int:
        return await self.repo.delete_all()
