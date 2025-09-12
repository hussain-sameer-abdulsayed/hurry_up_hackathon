#!/usr/bin/env python3
# fingerprint_matcher.py
import cv2
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union
from scipy import ndimage
from skimage import morphology, filters

class FingerprintMatcher:
    """
    Fingerprint matching class optimized for FastAPI backend usage
    """
    
    def __init__(self, distance_threshold: float = 0.75, geometric_threshold: float = 3.0):
        """
        Initialize the fingerprint matcher
        
        Args:
            distance_threshold: Threshold for Lowe's ratio test (optimized for better accuracy)
            geometric_threshold: Threshold for geometric verification (tighter for better precision)
        """
        self.distance_threshold = distance_threshold
        self.geometric_threshold = geometric_threshold
        
        # Enhanced SIFT with more features for better matching
        self.sift = cv2.SIFT_create(
            nfeatures=500,  # Increased from default for more keypoints
            contrastThreshold=0.03,  # Lower threshold for more features
            edgeThreshold=10,
            sigma=1.6
        )
        
        # ORB as secondary feature detector for robustness
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15
        )
        
        # FLANN matcher parameters optimized for accuracy
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)  # More trees for better accuracy
        search_params = dict(checks=100)  # More checks for better matches
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # BFMatcher for ORB features
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _preprocess_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing to enhance fingerprint ridges and remove noise
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Denoise using bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(enhanced, 9, 50, 50)
        
        # Apply Gaussian blur to reduce noise further
        blurred = cv2.GaussianBlur(denoised, (3, 3), 1)
        
        # Enhance ridges using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Sharpen the image
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(morph, -1, kernel_sharpen)
        
        # Final normalization
        final = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return final
    
    def _calculate_match_accuracy(self, kp1: List, kp2: List, matches: List) -> Tuple[float, float, float, List]:
        """
        Calculate matching accuracy using multiple metrics with improved scoring
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image  
            matches: Raw matches from matcher
            
        Returns:
            Tuple of (accuracy_total, accuracy_conservative, match_ratio, good_matches)
        """
        if len(matches) == 0:
            return 0.0, 0.0, 0.0, []
        
        # Enhanced filtering using adaptive threshold based on match statistics
        distances = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                distances.append(match_pair[0].distance)
        
        # Calculate adaptive threshold
        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            adaptive_threshold = min(self.distance_threshold, 
                                    (mean_dist - 0.5 * std_dist) / (mean_dist + 0.5 * std_dist))
        else:
            adaptive_threshold = self.distance_threshold
        
        # Filter good matches using adaptive Lowe's ratio test
        good_matches = []
        match_scores = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                ratio = m.distance / n.distance if n.distance > 0 else 1.0
                
                if ratio < adaptive_threshold:
                    good_matches.append(m)
                    # Calculate match quality score
                    match_scores.append(1.0 - ratio)
        
        # Calculate spatial distribution of matches
        if good_matches:
            match_points = np.array([kp1[m.queryIdx].pt for m in good_matches])
            spatial_std = np.std(match_points, axis=0).mean()
            spatial_distribution_score = min(1.0, spatial_std / 100.0)  # Normalize to 0-1
        else:
            spatial_distribution_score = 0.0
        
        # Calculate different accuracy metrics with weights
        total_keypoints = max(len(kp1), len(kp2))
        min_keypoints = min(len(kp1), len(kp2))
        
        # Weighted accuracy based on match quality
        if match_scores:
            weighted_accuracy = (sum(match_scores) / min_keypoints) * 100 if min_keypoints > 0 else 0
        else:
            weighted_accuracy = 0
        
        # Standard accuracy metrics
        accuracy_total = (len(good_matches) / total_keypoints) * 100 if total_keypoints > 0 else 0
        accuracy_conservative = (len(good_matches) / min_keypoints) * 100 if min_keypoints > 0 else 0
        
        # Enhanced match ratio considering quality and distribution
        match_ratio = ((len(good_matches) / len(matches)) * 100 * 
                      (0.7 + 0.3 * spatial_distribution_score)) if len(matches) > 0 else 0
        
        # Apply bonus for high-quality matches
        if weighted_accuracy > 50:
            accuracy_conservative *= 1.1  # 10% bonus for high-quality matches
            
        return accuracy_total, min(100, accuracy_conservative), match_ratio, good_matches
    
    def _geometric_verification(self, kp1: List, kp2: List, matches: List) -> Tuple[List, float]:
        """
        Enhanced geometric verification using multiple methods
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches to verify
            
        Returns:
            Tuple of (verified_matches, inlier_ratio)
        """
        if len(matches) < 4:
            return matches, 0.0
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        best_inliers = []
        best_ratio = 0.0
        
        # Try multiple geometric models
        methods = [
            (cv2.findHomography, cv2.RANSAC, self.geometric_threshold),
            (cv2.findFundamentalMat, cv2.FM_RANSAC, self.geometric_threshold),
        ]
        
        for method, flag, threshold in methods:
            try:
                if method == cv2.findHomography:
                    M, mask = method(src_pts, dst_pts, flag, threshold)
                else:
                    M, mask = method(src_pts, dst_pts, flag, threshold)
                
                if mask is not None:
                    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
                    inlier_ratio = len(inlier_matches) / len(matches) * 100
                    
                    # Keep best result
                    if inlier_ratio > best_ratio:
                        best_inliers = inlier_matches
                        best_ratio = inlier_ratio
            except:
                continue
        
        # If geometric verification fails, use distance-based verification
        if best_ratio == 0.0:
            # Calculate pairwise distances consistency
            if len(matches) >= 2:
                src_dists = []
                dst_dists = []
                for i in range(min(10, len(matches))):
                    for j in range(i + 1, min(10, len(matches))):
                        pt1_src = kp1[matches[i].queryIdx].pt
                        pt2_src = kp1[matches[j].queryIdx].pt
                        pt1_dst = kp2[matches[i].trainIdx].pt
                        pt2_dst = kp2[matches[j].trainIdx].pt
                        
                        src_dist = np.linalg.norm(np.array(pt1_src) - np.array(pt2_src))
                        dst_dist = np.linalg.norm(np.array(pt1_dst) - np.array(pt2_dst))
                        
                        if src_dist > 0:
                            src_dists.append(src_dist)
                            dst_dists.append(dst_dist)
                
                if src_dists:
                    # Check consistency of distances
                    ratios = [dst_dists[i] / src_dists[i] for i in range(len(src_dists)) if src_dists[i] > 0]
                    if ratios:
                        consistency = 1.0 - (np.std(ratios) / (np.mean(ratios) + 1e-6))
                        best_ratio = max(0, min(100, consistency * 100))
                        best_inliers = matches
        
        return best_inliers if best_inliers else matches, best_ratio
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Extract enhanced SIFT features from preprocessed fingerprint image
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            # Preprocess the image for better feature extraction
            processed = self._preprocess_fingerprint(image)
            
            # Extract SIFT features
            keypoints, descriptors = self.sift.detectAndCompute(processed, None)
            
            # If not enough features, try with adjusted parameters
            if len(keypoints) < 50:
                # Create a new SIFT with more lenient parameters
                sift_lenient = cv2.SIFT_create(
                    nfeatures=1000,
                    contrastThreshold=0.02,
                    edgeThreshold=5,
                    sigma=1.2
                )
                keypoints, descriptors = sift_lenient.detectAndCompute(processed, None)
            
            return keypoints, descriptors
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, None
    
    def match_fingerprints(self, sample_image: np.ndarray, candidate_image: np.ndarray) -> Dict:
        """
        Match two fingerprint images using enhanced multi-algorithm approach
        
        Args:
            sample_image: Reference fingerprint image
            candidate_image: Candidate fingerprint image to compare
            
        Returns:
            Dictionary containing matching results and metrics
        """
        # Extract SIFT features from both images
        kp_sample, desc_sample = self.extract_features(sample_image)
        kp_candidate, desc_candidate = self.extract_features(candidate_image)
        
        # Check if features were extracted successfully
        if desc_sample is None or desc_candidate is None:
            return {
                'success': False,
                'error': 'Failed to extract features from one or both images',
                'score': 0.0,
                'accuracy_conservative': 0.0,
                'match_count': 0
            }
        
        if len(kp_sample) < 10 or len(kp_candidate) < 10:
            return {
                'success': False,
                'error': 'Insufficient keypoints detected',
                'score': 0.0,
                'accuracy_conservative': 0.0,
                'match_count': 0
            }
        
        try:
            # Match descriptors using FLANN
            matches = self.flann.knnMatch(desc_sample, desc_candidate, k=2)
            
            # Calculate accuracy metrics
            acc_total, acc_conservative, match_ratio, good_matches = self._calculate_match_accuracy(
                kp_sample, kp_candidate, matches
            )
            
            # Apply geometric verification
            verified_matches, geometric_score = self._geometric_verification(
                kp_sample, kp_candidate, good_matches
            )
            
            # Try ORB matching as secondary validation
            orb_score = 0.0
            try:
                processed_sample = self._preprocess_fingerprint(sample_image)
                processed_candidate = self._preprocess_fingerprint(candidate_image)
                
                kp_orb_sample, desc_orb_sample = self.orb.detectAndCompute(processed_sample, None)
                kp_orb_candidate, desc_orb_candidate = self.orb.detectAndCompute(processed_candidate, None)
                
                if desc_orb_sample is not None and desc_orb_candidate is not None:
                    orb_matches = self.bf_matcher.match(desc_orb_sample, desc_orb_candidate)
                    orb_matches = sorted(orb_matches, key=lambda x: x.distance)
                    
                    # Calculate ORB score
                    if orb_matches:
                        good_orb = orb_matches[:int(len(orb_matches) * 0.3)]  # Top 30% matches
                        orb_score = (len(good_orb) / min(len(kp_orb_sample), len(kp_orb_candidate))) * 100
            except:
                pass
            
            # Calculate minutiae-based features similarity (simplified)
            minutiae_score = 0.0
            if len(verified_matches) > 5:
                # Check consistency of keypoint scales and orientations
                scale_ratios = []
                angle_diffs = []
                for m in verified_matches[:20]:  # Use top 20 matches
                    kp1 = kp_sample[m.queryIdx]
                    kp2 = kp_candidate[m.trainIdx]
                    
                    if kp1.size > 0 and kp2.size > 0:
                        scale_ratios.append(kp2.size / kp1.size)
                    angle_diffs.append(abs(kp2.angle - kp1.angle))
                
                if scale_ratios:
                    scale_consistency = 1.0 - (np.std(scale_ratios) / (np.mean(scale_ratios) + 1e-6))
                    angle_consistency = 1.0 - (np.std(angle_diffs) / 180.0)
                    minutiae_score = ((scale_consistency + angle_consistency) / 2.0) * 100
            
            # Enhanced combined score with multiple metrics
            combined_score = (
                acc_conservative * 0.35 +  # Primary weight on conservative accuracy
                match_ratio * 0.20 +        # Match quality
                geometric_score * 0.20 +    # Geometric consistency
                orb_score * 0.15 +          # Secondary algorithm validation
                minutiae_score * 0.10        # Minutiae consistency
            )
            
            # Apply confidence boost for high match counts
            if len(verified_matches) > 50:
                combined_score *= 1.15
            elif len(verified_matches) > 30:
                combined_score *= 1.08
            
            combined_score = min(100, combined_score)  # Cap at 100
            
            # Enhanced quality determination with stricter thresholds
            if combined_score > 35 and acc_conservative > 20 and len(verified_matches) > 30:
                quality = "EXCELLENT"
            elif combined_score > 25 and acc_conservative > 15 and len(verified_matches) > 20:
                quality = "GOOD"
            elif combined_score > 15 and acc_conservative > 10 and len(verified_matches) > 10:
                quality = "MODERATE"
            elif combined_score > 8 and len(verified_matches) > 5:
                quality = "POOR"
            else:
                quality = "NO_MATCH"
            
            return {
                'success': True,
                'score': combined_score,
                'accuracy_total': acc_total,
                'accuracy_conservative': acc_conservative,
                'match_ratio': match_ratio,
                'geometric_score': geometric_score,
                'orb_score': orb_score,
                'minutiae_score': minutiae_score,
                'match_count': len(verified_matches),
                'quality': quality,
                'keypoints_sample': len(kp_sample),
                'keypoints_candidate': len(kp_candidate)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Matching failed: {str(e)}',
                'score': 0.0,
                'accuracy_conservative': 0.0,
                'match_count': 0
            }
    
    def match_against_database(self, sample_image: np.ndarray, database_path: str, 
                             max_files: int = 1000, progress_callback=None) -> Dict:
        """
        Match sample fingerprint against a database with optimized processing
        
        Args:
            sample_image: Reference fingerprint image
            database_path: Path to directory containing fingerprint database
            max_files: Maximum number of files to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing best match results
        """
        if not os.path.exists(database_path):
            return {
                'success': False,
                'error': f'Database directory not found: {database_path}',
                'best_match': None
            }
        
        # Preprocess sample once for efficiency
        processed_sample = self._preprocess_fingerprint(sample_image)
        
        # Extract features from sample
        kp_sample, desc_sample = self.extract_features(sample_image)
        
        if desc_sample is None:
            return {
                'success': False,
                'error': 'Failed to extract features from sample image',
                'best_match': None
            }
        
        # Get list of image files
        files = [f for f in os.listdir(database_path) 
                if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))][:max_files]
        
        if not files:
            return {
                'success': False,
                'error': 'No image files found in database directory',
                'best_match': None
            }
        
        # Initialize best match tracking with enhanced metrics
        best_match = {
            'score': 0,
            'accuracy_conservative': 0,
            'filename': None,
            'match_count': 0,
            'quality': 'NO_MATCH'
        }
        
        # Keep track of top N matches for verification
        top_matches = []
        
        # Process each file in database
        for counter, filename in enumerate(files):
            if progress_callback:
                progress = (counter + 1) / len(files) * 100
                progress_callback(progress, filename, best_match)
            
            filepath = os.path.join(database_path, filename)
            candidate_image = cv2.imread(filepath)
            
            if candidate_image is None:
                continue
            
            # Match against current candidate
            result = self.match_fingerprints(sample_image, candidate_image)
            
            if result['success']:
                # Store in top matches
                match_info = {
                    'filename': filename,
                    **result
                }
                top_matches.append(match_info)
                
                # Update best match with stricter criteria
                if (result['score'] > best_match['score'] and 
                    result['match_count'] > 10 and
                    result['quality'] != 'NO_MATCH'):
                    
                    best_match.update({
                        'score': result['score'],
                        'accuracy_conservative': result['accuracy_conservative'],
                        'accuracy_total': result['accuracy_total'],
                        'match_ratio': result['match_ratio'],
                        'geometric_score': result['geometric_score'],
                        'match_count': result['match_count'],
                        'quality': result['quality'],
                        'filename': filename,
                        'keypoints_sample': result['keypoints_sample'],
                        'keypoints_candidate': result['keypoints_candidate'],
                        'orb_score': result.get('orb_score', 0),
                        'minutiae_score': result.get('minutiae_score', 0)
                    })
        
        # Sort top matches and verify consistency
        top_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Validate best match against second best for confidence
        confidence = 'HIGH'
        if len(top_matches) >= 2:
            score_gap = best_match['score'] - top_matches[1]['score']
            if score_gap < 5:
                confidence = 'LOW'
            elif score_gap < 10:
                confidence = 'MEDIUM'
        
        return {
            'success': True,
            'total_processed': len(files),
            'best_match': best_match if best_match['filename'] else None,
            'confidence': confidence if best_match['filename'] else None,
            'top_5_matches': top_matches[:5] if top_matches else []
        }

def load_image_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Utility function to load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    try:
        image = cv2.imread(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Utility function to load image from bytes (useful for FastAPI file uploads)
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Loaded image or None if failed
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image from bytes: {e}")
        return None
