#!/usr/bin/env python3
# fingerprint_matcher.py
import cv2
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union

class FingerprintMatcher:
    """
    Fingerprint matching class optimized for FastAPI backend usage
    """
    
    def __init__(self, distance_threshold: float = 0.7, geometric_threshold: float = 5.0):
        """
        Initialize the fingerprint matcher
        
        Args:
            distance_threshold: Threshold for Lowe's ratio test
            geometric_threshold: Threshold for geometric verification
        """
        self.distance_threshold = distance_threshold
        self.geometric_threshold = geometric_threshold
        self.sift = cv2.SIFT_create()
        
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def _calculate_match_accuracy(self, kp1: List, kp2: List, matches: List) -> Tuple[float, float, float, List]:
        """
        Calculate matching accuracy using multiple metrics
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image  
            matches: Raw matches from matcher
            
        Returns:
            Tuple of (accuracy_total, accuracy_conservative, match_ratio, good_matches)
        """
        if len(matches) == 0:
            return 0.0, 0.0, 0.0, []
        
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.distance_threshold * n.distance:
                    good_matches.append(m)
        
        # Calculate different accuracy metrics
        total_keypoints = max(len(kp1), len(kp2))
        min_keypoints = min(len(kp1), len(kp2))
        
        # Accuracy based on good matches vs total keypoints
        accuracy_total = (len(good_matches) / total_keypoints) * 100 if total_keypoints > 0 else 0
        
        # Accuracy based on good matches vs minimum keypoints (more conservative)
        accuracy_conservative = (len(good_matches) / min_keypoints) * 100 if min_keypoints > 0 else 0
        
        # Match ratio (percentage of good matches from all possible matches)
        match_ratio = (len(good_matches) / len(matches)) * 100 if len(matches) > 0 else 0
        
        return accuracy_total, accuracy_conservative, match_ratio, good_matches
    
    def _geometric_verification(self, kp1: List, kp2: List, matches: List) -> Tuple[List, float]:
        """
        Use geometric verification (homography) to filter out false matches
        
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
        
        # Find homography and filter inliers
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.geometric_threshold)
            if mask is not None:
                inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
                inlier_ratio = len(inlier_matches) / len(matches) * 100
                return inlier_matches, inlier_ratio
        except:
            pass
        
        return matches, 100.0
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Extract SIFT features from fingerprint image
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            return keypoints, descriptors
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, None
    
    def match_fingerprints(self, sample_image: np.ndarray, candidate_image: np.ndarray) -> Dict:
        """
        Match two fingerprint images and return similarity metrics
        
        Args:
            sample_image: Reference fingerprint image
            candidate_image: Candidate fingerprint image to compare
            
        Returns:
            Dictionary containing matching results and metrics
        """
        # Extract features from both images
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
            # Match descriptors
            matches = self.flann.knnMatch(desc_sample, desc_candidate, k=2)
            
            # Calculate accuracy metrics
            acc_total, acc_conservative, match_ratio, good_matches = self._calculate_match_accuracy(
                kp_sample, kp_candidate, matches
            )
            
            # Apply geometric verification
            verified_matches, geometric_score = self._geometric_verification(
                kp_sample, kp_candidate, good_matches
            )
            
            # Combined score (weighted average of different metrics)
            combined_score = (acc_conservative * 0.5 + match_ratio * 0.3 + geometric_score * 0.2)
            
            # Determine match quality
            if acc_conservative > 15:
                quality = "EXCELLENT"
            elif acc_conservative > 10:
                quality = "GOOD"
            elif acc_conservative > 5:
                quality = "MODERATE"
            else:
                quality = "POOR"
            
            return {
                'success': True,
                'score': combined_score,
                'accuracy_total': acc_total,
                'accuracy_conservative': acc_conservative,
                'match_ratio': match_ratio,
                'geometric_score': geometric_score,
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
        Match sample fingerprint against a database of fingerprints
        
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
        
        # Initialize best match tracking
        best_match = {
            'score': 0,
            'accuracy_conservative': 0,
            'filename': None,
            'match_count': 0,
            'quality': 'POOR'
        }
        
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
            
            if result['success'] and result['score'] > best_match['score'] and result['match_count'] > 10:
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
                    'keypoints_candidate': result['keypoints_candidate']
                })
        
        return {
            'success': True,
            'total_processed': len(files),
            'best_match': best_match if best_match['filename'] else None
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
