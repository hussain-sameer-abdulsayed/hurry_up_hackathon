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
    
    def __init__(self, distance_threshold: float = 0.75, geometric_threshold: float = 5.0, min_match_distance: float = 30.0):
        """
        Initialize the fingerprint matcher
        
        Args:
            distance_threshold: Threshold for Lowe's ratio test (0.75-0.8 recommended for fingerprints)
            geometric_threshold: Threshold for RANSAC in geometric verification
            min_match_distance: Minimum distance between matched keypoints to avoid duplicates
        """
        self.distance_threshold = distance_threshold
        self.geometric_threshold = geometric_threshold
        self.min_match_distance = min_match_distance
        self.sift = cv2.SIFT_create()
        
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def _remove_duplicate_matches(self, kp1: List, kp2: List, matches: List) -> List:
        """
        Remove duplicate and one-to-many matches
        Ensures each keypoint in both images is matched at most once
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches to filter
            
        Returns:
            Filtered list of unique matches
        """
        if len(matches) == 0:
            return matches
        
        # Sort matches by distance (best matches first)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        
        # Track which keypoints have been used
        used_query_idx = set()
        used_train_idx = set()
        unique_matches = []
        
        for match in sorted_matches:
            # Check if either keypoint has already been matched
            if match.queryIdx not in used_query_idx and match.trainIdx not in used_train_idx:
                # Also check for spatial proximity to avoid matching one point to nearby points
                is_duplicate = False
                
                query_pt = kp1[match.queryIdx].pt
                train_pt = kp2[match.trainIdx].pt
                
                # Check against existing matches for spatial duplicates
                for existing_match in unique_matches:
                    existing_query_pt = kp1[existing_match.queryIdx].pt
                    existing_train_pt = kp2[existing_match.trainIdx].pt
                    
                    # Calculate distances
                    query_dist = np.sqrt((query_pt[0] - existing_query_pt[0])**2 + 
                                       (query_pt[1] - existing_query_pt[1])**2)
                    train_dist = np.sqrt((train_pt[0] - existing_train_pt[0])**2 + 
                                       (train_pt[1] - existing_train_pt[1])**2)
                    
                    # If points are too close, consider it a duplicate
                    if query_dist < self.min_match_distance or train_dist < self.min_match_distance:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_matches.append(match)
                    used_query_idx.add(match.queryIdx)
                    used_train_idx.add(match.trainIdx)
        
        return unique_matches
    
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
                # Apply Lowe's ratio test
                if m.distance < self.distance_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # If only one match found, check if it's good enough
                m = match_pair[0]
                if m.distance < 100:  # Absolute threshold for single matches
                    good_matches.append(m)
        
        # Remove duplicate matches (one-to-many mappings)
        good_matches = self._remove_duplicate_matches(kp1, kp2, good_matches)
        
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
            # Not enough matches for homography, return original matches
            return matches, 100.0 if len(matches) > 0 else 0.0
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography and filter inliers
        try:
            # Use RANSAC to find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.geometric_threshold)
            
            if mask is not None and M is not None:
                # Get inlier matches
                inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
                
                # Remove any remaining duplicates after geometric verification
                inlier_matches = self._remove_duplicate_matches(kp1, kp2, inlier_matches)
                
                if len(inlier_matches) > 0:
                    inlier_ratio = (len(inlier_matches) / len(matches)) * 100
                    return inlier_matches, inlier_ratio
                else:
                    # No inliers found, return a subset of original matches
                    # This might indicate the geometric model doesn't fit well
                    return matches[:min(len(matches), 10)], 25.0
            else:
                # Homography failed, return original matches with lower score
                return matches, 50.0
                
        except Exception as e:
            print(f"Geometric verification exception: {e}")
            # If geometric verification fails completely, return original matches
            return matches, 50.0
    
    def extract_features(self, image: np.ndarray, enhance: bool = True) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Extract SIFT features from fingerprint image
        
        Args:
            image: Input fingerprint image
            enhance: Whether to apply image enhancement for better feature detection
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
            
            # Optional: Enhance fingerprint image for better feature detection
            if enhance:
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_image = clahe.apply(gray_image)
                
                # Apply slight Gaussian blur to reduce noise
                gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            
            # Extract SIFT features
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, None
    
    def match_fingerprints(self, sample_image: np.ndarray, candidate_image: np.ndarray,
                         skip_geometric: bool = False, enhance_images: bool = True) -> Dict:
        """
        Match two fingerprint images and return similarity metrics
        
        Args:
            sample_image: Reference fingerprint image
            candidate_image: Candidate fingerprint image to compare
            skip_geometric: If True, skip geometric verification
            enhance_images: If True, apply image enhancement before feature extraction
            
        Returns:
            Dictionary containing matching results and metrics
        """
        # Extract features from both images
        kp_sample, desc_sample = self.extract_features(sample_image, enhance=enhance_images)
        kp_candidate, desc_candidate = self.extract_features(candidate_image, enhance=enhance_images)
        
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
                'error': f'Insufficient keypoints detected (sample: {len(kp_sample)}, candidate: {len(kp_candidate)})',
                'score': 0.0,
                'accuracy_conservative': 0.0,
                'match_count': 0
            }
        
        try:
            # Match descriptors using FLANN
            # Use k=2 for Lowe's ratio test
            matches = self.flann.knnMatch(desc_sample, desc_candidate, k=2)
            
            # Filter out None matches
            matches = [m for m in matches if m is not None and len(m) > 0]
            
            # Calculate accuracy metrics and get good matches
            acc_total, acc_conservative, match_ratio, good_matches = self._calculate_match_accuracy(
                kp_sample, kp_candidate, matches
            )
            
            # Store count before geometric verification
            good_matches_count = len(good_matches)
            
            # Apply geometric verification (unless skipped)
            if skip_geometric or len(good_matches) < 4:
                verified_matches = good_matches
                geometric_score = 100.0 if len(good_matches) > 0 else 0.0
                geometric_applied = False
            else:
                verified_matches, geometric_score = self._geometric_verification(
                    kp_sample, kp_candidate, good_matches
                )
                geometric_applied = True
            
            # Final match count after all filtering
            final_match_count = len(verified_matches)
            
            # Calculate combined score
            if final_match_count == 0:
                combined_score = 0.0
            elif skip_geometric:
                combined_score = (acc_conservative * 0.6 + match_ratio * 0.4)
            else:
                combined_score = (acc_conservative * 0.5 + match_ratio * 0.3 + geometric_score * 0.2)
            
            # Determine match quality based on the number of verified matches and accuracy
            if final_match_count >= 20 and acc_conservative > 15:
                quality = "EXCELLENT"
            elif final_match_count >= 15 and acc_conservative > 10:
                quality = "GOOD"
            elif final_match_count >= 10 and acc_conservative > 5:
                quality = "MODERATE"
            elif final_match_count >= 5:
                quality = "POOR"
            else:
                quality = "VERY_POOR"
            
            return {
                'success': True,
                'score': combined_score,
                'accuracy_total': acc_total,
                'accuracy_conservative': acc_conservative,
                'match_ratio': match_ratio,
                'geometric_score': geometric_score,
                'match_count': final_match_count,
                'good_matches_before_geometric': good_matches_count,
                'quality': quality,
                'keypoints_sample': len(kp_sample),
                'keypoints_candidate': len(kp_candidate),
                'geometric_verification_applied': geometric_applied,
                'total_raw_matches': len(matches)
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
                             max_files: int = 1000, progress_callback=None,
                             skip_geometric: bool = False, enhance_images: bool = True) -> Dict:
        """
        Match sample fingerprint against a database of fingerprints
        
        Args:
            sample_image: Reference fingerprint image
            database_path: Path to directory containing fingerprint database
            max_files: Maximum number of files to process
            progress_callback: Optional callback function for progress updates
            skip_geometric: If True, skip geometric verification
            enhance_images: If True, apply image enhancement
            
        Returns:
            Dictionary containing best match results
        """
        if not os.path.exists(database_path):
            return {
                'success': False,
                'error': f'Database directory not found: {database_path}',
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
            'quality': 'VERY_POOR'
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
            result = self.match_fingerprints(
                sample_image, candidate_image, 
                skip_geometric=skip_geometric,
                enhance_images=enhance_images
            )
            
            # Update best match if this one is better
            if result['success'] and result['match_count'] > 0:
                # Consider both score and match count for determining best match
                is_better = False
                
                if result['match_count'] > best_match['match_count']:
                    is_better = True
                elif result['match_count'] == best_match['match_count'] and result['score'] > best_match['score']:
                    is_better = True
                
                if is_better:
                    best_match.update({
                        'score': result['score'],
                        'accuracy_conservative': result['accuracy_conservative'],
                        'accuracy_total': result['accuracy_total'],
                        'match_ratio': result['match_ratio'],
                        'geometric_score': result['geometric_score'],
                        'match_count': result['match_count'],
                        'good_matches_before_geometric': result.get('good_matches_before_geometric', 0),
                        'quality': result['quality'],
                        'filename': filename,
                        'keypoints_sample': result['keypoints_sample'],
                        'keypoints_candidate': result['keypoints_candidate'],
                        'geometric_verification_applied': result.get('geometric_verification_applied', False)
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

# Example usage for testing
if __name__ == "__main__":
    # Initialize matcher with adjusted parameters for fingerprints
    matcher = FingerprintMatcher(
        distance_threshold=0.75,  # Slightly relaxed for fingerprints
        geometric_threshold=5.0,
        min_match_distance=30.0   # Minimum pixel distance between matches
    )
    
    # Test with two images
    sample_path = "input.jpg"
    candidate_path = "output.jpg"
    
    sample_img = load_image_from_path(sample_path)
    candidate_img = load_image_from_path(candidate_path)
    
    if sample_img is not None and candidate_img is not None:
        print("="*60)
        print("FINGERPRINT MATCHING TEST")
        print("="*60)
        
        # Test with different configurations
        configs = [
            (False, True, "With geometric verification + enhancement"),
            (True, True, "Without geometric verification + enhancement"),
            (False, False, "With geometric verification, no enhancement"),
            (True, False, "Without geometric verification, no enhancement")
        ]
        
        for skip_geo, enhance, description in configs:
            print(f"\n{description}:")
            print("-"*40)
            result = matcher.match_fingerprints(
                sample_img, candidate_img, 
                skip_geometric=skip_geo,
                enhance_images=enhance
            )
            
            if result['success']:
                print(f"✓ Match count: {result['match_count']}")
                print(f"✓ Good matches (before geometric): {result.get('good_matches_before_geometric', 'N/A')}")
                print(f"✓ Score: {result['score']:.2f}%")
                print(f"✓ Quality: {result['quality']}")
                print(f"✓ Conservative accuracy: {result['accuracy_conservative']:.2f}%")
            else:
                print(f"✗ Error: {result['error']}")
