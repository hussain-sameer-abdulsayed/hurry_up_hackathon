#!/usr/bin/env python3
# visual.py
import cv2
import numpy as np
from fingerprint_matcher import FingerprintMatcher
import sys

def create_match_visualization(img1, img2, kp1, kp2, matches, title="Fingerprint Match Visualization"):
    """
    Create a side-by-side visualization of fingerprint matching
    
    Args:
        img1: First fingerprint image
        img2: Second fingerprint image
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of good matches
        title: Title for the visualization
        
    Returns:
        Visualization image
    """
    # Convert to RGB if grayscale
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(255, 0, 0),  # Blue color for unmatched keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add title and information overlay
    height, width = img_matches.shape[:2]
    
    # Create a header area
    header_height = 60
    result_img = np.zeros((height + header_height, width, 3), dtype=np.uint8)
    result_img[header_height:, :] = img_matches
    
    # Add white background for header
    result_img[:header_height, :] = (255, 255, 255)
    
    # Add title
    cv2.putText(result_img, title, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    return result_img

def get_matches_for_visualization(matcher, kp1, desc1, kp2, desc2, skip_geometric=False):
    """
    Get matches using the same logic as the matcher's match_fingerprints method
    
    Args:
        matcher: FingerprintMatcher instance
        kp1, desc1: Keypoints and descriptors from first image
        kp2, desc2: Keypoints and descriptors from second image
        skip_geometric: Whether to skip geometric verification
        
    Returns:
        List of verified matches
    """
    # Match descriptors
    matches = matcher.flann.knnMatch(desc1, desc2, k=2)
    
    # Filter good matches using Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < matcher.distance_threshold * n.distance:
                good_matches.append(m)
    
    # Apply geometric verification (unless skipped)
    if skip_geometric or len(good_matches) < 4:
        verified_matches = good_matches
    else:
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography and filter inliers
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, matcher.geometric_threshold)
            if mask is not None:
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                if len(inlier_matches) > 0:
                    verified_matches = inlier_matches
                else:
                    # No inliers found, use original good matches
                    verified_matches = good_matches
            else:
                # Homography failed, use original good matches
                verified_matches = good_matches
        except:
            # If homography fails, use original good matches
            verified_matches = good_matches
    
    return verified_matches

def main():
    # Initialize the matcher
    matcher = FingerprintMatcher()
    print("Fingerprint Matcher initialized")
    
    # Load fingerprint images
    fingerprint1 = cv2.imread('fingerprint1.jpg')
    fingerprint2 = cv2.imread('fingerprint2.jpg')
    
    # Check if images loaded successfully
    if fingerprint1 is None:
        print("Error: Could not load fingerprint1.jpg")
        print("Please make sure the file exists in the current directory")
        sys.exit(1)
    
    if fingerprint2 is None:
        print("Error: Could not load fingerprint2.jpg")
        print("Please make sure the file exists in the current directory")
        sys.exit(1)
    
    print("Images loaded successfully!")
    print(f"Image 1 dimensions: {fingerprint1.shape}")
    print(f"Image 2 dimensions: {fingerprint2.shape}")
    
    # Extract features
    print("\nExtracting features...")
    kp1, desc1 = matcher.extract_features(fingerprint1)
    kp2, desc2 = matcher.extract_features(fingerprint2)
    
    if desc1 is None or desc2 is None:
        print("Error: Failed to extract features from images")
        sys.exit(1)
    
    print(f"Keypoints detected in image 1: {len(kp1)}")
    print(f"Keypoints detected in image 2: {len(kp2)}")
    
    # Perform matching WITH geometric verification
    print("\nPerforming fingerprint matching WITH geometric verification...")
    match_result = matcher.match_fingerprints(fingerprint1, fingerprint2, skip_geometric=False)
    
    # Also try WITHOUT geometric verification for comparison
    print("Performing fingerprint matching WITHOUT geometric verification...")
    match_result_no_geo = matcher.match_fingerprints(fingerprint1, fingerprint2, skip_geometric=True)
    
    # Print results
    print("\n" + "="*50)
    print("MATCHING RESULTS (WITH GEOMETRIC VERIFICATION)")
    print("="*50)
    
    if match_result.get('success', False):
        print(f"✓ Match Success: True")
        print(f"✓ Match Score: {match_result.get('score', 0.0):.2f}%")
        print(f"✓ Conservative Accuracy: {match_result.get('accuracy_conservative', 0.0):.2f}%")
        print(f"✓ Total Accuracy: {match_result.get('accuracy_total', 0.0):.2f}%")
        print(f"✓ Match Ratio: {match_result.get('match_ratio', 0.0):.2f}%")
        print(f"✓ Geometric Score: {match_result.get('geometric_score', 0.0):.2f}%")
        print(f"✓ Number of verified matches: {match_result.get('match_count', 0)}")
        print(f"✓ Good matches before geometric: {match_result.get('good_matches_before_geometric', 0)}")
        print(f"✓ Geometric verification applied: {match_result.get('geometric_verification_applied', False)}")
        print(f"✓ Match Quality: {match_result.get('quality', 'N/A')}")
        print(f"✓ Keypoints in fingerprint1: {match_result.get('keypoints_sample', 0)}")
        print(f"✓ Keypoints in fingerprint2: {match_result.get('keypoints_candidate', 0)}")
        
        print("\n" + "="*50)
        print("MATCHING RESULTS (WITHOUT GEOMETRIC VERIFICATION)")
        print("="*50)
        print(f"✓ Match Score: {match_result_no_geo.get('score', 0.0):.2f}%")
        print(f"✓ Number of matches: {match_result_no_geo.get('match_count', 0)}")
        
        # Get matches for visualization using the same logic as match_fingerprints
        print("\nGetting matches for visualization...")
        
        # Get matches WITH geometric verification
        verified_matches = get_matches_for_visualization(matcher, kp1, desc1, kp2, desc2, skip_geometric=False)
        print(f"Matches for visualization (with geometric): {len(verified_matches)}")
        
        # Get matches WITHOUT geometric verification
        good_matches = get_matches_for_visualization(matcher, kp1, desc1, kp2, desc2, skip_geometric=True)
        print(f"Matches for visualization (without geometric): {len(good_matches)}")
        
        # Create visualizations for both cases
        print("\nCreating visualizations...")
        
        # Visualization WITH geometric verification
        if len(verified_matches) > 0:
            title_geo = f"With Geometric - Matches: {len(verified_matches)} | Score: {match_result.get('score', 0.0):.1f}%"
            vis_img_geo = create_match_visualization(
                fingerprint1, fingerprint2,
                kp1, kp2,
                verified_matches[:50],  # Limit to 50 matches for clarity
                title_geo
            )
            cv2.imwrite("match_visualization_with_geometric.jpg", vis_img_geo)
            print(f"✓ Saved 'match_visualization_with_geometric.jpg' ({len(verified_matches)} matches)")
        else:
            print("✗ No matches after geometric verification")
        
        # Visualization WITHOUT geometric verification
        if len(good_matches) > 0:
            title_no_geo = f"Without Geometric - Matches: {len(good_matches)} | Score: {match_result_no_geo.get('score', 0.0):.1f}%"
            vis_img_no_geo = create_match_visualization(
                fingerprint1, fingerprint2,
                kp1, kp2,
                good_matches[:50],  # Limit to 50 matches for clarity
                title_no_geo
            )
            cv2.imwrite("match_visualization_without_geometric.jpg", vis_img_no_geo)
            print(f"✓ Saved 'match_visualization_without_geometric.jpg' ({len(good_matches)} matches)")
        else:
            print("✗ No good matches found")
        
        # Use the version with more matches for the main visualization
        if len(good_matches) > len(verified_matches):
            print(f"\nUsing non-geometric matches for main visualization ({len(good_matches)} matches)")
            main_matches = good_matches
            main_title = f"Matches: {len(good_matches)} | Quality: {match_result_no_geo.get('quality', 'N/A')} | Score: {match_result_no_geo.get('score', 0.0):.1f}%"
        else:
            print(f"\nUsing geometric-verified matches for main visualization ({len(verified_matches)} matches)")
            main_matches = verified_matches
            main_title = f"Matches: {len(verified_matches)} | Quality: {match_result.get('quality', 'N/A')} | Score: {match_result.get('score', 0.0):.1f}%"
        
        if len(main_matches) > 0:
            vis_img_main = create_match_visualization(
                fingerprint1, fingerprint2,
                kp1, kp2,
                main_matches[:50],
                main_title
            )
            cv2.imwrite("match_visualization.jpg", vis_img_main)
            print(f"✓ Main visualization saved as 'match_visualization.jpg'")
        
        # Also create individual keypoint visualizations
        print("\nCreating individual keypoint visualizations...")
        
        # Draw keypoints on fingerprint1
        img1_kp = cv2.drawKeypoints(fingerprint1, kp1, None, 
                                    color=(0, 255, 0), 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("fingerprint1_keypoints.jpg", img1_kp)
        print(f"✓ Saved 'fingerprint1_keypoints.jpg'")
        
        # Draw keypoints on fingerprint2
        img2_kp = cv2.drawKeypoints(fingerprint2, kp2, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("fingerprint2_keypoints.jpg", img2_kp)
        print(f"✓ Saved 'fingerprint2_keypoints.jpg'")
        
        # Create a summary image with all visualizations
        print("\nCreating summary visualization...")
        
        # Resize images for summary
        h, w = 300, 300
        img1_resized = cv2.resize(fingerprint1, (w, h))
        img2_resized = cv2.resize(fingerprint2, (w, h))
        img1_kp_resized = cv2.resize(img1_kp, (w, h))
        img2_kp_resized = cv2.resize(img2_kp, (w, h))
        
        # Create 2x2 grid
        top_row = np.hstack([img1_resized, img2_resized])
        bottom_row = np.hstack([img1_kp_resized, img2_kp_resized])
        summary = np.vstack([top_row, bottom_row])
        
        # Add labels
        cv2.putText(summary, "Original 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(summary, "Original 2", (w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(summary, f"Keypoints: {len(kp1)}", (10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(summary, f"Keypoints: {len(kp2)}", (w+10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite("summary_visualization.jpg", summary)
        print(f"✓ Saved 'summary_visualization.jpg'")
        
        print("\n" + "="*50)
        print("VISUALIZATION COMPLETE!")
        print("="*50)
        print("\nGenerated files:")
        print("  1. match_visualization.jpg - Main matching visualization")
        print("  2. match_visualization_with_geometric.jpg - Matches with geometric verification")
        print("  3. match_visualization_without_geometric.jpg - Matches without geometric verification")
        print("  4. fingerprint1_keypoints.jpg - Keypoints on first image")
        print("  5. fingerprint2_keypoints.jpg - Keypoints on second image")
        print("  6. summary_visualization.jpg - Combined summary view")
        
        print("\n" + "="*50)
        print("DEBUGGING INFORMATION")
        print("="*50)
        print(f"Total keypoints in image 1: {len(kp1)}")
        print(f"Total keypoints in image 2: {len(kp2)}")
        print(f"Good matches (Lowe's ratio test): {len(good_matches)}")
        print(f"Verified matches (after geometric): {len(verified_matches)}")
        print(f"Match count from match_fingerprints: {match_result.get('match_count', 0)}")
        
    else:
        print(f"✗ Match Success: False")
        if 'error' in match_result:
            print(f"✗ Error: {match_result['error']}")
        print("\nNo visualization created due to matching failure")

if __name__ == "__main__":
    main()
