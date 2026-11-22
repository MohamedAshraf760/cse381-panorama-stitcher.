import cv2
import numpy as np
import argparse

def stitch_images(img1_path, img2_path):
    # 1. Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load one of the images. Check the file paths.")
        return

    # 2. SIFT detection and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 3. Ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")

    if len(good_matches) > 4:
        # 4. RANSAC Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Count inliers
        inliers = np.sum(mask)
        print(f"RANSAC Inliers: {inliers}/{len(good_matches)}")

        # 5. Warping and Stitching
        # Get dimensions of the second image
        h, w = img2.shape[:2]
        
        # Warp the first image to align with the second
        result = cv2.warpPerspective(img1, M, (w + img1.shape[1], h))
        
        # Place the second image on top
        result[0:h, 0:w] = img2
        
        # 6. Save output
        cv2.imwrite("demo.png", result)
        print("Success! Saved as demo.png")
    else:
        print("Not enough matches found to stitch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panorama Stitcher")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    args = parser.parse_args()
    
    stitch_images(args.img1, args.img2)
