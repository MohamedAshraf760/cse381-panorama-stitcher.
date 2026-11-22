# CSE381 Panorama Stitcher

Stitches two overlapping photos into a seamless panorama using SIFT feature matching, RANSAC homography estimation, and inverse warping.

![demo](demo.png) 

## Result
RANSAC inliers: 142/158 matches (89.9%) · Final panorama: 1400×760 px (Update these numbers later if yours are different!)

## How to run

```bash
git clone https://github.com/YOUR_USERNAME_HERE/cse381-panorama-stitcher
cd cse381-panorama-stitcher
pip install -r requirements.txt
python stitch.py --img1 photos/left.jpg --img2 photos/right.jpg

What I learned

Implementing RANSAC from scratch made me understand why a single outlier can wreck least-squares fitting — watching the inlier mask visually separate the correct matches from mismatched ones was the most useful debugging moment of the project.

Built with

Python · OpenCV · NumPy — CSE381 Computer Vision, MSA University
