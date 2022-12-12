# ECE558_project3_option2
## Problem description
This project focuses on improving the implementation of the paper “single view metrology” (Criminisi, Reid and Zisserman, ICCV99). It substitutes the manual annotation of vanishing points with a detection algorithm based on a line segment
detector (http://www.ipol.im/pub/art/2012/gjmr-lsd/ ) and the RANSAC algorithm. You are encouraged to explore other methods of detecting vanishing points automatically.

## Installation
```
pip install -r requirements.txt
```

## Image acquisition.
Take a photo by yourself following the guide of 3-point perspective image as input.
![box_perspective](box.png)

## Color filtering mask
An interface is wriiten for self-adjust the color mask.
![mask_interface](./result_images/line_detector/mask_interface.gif)

## Canny and Houghline detection results
The results of canny and Hough line are much better dur to the mask.

![canny_detection](./result_images/line_detector/canny_masked_default.png)
![houghline_detection](./result_images/line_detector/houghline_masked_default.png)

## Final 3D reconstruction model
