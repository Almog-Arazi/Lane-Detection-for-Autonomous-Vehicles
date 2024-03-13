# Lane Detection for Autonomous Vehicles

This project focuses on implementing a lane detection system for autonomous vehicles using computer vision techniques. The project encompasses various aspects of image processing and analysis, including lane marking detection, night-time adaptation, and crosswalk detection. The algorithm aims to accurately identify lane markings under different conditions and enhance the safety and reliability of autonomous driving systems.

## Core Project
Our algorithm for finding paths in several steps:
1. Setup and Initialization
2. Pre-process each frame
3. Focus on Region of Interest (ROI)
4. Detect Lane Lines
5. Smooth and average Lane Line Detection
6. Draw Lane Lines
7. Detect and Indicate Route Changes
8. Adjust Brightness and Contrast if needed (Night Mode)
9. Crosswalk detection (if needed)
10. Process Video

During the code, we used several techniques we learned in order to reach an optimal result:
 Greyscaling: in order to simplify the processing and calculations in each frame.
 Canny: in order to extract the edges of the objects, especially the paths.
 ROI Crop: Analysis of the relevant parts in each frame, with the first being the lane in which the car is located.
 Hough Line Transform: Identifying straight lines in the frame - enables the extraction of paths from among all existing objects.
 Use of mathematical manipulations in order to avoid noises and enable the marking of the lanes smoothly and uniformly.
 In order to identify intersections, we used morphological operations to highlight objects, and finding contours by cv.findContours.

### Lane Marking Detection and Visualization
- Annotate each frame of the video with accurate lane markings.
- Include a demonstration of a lane change with appropriate markings.
- Implement strategies for cropping, color thresholding, and line verification.

###  Frame Analysis
- Provide an in-depth analysis of an example frame.
- Illustrate the approach to identifying and marking lane lines.
- Address complexities like varying light conditions or nearby traffic.

## Night-Time Lane Detection

### Objective
Adapt the lane detection algorithm for low-light conditions.
Implement image enhancement techniques to improve lane visibility.
Ensure high accuracy comparable to daytime performance.

## Crosswalk Detection

### Objective
Enhance the lane detection system with crosswalk detection capability.
Implement algorithms to recognize crosswalk patterns and lines.
Differentiate crosswalks from other road markings for safety.

## Results
### Day Result



https://github.com/Almog-Arazi/Lane-Detection-for-Autonomous-Vehicles/assets/112971847/50d1b6ce-786e-4a21-81cf-37617d20c594


https://youtu.be/aVcRgJhPpVo


### Night-Time Lane Detection Result


https://github.com/Almog-Arazi/Lane-Detection-for-Autonomous-Vehicles/assets/112971847/3fc6225b-eece-427e-ac02-4ac1811be799

https://youtu.be/rlmJ9cQNHYw

### Crosswalk Detection Result


https://github.com/Almog-Arazi/Lane-Detection-for-Autonomous-Vehicles/assets/112971847/3a06be18-0667-4325-98b8-8aa4b89bfe18


https://youtu.be/pUXdubbXLzc?si=guNPnD2d0_9Ngf1E
## Python Code Overview

### Day and Night Algorithm
- Implements lane detection for both daytime and nighttime conditions.
- Utilizes Canny edge detection, Hough transform, and image manipulation.
- Includes functions for adjusting brightness and contrast.

### Crosswalk Algorithm
- Detects crosswalks alongside lane markings in video footage.
- Integrates contour detection and pattern recognition techniques.
- Enhances safety features for comprehensive lane detection.

**Note:** Ensure appropriate video inputs and parameters for optimal performance of the algorithms.
