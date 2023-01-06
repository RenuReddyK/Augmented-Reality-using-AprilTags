# Augmented-Reality-using-AprilTags



https://user-images.githubusercontent.com/68454938/210908731-cca70bc9-bfea-450b-b60c-7a35299112fe.mp4


A video with AprilTags in each frame are the inputs. The aim of this project to generate a video that contains several virtual object models as if they exist in the real world. Furthermore, we will be able to specify pixel positions to place an arbitrary object. 

Two approaches can be used to recover the camera poses:

Solving the Perspective-N-Point (PnP) problem with coplanar assumption: The camera pose is estimated from an AprilTag based on homography estimation. The calculated homography matrix might have scale ambiguity. So it is important to normalise the homography matrix to get rid of the unreal situations.  
Solving the Persepective-three-point (P3P) and the Procrustes problem: The camera pose is calculated by first calculating the 3D coordinates of any 3 (out of 4) corners of the AprilTag in the camera frame. To solve the Procrustes problem the correspondence of the same 3 points in the world frame and the camera frame are used to solve for camera pose. 
After retrieving the 3D relationship between the camera and world, we can place an arbitrary objects in the scene. 

