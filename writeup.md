# **Finding Lane Lines on the Road** 

## Writeup Template


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road on images.
* Make a pipeline that finds lane lines on the road on videos.


[//]: # (Image References)

[image1]: ./writeup_images/grayscale.png "Grayscale"
[image2]: ./writeup_images/edges.png "Canny Edges"
[image3]: ./writeup_images/edges_mask.png "Masked Edges"
[image4]: ./test_images_output/solidYellowCurve.jpg "Solid Yellow Curve Example"


---

### Reflection

### 1. Description of the pipeline. 

My pipeline consisted of 6 steps.

#### a) Convert the image to grayscale.

![alt text][image1]

#### b) Pass the image through a Gaussian filter. Kernel size of 5 has been selected.

#### c) Run Canny line detection using 50 and 150 as low and high threshold respectively.
![alt text][image2]
#### d) Mask the edge image with a four side polygon. Relative percentages of the image had been used in order to adapt well to different resolutions:
		Bottom left: 5%(x) - 100%(y)
		Bottom right: 95%(x) - 100%(y)
		Upper left: 47% (x) - 60% (y)
		Upper right: 53% (x) - 60% (y)
This mask will change dependendig of the offset of the camera respect the middle of the vehicle and focal length, but it adapts well to all images and videos of the current project.

![alt text][image3]

#### e) Hough transform to find all the segments in the image. Parameters used: rho=2, theta=pi/180, threshold=10, min_line_length= 40, max_line_gap=20
#### f) Merges the output of the Hough transform with the original image to represent the lines on it.
	
In order to draw a single line on the left and right lanes, I have modified the draw_lines() function by introducing the following steps:

#### a) Split segments in left and right according its slope. Filter also by slope limits. 0.4 to 0.8 (right) and -0.4 to -0.8 (left). The rest of the lines are discarded.
	
#### b) Average slope and bias has been performed doing a Linear Regression of the points that form all the segments in each side. 
	
#### c) Extrapolate and plot the line (slope and bias) found in b) to the bottom of the image 100% (y) and to the 60%(y) on the top. Done for both sides L and R. Below we can find one example of the final result.
All the results can be found under the test_images_output folder.
![alt text][image4]


#### d) Only for video - In this case a moving average of the slope detected is done. A circular window of 20 frames had been selected. In every frame the newest calculated slope is added and the oldest one removed. Then the 20 values are averaged. In the case that no lines are detected no changes are done and it will keep the previous values.
The sample videos could be found [here](test_videos).

The video after the transformation are saved on the [test_videos_output](test_videos_output) directory.

All the videos can be found under the test_videos_output folder.	


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when no lines are detected for a lot of frames. In this case I discard that frames and use the information from the previous ones but it can lead to errors if the number of noisy frames is too high.

Another shortcoming could be if the camera mount is offseted differently. New polygon mask values should be set, it is not self adapting.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to discard the points from the linear regression that are above standard deviation to remove noise and after performe the linear regresion again.

Another potential improvement could be to reprogram the code into classes to be more clear and reusable.
