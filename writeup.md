# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/dataset_raw.png "Visualization"
[image2]: ./img/dataset_prep.png "Gray scale and Normalize"
[image3]: ./img/dataset_aug.png "Augmented data"
[image4]: ./img/traffic-signs-architecture.png "Model Architecture"
[image5]: ./img/histogram.png "Classes distribution"
[image6]: ./img/custom.png "Custom Traffic Signs"
[image7]: ./img/probability1.png "Probabilities custom image 1 classification"
[image8]: ./img/probability2.png "Probabilities custom image 2 classification"
[image9]: ./img/probability3.png "Probabilities custom image 3 classification"
[image10]: ./img/probability4.png "Probabilities custom image 4 classification"
[image11]: ./img/probability5.png "Probabilities custom image 5 classification"
[image12]: ./img/probability6.png "Probabilities custom image 6 classification"
[image13]: ./img/probability7.png "Probabilities custom image 7 classification"
[image14]: ./img/probability8.png "Probabilities custom image 8 classification"
[image15]: ./img/probability9.png "Probabilities custom image 9 classification"
[image16]: ./img/probability10.png "Probabilities custom image 10 classification"
[image17]: ./img/layer1.png "Layer 1 Features map"
[image18]: ./img/layer2.png "Layer 1 Features map"


## Rubric Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/saultr/project_2_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I have used pytnon and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows each sign class paired by [label/picture] from 0 to 42

![alt text][image1]

The following graph represents the classes distribution on the train set. We can see than the datasets are not well balance. We see the same distribution for the test set.

![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) mentioned in their paper, using color channels didn’t seem to improve things a lot. After some tests I got the best performance using only the Green channel fron the RGB image.

As second step, I normalize the values from [0 255] to [-0.4 1.0] looking for a 0 mean and 0 variance.

Here is the exploratory visualization of the data set already preprocesed.

![alt text][image2]

As a last step, I decided to generate additional data because there were classes with less than 200 samples being not enough for training. I decided to add an extra 5 images per feature.

To add more data to the the data set, I used the following techniques:

 	1. Rotation randomly up to 20 degrees.
	
	2. Translate the image randomly in both axis.
	
	3. Shear the shape of the image randomly in both axis.

Here is an example of an original image and an augmented image:

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Max pooling/dropout* 	| 4x4 stride,  outputs 4x4x32 *				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 			    	|
| Max pooling/dropout* 	| 2x2 stride,  outputs 4x4x64* 				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x128     |
| RELU					|												|
| Max pooling/dropout  	| 2x2 stride,  outputs 4x4x128* 			    |
| Flatten				| merge and flatten all pooling*, output 3584   | 
| Fully conn/dropout    | output 1024     								|
| Softmax				| output 43       								|
*Fully con to flatten

![alt text][image4]
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning ratio of 0.001. The batch size was 128 as I was a bit limited by the GPU memory. 
Number of epochs used were 20 because from that point the system was overfitting and better results in training didn't follow with better results in testing.

The dropouts hyperparameters were (keep_p) for each layer:

			Type           Size         keep_p      Dropout
	Layer 1        5x5 Conv       32           0.9         10% of neurons
	Layer 2        5x5 Conv       64           0.8         20% of neurons
	Layer 3        5x5 Conv       128          0.7         30% of neurons
	Layer 4        FC             1024         0.5         50% of neurons
	


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.951 
* test set accuracy of 0.960

* What was the first architecture that was tried and why was it chosen?
	
	At the begining I have chosen LeNet model but the network was underfitting not being able to pass an accuracy of 0.92. After I decided to change to the recommended model in Pierre Sermanet and Yann LeCun document. 

* What were some problems with the initial architecture?
	
	The initial arquitecture was not capturing enough details. None of the layers were passed in full connection to the softmax funtion losing interesting information from the first layers, specially the ones refered to sign shape.

* What final architecture was chosen
	
	The 4 layer model recommended in [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) mentioned in their paper and in [Alex Staravoitau project](https://navoshta.com/traffic-signs-classification/). It has 3 convolutional layers for feature extraction and one fully connected layer as a classifier.
	
* Why did you believe it would be relevant to the traffic sign application?
	
	As opposed to usual strict feed-fordward CNNs this architecture is not only forwarding the convolutional layers to hte subsequent layer, but is also brached off and fed into the classifier (fully connected layer). That allows to not only use dropouts before the classifier layer but also with previous convolutional layers making a slight improvement in performance.  

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 	
	The system is overfitting because is achieving 0.99 in trainning and only 0.951 in the validation set. A posible solution will be using early stoping technique to prevent it, but has not being used here. The test set is performing well, similar or better to the validation set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6]

Image 1: It is a 'stop' sign with good quality but in a forest with lot of tree braches covering it. It can be difficult for the classifier for these black lines (branches) crossing over the sign.

Image 2: 'no vehicles' sign. The picture has been taking at night and the white balance is towards green channel and alos a bit blurry.

Image 3: It is a LED night 'speed limit (120km/h)' sign. The round circle despite being red is more narrow than normal and the numbers a bit bigger and made of LEDs. Probably impossible for the calssifier.

Image 4: 'Dangerous curve to the right' sign. The sign is not perpendicular and it is a bit oblique.

Image 5: 'No passing' sign. It is very dirty and the right car looks like has been painted. Could be a problem for the classifier.

Image 6: 'Speed limit (30km/h)' sing. Red circle is really worn out. It can be a problem for the classifier not detecting the circle.

Image 7: 'Keep left' sign. It has good quality but has a water mark on top of it. It should not be any problem for the classifier.

Image 8: 'chindren crossing' sign. Very similar to bicycle crossing, snow or wild animals. 

Image 9: 'no passing for vehicles over 3.5 metric tones. It is quite clear, it should not have any problem for the classifier.

Image 10: 'traffic signals'. Also very clear image, just a bit rotated towards the left, but should not be a problem to be correctly classified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Labels:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[14 15 8 20 9 1 39 28 10 26]

Predicted: [14 15 14 26 9 1 39 29 10 26]

Hits:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7 / 10


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. This is quite below the test set accuracy of 0.96, but it is normal due to the particularities or difficulties some sign had. 

Image 1: The classifier has no problem to do a correct prediction despite some branches on top of the sign.

Image 2: Correct prediction despite the different white balance of the image. Doesn't seem to affect much on the decission.

Image 3: This sign was really difficult for the classifier. It is a nigh LED signal and borders, numbers and background is very different than the standard one. It is detecting a stop instead.

Image 4: It is surprissing that the classifier is failing in this one becouse apparently it looks easy. Perspective is not right so probably the augmented data has not enought gain while doing the perspective distortion. It will be tested with more.

Image 5: This sign has been surprissing correctly classified despite being really dirty and having right car masked with some paint. Perspective it is also not fully perpendicular.

Image 6: Good job in this image for the model. The red circle is really worn out and despite of being a difficult one it has success. It is on the limit as we can see in the next point having more distribute probabilities than others.   

Image 7, 8, 9, 10: This set of images were really standard with no big particularities, and all four were perfectly classified.
 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is really on the limit to make the correct decission, having similar probabilities than 'Bicycles crossing sign'.

For the rest of the images the decission is clear and the system have big waranties.
  
Stop 

![alt text][image7]

No vehicles 

![alt text][image8]

Speed limit (120km/h) 

![alt text][image9]

Dangerous curve to the right

![alt text][image10]

No passing

![alt text][image11]

Speed limit (30km/h)

![alt text][image12]

Keep left

![alt text][image13]

Children crossing

![alt text][image14]

No passing for vehicles over 3.5 metric tons

![alt text][image15]

Traffic signals

![alt text][image16]
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

* Layer 1: 
	In the first convolutional layer it is clear that the shape of the sign is the charasteristic that the neural working is using to classify.

![alt text][image17]



* Layer 2:
	In the second convolutional layer we can apreciate some 'V' botton shapes and some diagonal lines aswell.
![alt text][image18]

