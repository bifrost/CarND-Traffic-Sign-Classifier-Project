# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram-class-count.png "Visualization"
[image2a]: ./examples/sign-colors.png "Color"
[image2b]: ./examples/sign-gray.png "Grayscaling"
[image3]: ./images/double_curve.jpeg "Traffic Sign 1"
[image4]: ./images/road_work.jpeg "Traffic Sign 2"
[image5]: ./images/70.jpeg "Traffic Sign 3"
[image6]: ./images/Right-of-way.jpeg "Traffic Sign 4"
[image7]: ./images/priority_road.jpeg "Traffic Sign 5"
[image8]: ./images/Traffic_signals.png "Traffic Sign 6"
[image9]: ./examples/featuremap.png "FeatureMaps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bifrost/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of classes from 0 to 42.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the structure seems to be more important for the classification than the colors. I tried both RGB and HLS images as well as single channels but grayscale seems to work best. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2a]

![alt text][image2b]

I did not normalize the image before processing because I am using batch_normalization as first layer in the neural network.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| batch_normalization	|												|
|						|												|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 28x28x20 	|
| batch_normalization	|												|
| dropout				| keep_prop 0.5									|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x20 				|
|						|												|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x40 	|
| batch_normalization	|												|
| dropout				| keep_prop 0.5
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x40 					|
|						|												|
| flatten				| 												|
|						|												|
| Fully connected		| 200											|
| batch_normalization	|												|
| dropout				| keep_prop 0.5									|
| RELU					|												|
|						|												|
| Fully connected		| 200											|
| batch_normalization	|												|
| dropout				| keep_prop 0.5									|
| RELU					|												|
|						|												|
| Fully connected		| 43											|
| Softmax				| 												|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a `learning rate = 0.0005`, `EPOCHS = 80` and `BATCH_SIZE = 128`.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.969 
* test set accuracy of 0.946

An iterative approach was chosen:
* First I tried to use Inception modules, but it seems to be overkill for the task and fast I reached a much better performance with a modified LeNet. I choosed LeNet because we have seen it worked well on small images.
* The most important modification I did to LeNet was adding batch_normalization as first layer and dropout layers so I could increase the model size and parameters without overfitting.
* The most important parameters I had to tune was the learning rate. I had to lower it a bit to increase accuracy on the training sets. The dropout rate was important to increase accuracy on the validation sets.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because the sign is very light.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image									| Prediction							| 
|:-------------------------------------:|:-------------------------------------:| 
| Double curve							| Double curve   						| 
| Road work								| Road work 							|
| 70 km/h								| 70 km/h								|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Priority road							| Priority road							|
| Traffic signals						| Priority road							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This is far away as good as the accuracy of the test sets 94%. But the comparison is not fair because we need much more than 6 images to compare the different data sets.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 56th cell of the Ipython notebook.

For image 1, 3, 4 and 5 the model is very sure about the prediction (probability > 0.94). It was relatively sure that image 2 is Road work (probability of 0.68) meanwhile very uncertain about image 6 (probability of 0.36) that it also predicted wrong.

![alt text][image3]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 94.78%				| Double curve   						| 
| 2.08%					| Road narrows on the right 			|
| 1.89%					| Road work								|
| 0.45%					| Wild animals crossing					|
| 0.24%					| Speed limit							|

![alt text][image4]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 68.27%				| Road work   							| 
| 8.02%					| Road narrows on the right 			|
| 6.76%					| Bicycles crossing						|
| 4.01%					| Bumpy road							|
| 3.77%					| Children crossing						|

![alt text][image5]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 98.00%				| Speed limit (70km/h)					| 
| 1.96%					| Speed limit (20km/h) 					|
| 0.03%					| Speed limit (30km/h)					|
| 0.01%					| Speed limit (60km/h)					|
| 0.00%					| Keep left								|

![alt text][image6]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 100.00%				| Right-of-way at the next intersection	| 
| 0.00%					| Pedestrians 							|
| 0.00%					| Beware of ice/snow					|
| 0.00%					| General caution						|
| 0.00%					| Children crossing						|

![alt text][image7]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 100.00%				| Priority road							| 
| 0.00%					| Roundabout mandatory 					|
| 0.00%					| No entry								|
| 0.00%					| Stop									|
| 0.00%					| Speed limit (50km/h)					|

![alt text][image8]

| Probability			| Prediction							| 
|:---------------------:|:-------------------------------------:| 
| 35.73%				| Priority road							| 
| 27.43%				| Speed limit (80km/h)					|
| 8.04%					| Road work								|
| 5.66%					| Speed limit (60km/h)					|
| 4.51%					| Speed limit (30km/h)					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The image below shows the feature maps from first convolution layer. It is not clear to me what characteristics the neural network use to make classifications. But most of the feature maps seems to be a kind of learned edge detectors except from feature map 5 and 9.

![alt text][image9]


