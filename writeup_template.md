#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/train_label.jpg "Training label distribution"
[image2]: ./examples/valid_label.jpg "Validation label distribution"
[image3]: ./examples/test_label.jpg "Test label distribution"
[image4]: ./examples/image_rotation.jpg "Example for image rotation"
[image5]: ./test-examples/0.jpg "Traffic sign 1"
[image6]: ./test-examples/1.jpg "Traffic sign 2"
[image7]: ./test-examples/2.jpg "Traffic sign 3"
[image8]: ./test-examples/3.jpg "Traffic sign 4"
[image9]: ./test-examples/4.jpg "Traffic sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python/numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of labels.

![training label distribution][image1]
![validation label distribution][image2]
![test label distribution][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the image using np.sum(x/3, axis=3, keepdims=True). However, it did not seem to help improving accuracy ...

Then I try to do oversampling to produce under-sampled data, after which all labels have the same number of training samples. However, the model sometimes seem to work well (achieve accuracy > 0.93), but sometimes it stuck at a lower level.

Since the training accuracy sometimes stuck at a low point (~90%), I realize the model is under fitting and it maybe caused by not enough data. Then I tried generating fake data. I tried two ways: 1) simply copy/paste existing images. 2) create new images with angle rotation. 1) did not work out well while 2) works pretty good.

Here is an example of a traffic sign image before and after rotation.

![example of image rotation][image4]

I have generated 5x new data, so now I have enough data for training, and the result is great.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | input 5x5x16,  outputs 400                    |
| Fully connected       | input 400, output 120    						|
| Dropout               | keep_prob 0.5                                 |
| Fully connected		| input 120, output 84        					|
| Dropout               | keep_prob 0.5                                 |
| Fully connected		| input 84, output 43       					|
| Softmax				|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, with params as below:
* rate = 0.0005
* EPOCHS = 50
* BATCH_SIZE = 128

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.948
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? 
LeNet

* Why did you believe it would be relevant to the traffic sign application?
As suggested by project description: "it's plug and play"

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
All training/validation/test has accuracy >= 0.940
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]

The first image seem to be easy.
The second image seem to be difficult because the left half of sign is not standard round. 
The fourth image is hard because the sign is smaller than usual size.
The third and fifth image seem to be hard because it is dark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			         |     Prediction	        					| 
|:----------------------:|:--------------------------------------------:| 
| 3.5 metrics prohibited | 3.5 metrics prohibited						| 
| 30 km/h       		 | 30 km/h    									|
| Keep right	         | Keep right									|
| Turn right ahead 	     | Turn right ahead			    				|
| Right-of-way  	     | Right-of-way     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is higher than the accuracy on the test set of original dataset.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all five images, the model is pretty sure about the predication, which are the correct answers. Please find the results of top5 softmax on notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


