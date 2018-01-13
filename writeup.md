# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

Overview
---

The steps of this project are the following:
- Load the data set
- Explore, summarize and visualize the data set
- Augment the training data
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images

[//]: # (Image References)

[training_set]: ./writeup_image/training_set.jpg "training_set"
[validation_set]: ./writeup_image/validation_set.jpg "validation_set"
[test_set]: ./writeup_image/test_set.jpg "test_set"

[train_distribution]: ./writeup_image/train_distribution.jpg "train_distribution"
[processed_train_distribution]: ./writeup_image/processed_train_distribution.jpg "processed_train_distribution"
[valid_distribution]: ./writeup_image/valid_distribution.jpg "valid_distribution"
[test_distribution]: ./writeup_image/test_distribution.jpg "test_distribution"

[original_img]: ./writeup_image/original_img.jpg "original_img"
[grayscaled_img]: ./writeup_image/grayscaled_img.jpg "grayscaled_img"
[normalized_img]: ./writeup_image/normalized_img.jpg "normalized_img"
[rotated_img]: ./writeup_image/rotated_img.jpg "rotated_img"
[Perspective_img]: ./writeup_image/Perspective_img.jpg "Perspective_img"
[translate_img]: ./writeup_image/translate_img.jpg "translate_img"

[validation_accracy]: ./writeup_image/validation_accracy.jpg "validation_accracy"

[top5_1]: ./writeup_image/top5_1.jpg "top5_1"
[top5_2]: ./writeup_image/top5_2.jpg "top5_2"
[top5_3]: ./writeup_image/top5_3.jpg "top5_3"
[top5_4]: ./writeup_image/top5_4.jpg "top5_4"
[top5_5]: ./writeup_image/top5_5.jpg "top5_5"
[top5_6]: ./writeup_image/top5_6.jpg "top5_6"
[top5_7]: ./writeup_image/top5_7.jpg "top5_7"
[top5_8]: ./writeup_image/top5_8.jpg "top5_8"
[top5_9]: ./writeup_image/top5_9.jpg "top5_9"

[Conv_Layer_1]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_1"
[Conv_Layer_1_relu]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_1_relu"
[Conv_Layer_1_maxpool]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_1_maxpool"
[Conv_Layer_2]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_2"
[Conv_Layer_2_relu]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_2_relu"
[Conv_Layer_2_maxpool]: ./writeup_image/Conv_Layer_1.jpg "Conv_Layer_2_maxpool"

[Traffic Sign 1]: ./test_imgs/1.jpg "Traffic Sign 1"
[Traffic Sign 2]: ./test_imgs/2.jpg "Traffic Sign 2"
[Traffic Sign 3]: ./test_imgs/3.jpg "Traffic Sign 3"
[Traffic Sign 4]: ./test_imgs/4.jpg "Traffic Sign 4"
[Traffic Sign 5]: ./test_imgs/5.jpg "Traffic Sign 5"
[Traffic Sign 6]: ./test_imgs/6.jpg "Traffic Sign 6"
[Traffic Sign 7]: ./test_imgs/7.jpg "Traffic Sign 7"
[Traffic Sign 8]: ./test_imgs/8.jpg "Traffic Sign 8"
[Traffic Sign 9]: ./test_imgs/9.jpg "Traffic Sign 9"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/x327397818/UDC-Term1-project2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic signs data set:

* The size of training set is `34799`. After Augment, the training set is `313191`.
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32,32,3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It gives 15 images from training, validation and test sets. 

####Training set

![alt text][training_set]

####Validation set

![alt text][validation_set]

####Test set
![alt text][test_set]

Here are the charts that show the data distribution of training/validation/test sets,

#### Training data distribution

![alt text][train_distribution]

#### Validation data distribution

![alt text][Valid_distribution]

#### Test data distribution

![alt text][Test_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to rotate the images to grayscale because the size of a grayscaled image is 32x32x1 which is 1/3 of a RGB image. This saves lots of memory and increase the training speed,

Here is an example of a traffic sign image before and after grayscaling.

![alt text][original_img]    ![alt text][grayscaled_img]

As a last step, I normalized the image data because it scales the data to 0~1, which speeds up the training and improves the performance,

![alt text][original_img]    ![alt text][normalized_img]

I decided to generate additional data because have more training data can improve the accuracy and decrease overfitting.
  

The way I used to add more data includes rotating, perspective, translate,
Here are examples of original images and augmented images:

#### rotate image

![alt text][original_img]    ![alt text][rotated_img]

#### perspective image

![alt text][original_img]    ![alt text][Perspective_img]

#### translate image

![alt text][original_img]    ![alt text][translate_img]

The difference between the original data set and the augmented data set is the following,

 ![alt text][train_distribution]  ![alt text][processed_train_distribution] 

 The data quantity in each label increased 10 times

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Flatten		        | output 400        							|
| Fully Connected		| Input 400, Output 120        			        |
| RELU					|												|
| Fully Connected	    | Input 120, Output 84							|
| RELU					|												|
| Fully Connected	    | Input 84, Output 43							| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet network. In the First run the validation accuracy is around 90, then I start to play around it by following steps,

* 1. Add more training data. I did this by adding augmented image
* 2. Tuning the hyperparameters to get a higher accuracy
* 3. Add a dropout layer to solve overfitting problem

Final parameters,

| Hyperparameters       | Value	 | 
|:---------------------:|:------:| 
| Optimizer         	| ADAM   | 
| the batch size     	| 100 	 |
| number of epochs		| 50	 |
| learning rate	      	| 0.001	 |
| dropout keep_prob	    | 0.75   |

Here is the validation accuracy across the training,

![alt text][validation_accracy]



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of `98.3%`
* test set accuracy of `95.8%`

If a well known architecture was chosen:


* What architecture was chosen?
<br> I choosed LeNet architecture since it is already given in the previous assignment

* Why did you believe it would be relevant to the traffic sign application?
<br> It is widely used in image recognition area.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
<br> The final validation accuracy is 98.3% and test accuracy is 95.8%, which is relatively good. And I used random 9 images from web to test it, the accuracy is 100%. 

### Test a Model on New Images

#### 1. Choose nine German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][Traffic Sign 1] ![alt text][Traffic Sign 2] ![alt text][Traffic Sign 3] ![alt text][Traffic Sign 4] ![alt text][Traffic Sign 5] ![alt text][Traffic Sign 6] ![alt text][Traffic Sign 7] ![alt text][Traffic Sign 8] ![alt text][Traffic Sign 9]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycles crossing      								| Bicycles crossing 									| 
|Speed limit (30km/h)     								| Speed limit (30km/h) 									|
| Right-of-way at the next intersection					| Right-of-way at the next intersection					|
| General caution	      								| General caution					 					|
| Road work												| Road work     										|
| Children crossing										| Children crossing     								|
| Priority road											| Priority road    										|
| Bumpy road											| Bumpy road   											|


The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.8%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following image shows the top 5 softmax probabilities for each image,

#### Image1

![alt text][top5_1]

#### Image2

![alt text][top5_2]

#### Image3

![alt text][top5_3]

#### Image4

![alt text][top5_4]

#### Image5

![alt text][top5_5]

#### Image6

![alt text][top5_6]

#### Image7

![alt text][top5_7]

#### Image8

![alt text][top5_8]

#### Image9

![alt text][top5_9]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Here is the visualizing of my neural Network,

#### Conv_Layer_1:
![alt text][Conv_Layer_1]
#### Conv_Layer_1_relu:
![alt text][Conv_Layer_1_relu]
#### Conv_Layer_1_maxpool:
![alt text][Conv_Layer_1_maxpool]
#### Conv_Layer_2：
![alt text][Conv_Layer_2]
#### Conv_Layer_2_relu:
![alt text][Conv_Layer_2_relu]
#### Conv_Layer_1_maxpool:
![alt text][Conv_Layer_2_maxpool]




