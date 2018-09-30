# Machine Learning Engineer Nanodegree
## Capstone Proposal
Hossam Fawzy Elsafty <br>
Septmber 27th,2018

## Dogs vs. Cats Classification

###  Introduction
you'll write an algorithm to classify whether images contain either a dog or a cat.  This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult.

![alt text](https://raw.githubusercontent.com/hosamelsafty/Cats-VS-Dogs/master/woof_meow.jpg?token=AgwGphlRy5nAWk6QdJGdJ4Vro5N-aZuTks5btgZYwA%3D%3D)



### The Asirra data set
Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords.

Asirra (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Many even think it's fun! Here is an example of the Asirra interface:

Asirra is unique because of its partnership with Petfinder.com, the world's largest site devoted to finding homes for homeless pets. They've provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States. Kaggle is fortunate to offer a subset of this data for fun and research. 

### Domain Background
Deep Learning is one of the most promising field in machine learning and used to achieve many classification of images and detect the type of objects in pictures,Computer vision and pattern recognition such as ( Reenacting politicians, Restore colors in B&W photos and videos , Pixel restoration CSI style). we will use Convolutional Neural Networks to train classifer if the picture contain dogs or cats that will help as to use many application like CAPTCHA.
Convolutional Neural Networks is used in many feild similar to this problem like Flowers Recognition [kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition) . 
Also This problem was a challenge on Kaggle on this [link](https://www.kaggle.com/c/dogs-vs-cats).
resources: [link] (http://www.yaronhadad.com/deep-learning-most-amazing-applications/)
  
### Problem Statement
This is a Classification problem given Thousands of cats and dogs picture , We need to classify if the picture is picture of dog or cat.

### Datasets and Inputs
The training archive contains 25,000 images of dogs and cats. Train your algorithm on these files and predict the labels for test1.zip (1 = dog, 0 = cat).

we will train in this data in this [link](https://www.kaggle.com/c/dogs-vs-cats/data).
### Solution Statement
1- load data.
2- split the training into training and validation data.
3- preprocessing the data such as: (resize, RGB color value 0~1). 
4- build CNN model from skratch (using keras)
5- build model using Transfer Learning :
VGG-19 bottleneck features
ResNet-50 bottleneck features
Inception bottleneck features
Xception bottleneck features

and choose the best one with high accuracy.

### Benchmark Model
in the kaggle competition in the loadboard the best time that make high score is :Pierre Sermanet score is :0.98533
the score in the competition is the percentage of correct prediction in the testing set.

### Evaluation Metrics
Accuracy in classification problems is the number of correct predictions made by the model over all kinds predictions made.
![alt text](https://raw.githubusercontent.com/hosamelsafty/Cats-VS-Dogs/master/1_5XuZ_86Rfce3qyLt7XMlhw.png?token=AgwGpjsDQHRUz0tg5fOfw_MopxTykxLuks5bujGAwA%3D%3D)
[resources](https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)
### Project Design

The project will be processed in steps:
- Data Analysis
	- Data reading
	- Data visualization – using maps, heat maps and contour plots to observe the crime distribution by different features. 
- Data preprocessing
	- Relevant information 
	- Data transformation
	- Dataset split
	- Dataset reduction (clustering)
- Models implementation
	- Implement many algorithms to compare them
- Models evaluation
	- Evaluate each algorithm and choose the best using the evaluation metrics.
- Conclusion and result


**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
