# Face detection python version
Face detection is one of the most common applications of Artificial Intelligence.

## The goal of the project:
The purpose of the project is to build a face recognition app. 

The face will allow to predict the emotion, age, gender, race and type of emoji.

## DataSet:
In the project I used two dataStes taken from the Kaggle website.

For emotion detection:
https://www.kaggle.com/gauravsharma99/ck48-5-emotions

For age, gender and race:
https://www.kaggle.com/jangedoo/utkface-new

## Pipeline:
1. import datasets.
2. Create vectors and labels by classes.
3. Division into train and test.
4. To identify age, gender and race I used a multi-output deep learning model (CNN). The model capable of predicting three distinct outputs. For the purpose of recognizing emotions by face I built a CNN model. In addition, I made a comparison with different classifiers.
5. Evaluation of the models.
6. Build a website in a Flask. The site will allow the user to choose between three options: video, photos or camera. 
* If the user selects images or video, you will need to enter data. 
* Selecting a camera will open the user's computer camera.

## Evaluation of the models:
For emotion detection:

![Accuracy_emotion_detection_CNN](https://user-images.githubusercontent.com/63209732/138656977-763fdf1e-de5e-46e5-9b48-e37fdac0f2ec.png)

![Loss_emotion_detection_CNN](https://user-images.githubusercontent.com/63209732/138657004-396a3f5c-cfe8-4720-9fe8-53ff1b451850.png)

For age, gender and race:

![Accurecy_agr_detection_CNN](https://user-images.githubusercontent.com/63209732/138657347-25219dad-60c8-477f-9a75-bc9f85421824.png)

![Accurecy_agr_detection_CNN](https://user-images.githubusercontent.com/63209732/138657064-d0d28816-d76b-4fd3-8472-51e3eb741bec.png)

## The site
Main page:

For images:

![img_res](https://user-images.githubusercontent.com/63209732/138657376-8d3d63b9-7c50-4fe4-9761-6a23de443b35.png)

For video:

