# Face detection python version
Face detection is one of the most common applications of Artificial Intelligence.

## The goal of the project:
The purpose of the project is to build a face recognition app. 

The face will allow to predict the emotion, age, gender, race and type of emoji.

## DataSet
In the project I used two dataStes taken from the Kaggle website.

For emotion detection:
https://www.kaggle.com/gauravsharma99/ck48-5-emotions

For age, gender and race:
https://www.kaggle.com/jangedoo/utkface-new

## Pipeline
1. import datasets.
2. Create vectors and labels by classes.
3. Division into train and test.
4. To identify age, gender and race I used a multi-output deep learning model (CNN). The model capable of predicting three distinct outputs. For the purpose of recognizing emotions by face I built a CNN model. In addition, I made a comparison with different classifiers.
5. Evaluation of the models.
6. Build a website in a Flask. The site will allow the user to choose between three options: video, photos or camera. 
* If the user selects images or video, you will need to enter data. 
* Selecting a camera will open the user's computer camera.




