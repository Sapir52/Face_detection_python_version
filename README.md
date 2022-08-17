# Face detection python version

Face detection is one of the most common applications of Artificial Intelligence.

## The goal of the project:
The purpose of the project is to build a face recognition app. 

The face will allow to predict the emotion, age, gender, race and type of emoji.

## DataSet:
In the project I used two dataSets taken from the Kaggle website.

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

![emotion](https://user-images.githubusercontent.com/63209732/138666921-9e703520-3ec6-4511-b7c7-193b9a67a5ff.png)

![Accuracy_emotion_detection_CNN](https://user-images.githubusercontent.com/63209732/138656977-763fdf1e-de5e-46e5-9b48-e37fdac0f2ec.png)

![Loss_emotion_detection_CNN](https://user-images.githubusercontent.com/63209732/138657004-396a3f5c-cfe8-4720-9fe8-53ff1b451850.png)

For age, gender and race:

![Accurecy_agr_detection_CNN](https://user-images.githubusercontent.com/63209732/138657347-25219dad-60c8-477f-9a75-bc9f85421824.png)

![Loss_agr_detection_CNN](https://user-images.githubusercontent.com/63209732/138660886-02b7c62f-059b-4a06-bcaa-01cb2e62c131.png)

![age](https://user-images.githubusercontent.com/63209732/138667121-c187d620-405e-4416-adac-5ac007f0fb39.png)

![gender](https://user-images.githubusercontent.com/63209732/138667154-21c44b03-cdae-4500-87a5-7a54efc7393e.png)

![race](https://user-images.githubusercontent.com/63209732/138667179-0f1521fc-1b10-4dd7-9306-93c4e45365e2.png)


## The site:

Main page:

![main_img](https://user-images.githubusercontent.com/63209732/138659517-18cfe92f-6c9e-4fd7-aaa7-6d5501143309.png)

For images:

![img_page](https://user-images.githubusercontent.com/63209732/138722508-8d5cae32-9e81-4a2f-9cc5-d6c47a4f6058.png)

For videos:
![video_up](https://user-images.githubusercontent.com/63209732/138660436-091e89da-60dd-42d4-a48a-4aa269fac318.png)

![video_img](https://user-images.githubusercontent.com/63209732/138659540-c1866a9f-8fec-40a0-a3cb-950f586f270d.png)

## Run: 

application.py
  
