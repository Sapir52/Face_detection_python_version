import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import emoji
from PIL import Image
from tensorflow.keras.models import load_model

class ShowPrediction():
    def get_model(self, name_json, nane_model_agr):
        '''
        A method that performs importing models
        :param name_json: Name of the face module
        :param nane_model_agr: Name of the age, gender, race module
        :return: model_emotion_detection, classifier, model_age_gender_race
        '''
        # Load model emotion detection from JSON file
        json_file = open(name_json + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_emotion_detection = model_from_json(loaded_model_json)
        # Load weights and them to model
        model_emotion_detection.load_weights(name_json + '.h5')
        classifier = cv2.CascadeClassifier('file_input/haarcascade_frontalface_default.xml')
        # Load age, gender and race model
        model_age_gender_race = load_model(nane_model_agr)
        return model_emotion_detection, classifier, model_age_gender_race

    # ----------------------------------- image
    def image(self):
        '''
        A method that returns the images submitted by the user after
        identifying emotions, age, gender and committee by face
        '''
        for subdir, dirs, files in os.walk('static/upload_image/'):
            for file in files:
                img = cv2.imread('static/upload_image/' + file)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = self.classifier.detectMultiScale(gray_img, 1.18, 6)
                resized_img = self.show_data(img, gray_img, faces_detected, file)
                cv2.imwrite(os.path.join('static/output_images/', file), resized_img)  # save image

                # This statement only works once per frame. Basically, if we get a key, and that key is q,
                # we'll get out of the while loop with close all imshow windows ().
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    def preprosscing_image(self, x, y, w, h, img, gray_img, file):
        '''
        :params x, y, w, h : Sizes of face identified
        :param img:
        :param gray_img:
        :param file: Image name for identification
        :return: img_pixels: Vector of image pixels
        '''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        array = np.array(roi_gray)
        new_image = Image.fromarray(array)
        new_image.save(file + '.jpg')
        return img_pixels

    def show_data(self, img, gray_img, faces_detected, file="None"):
        '''
        A method that displays the results of the models on the data received
        :param img:
        :param gray_img:
        :param faces_detected:
        :return: resized_img - A data array that represents the data
        '''
        for (x, y, w, h) in faces_detected:
            # preprosscing image
            img_pixels = self.preprosscing_image(x, y, w, h, img, gray_img, file)
            # Get predict and name predict of age, gender and race
            age,max_age, gender, max_gender, race, max_race = self.get_gender_race_age(file + '.jpg')
            # Get emotion recognition by face
            name_predicted_emotion, vector_predictions = self.get_face_emotion_detection(img_pixels)
            # Get emojize
            self.get_emojize(name_predicted_emotion)
            # The model prediction results will be displayed to the user
            data_emotion = name_predicted_emotion + " " + str( int(np.amax(vector_predictions) * 100)) + "%"
            data_agr = gender + " " + max_gender+ "%," + race + " " + max_race +"%," + age + " "+ max_age+"%"
            # Draw bounding boxes and labels on image
            cv2.putText(img, data_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)
            cv2.putText(img, data_agr, (int(x), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0 ,255), 2)
        resized_img = cv2.resize(img, (1024, 768))  # resize for image
        return resized_img




    # ----------------------------------- live camera
    def generate_frames_camera(self):
        '''
        A method that returns the live camera submitted by the user after
        identifying emotions, age, gender and committee by face
        '''
        # Return video from the first webcam on your computer
        cap = cv2.VideoCapture(0)
        # Initiates an infinite loop (to be broken later by a break statement),
        # where we have ret and frame being defined as the cap.read().
        # Basically, ret is a boolean regarding whether or not there was a return at all,
        # at the frame is each frame that is returned.
        # If there is no frame, you wont get an error, you will get None.
        while True:
            ret, img = cap.read()
            if not ret:
                break

            frame = self.get_resized_img(img)
            # This statement just runs once per frame. Basically, if we get a key, and that key is a q, we will exit the while loop with a break.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        self.releases_webcam(cap)

    def releases_webcam(self, cap):
        # This releases the webcam, then closes all of the imshow() windows.
        cap.release()
        cv2.destroyAllWindows()
    # ------------------------------------------------------------

    def get_emojize(self, name_predicted_emotion):
        '''
        A method that prints emoji for the type of emotion received
        :param name_predicted_emotion: Name of emotion prediction
        :return: None
        '''
        # Get emojize
        get_emoji_name = self.statusIcons.get(name_predicted_emotion)
        tick = str(emoji.emojize((get_emoji_name)))
        print(tick)

    def get_face_emotion_detection(self, img_pixels):
        '''
        :param img_pixels: Vector of image pixels
        :return: name_predicted_emotion - Name of emotion prediction,
                 predictions - Vectors of predicting results
        '''
        predictions = self.model_emotion_detection.predict(img_pixels)
        # Get an index of the emotion type with the Maximum Prediction
        max_index = int(np.argmax(predictions))
        # Types of emotions
        emotions = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # Get the name of the emotion
        name_predicted_emotion = emotions[max_index]
        return name_predicted_emotion, predictions

    def get_gender_race_age(self, img_name):
        '''
        :param img_name: Name of the image
        :return: age, gender, race - Result of model prediction
        '''
        age_pred, gender_pred, race_pred = self.model_age_gender_race.predict(self.loadImage(img_name))
        print("age_pred:",age_pred[0])
        race = self.dataset_dict['race_id'][np.argmax(race_pred[0])]
        age = self.dataset_dict['age_id'][np.argmax(age_pred[0])]
        gender = self.dataset_dict['gender_id'][np.argmax(gender_pred[0])]
        max_gender = str( int(np.amax(gender_pred[0]) * 100))
        max_age = str( int(np.amax(age_pred[0]) * 100))
        max_race = str( int(np.amax(race_pred[0]) * 100))
        print("max_age, max_race,max_gender:",max_age, max_race,max_gender)
        return [age, max_age, gender, max_gender, race, max_race]

    def loadImage(self, filepath):
        test_img = image.load_img(filepath, target_size=(198, 198))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        test_img /= 255
        return test_img

    def get_resized_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.classifier.detectMultiScale(gray_img, 1.2, 6)
        resized_img = self.show_data(img, gray_img, faces_detected)
        ret, buffer = cv2.imencode('.jpg', resized_img)
        resized_img = buffer.tobytes()
        return resized_img
    # ----------------------------------- live video
    def video(self):
        '''
        A method that returns the videos submitted by the user after
        identifying emotions, age, gender and committee by face
        '''
        for subdir, dirs, files in os.walk('static/upload_video/'):
            for file in files:
                # Return the selected video
                cap = cv2.VideoCapture('static/upload_video/' + file)
                # Initiates an infinite loop (to be broken later by a break statement),
                # where we have ret and frame being defined as the cap.read().
                # Basically, ret is a boolean regarding whether or not there was a return at all,
                # at the frame is each frame that is returned.
                # If there is no frame, you wont get an error, you will get None.
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break


                    frame = self.get_resized_img(img)
                    # This statement just runs once per frame. Basically, if we get a key, and that key is a q, we will exit the while loop with a break.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # yield frame in byte format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                self.releases_webcam(cap)


    # ------------------------------------------------------------

    def __init__(self):
        '''
        Initialize the attributes of the class
        '''
        self.model_emotion_detection, self.classifier, self.model_age_gender_race = self.get_model(
            'Emotion_detection\Model_emotion_detection_CNN_10000', 'Age_gender_race_detection\Model_agr_CNN')

        self.statusIcons = {
            'Neutral': emoji.demojize('🙂'),
            'Happy': emoji.demojize('😀'),
            'Surprise': emoji.demojize('😳'),
            'Sad': emoji.demojize('😥'),
            'Angry': emoji.demojize('😠'),
            'disgust': emoji.demojize('🤢'),
            'Fear': emoji.demojize('😨')
        }

        self.dataset_dict = {
            'race_id': {
                0: 'white',
                1: 'black',
                2: 'asian',
                3: 'indian',
                4: 'others'
            },
            'gender_id': {
                0: 'male',
                1: 'female'
            },
            'age_id': {
                0: '0-24',
                1: '25-49',
                2: '50-74',
                3: '75-99',
                4: '100-124'
            }
        }

