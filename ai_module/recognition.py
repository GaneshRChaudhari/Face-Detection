import os
import cv2
import face_recognition
import joblib


class FaceRecognition:
    __instance = None
    model_filename = "ai_module/face_recognition_model.clf"

    @staticmethod
    def get_instance():
        if FaceRecognition.__instance == None:
            FaceRecognition.__instance = FaceRecognition()
        return FaceRecognition.__instance
    

    def __init__(self):
        if FaceRecognition.__instance != None:
            raise Exception("Singleton Instance Already Present")
        else:
            FaceRecognition.__instance = self


    def recognize_person(self, frame):
        h, w = frame.shape[:2]
        try:
            knn_clf, saved_encodings, saved_labels = self.check_and_load_model()

            if knn_clf is None:
                cv2.putText(frame, f"MODEL NOT AVAILABLE", (h//2, w//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return False, frame

            # Encode the face from the input image
            input_face_encoding = face_recognition.face_encodings(frame)

            # Ensure that only one face is detected in the input image
            if len(input_face_encoding) != 1:
                cv2.putText(frame, f"NO FACE OR MULTIPLE FACE DETECTED", (h//2, w//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return False, frame

            faces = self.get_faces(frame)

            user_name = self.find_matches(knn_clf, input_face_encoding, saved_labels)

            self.add_text_to_frame(faces, frame, user_name)
            return True, frame
        except Exception as err:
            print("EXCEPTION IN RECOGNITION : ", err)
            return False, frame
        
    
    # Predict the person based on the trained model
    def find_matches(self, knn_clf, input_face_encoding, saved_labels):
        closest_distances = knn_clf.kneighbors(input_face_encoding, n_neighbors=3)
        matches = [closest_distances[0][i][0] <= 0.6 for i in range(1)]

        if matches[0]:
            label = knn_clf.predict(input_face_encoding)[0]
            # Find the name associated with the label
            for i, enc_label in enumerate(saved_labels):
                if enc_label == label:
                    name = saved_labels[i]
        else:
            name = "unknown"
        return name
        
    
    def get_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    

    def add_text_to_frame(self, faces, frame, text):
        for (x, y, w, h) in faces:
            cv2.putText(frame, f"{text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

        
    def check_and_load_model(self):
        knn_clf, saved_encodings, saved_labels = None, [], []
        if os.path.exists(self.model_filename):
            knn_clf, saved_encodings, saved_labels = joblib.load(self.model_filename)
        return knn_clf, saved_encodings, saved_labels    



# obj = FaceRecognition.get_instance()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if ret:
#         success, frame = obj.recognize_person(frame)
#         cv2.imshow('frame', frame)

#         key = cv2.waitKey(1)

    
