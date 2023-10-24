import os
import face_recognition
import joblib


class FaceRegistration:
    __instance = None
    dataset_dir = "ai_module/facebank"
    model_filename = "ai_module/face_recognition_model.clf"

    @staticmethod
    def get_instance():
        if FaceRegistration.__instance == None:
            FaceRegistration.__instance == FaceRegistration()
        return FaceRegistration.__instance
    
    def __init__(self):
        if FaceRegistration.__instance != None:
            raise Exception("Singleton Instance Already Present")
        else:
            FaceRegistration.__instance = self

    def train_model(self):
        try:
            new_face_encodings, new_labels = [], []

            for image_file in os.listdir(self.dataset_dir):
                image_path = os.path.join(self.dataset_dir, image_file)

                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)

                # Ensure that only one face is detected in each image
                if len(face_encoding) == 1:
                    new_face_encodings.append(face_encoding[0])
                    new_labels.append(self.dataset_dir)

                # os.remove(image_path)
            self.save_encoding(new_face_encodings, new_labels)
            
        except Exception as err:
            print("Exception ocurred while training model", err)


    def save_encoding(self, new_face_encodings, new_labels):
        knn_clf, saved_encodings, saved_labels = self.check_and_load_model()

        combined_encodings = saved_encodings + new_face_encodings
        combined_labels = saved_labels + new_labels

        if knn_clf is None:
            knn_clf = self.create_new_model()

        # Train the model with the combined data
        knn_clf.fit(combined_encodings, combined_labels)

        # Save the updated model to a file
        joblib.dump((knn_clf, combined_encodings, combined_labels), self.model_filename)
        print(f"Model updated and saved as {self.model_filename}")


    def check_and_load_model(self):
        knn_clf, saved_encodings, saved_labels = None, [], []
        if os.path.exists(self.model_filename):
            knn_clf, saved_encodings, saved_labels = joblib.load(self.model_filename)
        return knn_clf, saved_encodings, saved_labels
    

    # function to Create a new model
    def create_new_model(self):
        from sklearn import neighbors
        return neighbors.KNeighborsClassifier(n_neighbors=3)
    

# obj = FaceRegistration.get_instance()
# obj.train_model()
