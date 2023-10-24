import cv2
from multiprocessing import Process
from ai_module.recognition import FaceRecognition
from resource_manager.resource_manager import ResourceManager


class MainWindow:
    __instance = None
    resource_manager = ResourceManager.get_instance()
    rec = FaceRecognition.get_instance()

    @staticmethod
    def get_instance():
        if MainWindow.__instance == None:
            MainWindow.__instance == MainWindow()
        return MainWindow.__instance
    
    def __init__(self):
        if MainWindow.__instance != None:
            raise Exception("Singleton Instance Already Created")
        else:
            MainWindow.__instance = self
        

    def capture_frame(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open the camera.")
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    self.frame_data = frame
                    self.resource_manager.add_item('frame', frame)
                    
                    # cv2.imshow("capture", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    print("Frame not available")

        cap.release()
        cv2.destroyAllWindows()


    def start_recognition(self):
        while True:
            frame = self.resource_manager.get_item_value('frame')
            if frame is not None:
                success,frame = self.rec.recognize_person(frame)
                
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def main():
    obj = MainWindow.get_instance()
    
    capture_process = Process(target=obj.capture_frame)
    capture_process.start()

    recognition_process = Process(target=obj.start_recognition)
    recognition_process.start()

    # capture_process.join()
    # recognition_process.join()


if __name__ == "__main__":
    main()
    while True:
        pass

