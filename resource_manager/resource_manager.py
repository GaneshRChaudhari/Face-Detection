from multiprocessing import Manager, Lock
import resource_manager.local_dict as local_dict


class ResourceManager:
    __instance = None

    @staticmethod
    def get_instance():
        if ResourceManager.__instance == None:
            ResourceManager.__instance = ResourceManager()
        return ResourceManager.__instance


    def __init__(self):
        if ResourceManager.__instance != None:
            raise Exception("Singleton Instance Already Present")
        else:
            ResourceManager.__instance = self

        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.shared_dict_lock = Lock()

        for key in local_dict.data.keys():
            self.add_item(key, local_dict.data[key])       


    def add_item(self, key, value):
        with self.shared_dict_lock:
            self.shared_dict[key] = value

    def get_item_value(self, key):
        with self.shared_dict_lock:
            frame = self.shared_dict[key]
            return frame