import abc
from abc import ABC, abstractmethod
from detected_object import DetectedObject
class Detector(ABC):
        
    @abstractmethod
    def __init__(self, model_threshold, model_path):
        pass
    
    @abstractmethod
    def detect_objects(self, image, classes = [0, 32]) -> [DetectedObject]:
        pass
