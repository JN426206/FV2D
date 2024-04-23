from os import stat
import re
import cv2
import os
import detected_object

DATA_DIR = "data/"
IMAGES_DIR = DATA_DIR + "jsladjf1_good_images/"
LABELS_DIR = DATA_DIR + "jsladjf1_good/"
IMAGE_FILE_EXTENSION = "jpg"
LABEL_FILE_EXTENSION = "txt"
LABEL_SEPERATOR = " " #If tab put /t
PERSON_OBJECT_CLASS = 0
PERSON_BBOX_COLOR = (0,0,255)
BALL_OBJECT_CLASS = 1
BALL_BBOX_COLOR = (0,255,0)
OTHER_BBOX_COLOR = (255,0,0)

class BBox():
    
    def __init__(self, object_class: int, x: float, y: float, width: float, height: float):
        self.__object_class = object_class
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    def get_object_class(self):
        return self.__object_class

    def get_coordinates(self):
        return self.__x, self.__y, self.__width, self.__height

    def get_coordinates_coco(self, image_shape):
        height, width, _ = image_shape
        # const width = cols[3] * image.width
        # const x = cols[1] * image.width - width * 0.5
        # const height = cols[4] * image.height
        # const y = cols[2] * image.height - height * 0.5
        real_width = self.__width * width
        real_x = self.__x * width - real_width * 0.5
        real_height = self.__height * height
        real_y = self.__y * height - real_height * 0.5 
        return int(real_x), \
                int(real_y), \
                int(real_x+real_width), \
                int(real_y+real_height)

    def __str__(self):
        return f"Object class: {self.__object_class} x: {self.__x} y: {self.__y} width: {self.__width} height: {self.__height}"

class Accuracer:

    def __init__(self):
        self.files_id = []
        # self.labels = []

    @staticmethod
    def read_labels_file(file_id):
        labels = []
        with open(LABELS_DIR + file_id + "." + LABEL_FILE_EXTENSION) as f:
            for line in f: #.replace("\n","").replace("\r","").
                values = line.replace("\n","").replace("\r","").split(LABEL_SEPERATOR)
                labels.append(BBox(int(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])))
        return labels
                
    @staticmethod
    def open_image_file(file_id):
        return cv2.imread(IMAGES_DIR + file_id + "." + IMAGE_FILE_EXTENSION)

    @classmethod
    def check_accuracy(self, bboxes, pred_classes, file_id):
        for label_box in Accuracer.read_labels_file(file_id):
            print(label_box)

    @classmethod
    def read_files_id(self):
        for file_name in os.listdir(IMAGES_DIR):
            self.files_id.append(os.path.splitext(os.path.basename(file_name))[0])

    @classmethod
    def get_files_id(self):
        self.files_id = []
        if len(self.files_id) == 0:
            self.read_files_id()
        return self.files_id
    
    # @classmethod
    # def get_labels(self, file_id):
    #     if len(self.labels) == 0:
    #         self.labels = Accuracer.read_labels_file(file_id)
    #     return self.labels

    @classmethod
    def create_labeled_images(self, output_dir):
        for file_id in self.get_files_id():
            print(file_id)
            image = Accuracer.open_image_file(file_id)
            for bbox in Accuracer.read_labels_file(file_id):
                box = bbox.get_coordinates_coco(image.shape)
                bbox_color = OTHER_BBOX_COLOR
                if bbox.get_object_class() == PERSON_OBJECT_CLASS:
                    bbox_color = PERSON_BBOX_COLOR
                elif bbox.get_object_class() == BALL_OBJECT_CLASS:
                    bbox_color = BALL_BBOX_COLOR
                cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), bbox_color, 2)
            # cv2.imshow("TV2D Accuracer", image)
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(f"{output_dir}/{file_id}.{IMAGE_FILE_EXTENSION}", image)

    def create_labeled_image_from_prediction(self, detected_objects, file_id, output_dir):
        image = Accuracer.open_image_file(file_id)
        for detecedObject in detected_objects:
            bbox_color = OTHER_BBOX_COLOR
            if detecedObject.object_class == PERSON_OBJECT_CLASS:
                bbox_color = PERSON_BBOX_COLOR
            elif detecedObject.object_class == BALL_OBJECT_CLASS:
                bbox_color = BALL_BBOX_COLOR
            bbox = detecedObject.bbox
            cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[2],bbox[3]), bbox_color, 2)
        # cv2.imshow("TV2D Accuracer", image)
        # if cv2.waitKey(0) == 27:
        #     break  # esc to quit
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(f"{output_dir}/{file_id}.{IMAGE_FILE_EXTENSION}", image)

if __name__ == "__main__":
    accuracer = Accuracer()
    # accuracer.create_labeled_images("labeled")

    exit(0)