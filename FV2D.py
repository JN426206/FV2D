import sys
import os
from pathlib import Path
# print(sys.path)
sys.path.append(os.path.dirname(__file__))
# print(sys.path)

import argparse
import glob
import tqdm
import time
import tempfile
# Some basic setup:

# import some common libraries
import numpy as np
import os, json, cv2, random


import detector_detectron2
import detected_object
import homography
import team_detector
import tracker
from enum import Enum, auto

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 football detector.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        help="Path to image or directory with images. Directory can only contain images!",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument("--export-data-path", help="A file to export data (frame_id, bounding boxes, score).")
    parser.add_argument("--no-gui", help="Disable gui mean will not disaplay opencv window good option on server if we want only export data file without process output video or image/s")

    return parser

class TV2D:
    
    class DetectorType(Enum):
        DETECTRON = auto()
    class RunOn(Enum):
        VIDEO = auto()
        IMAGE = auto()

    PERSON_OBJECT_CLASS = 0
    BALL_OBJECT_CLASS = 32
    WINDOW_NAME = "TV2D"

    def __init__(self, object_detection_model_path, object_detection_config_path, detector_type = DetectorType.DETECTRON, object_detection_threshold = 0.3, 
                 homography_on = True, 
                 homography_keypoint_path = "models/FPN_efficientnetb3_0.0001_4.h5",
                 homography_deephomo_path = "models/HomographyModel_0.0001_4.h5",
                 homography_threshold = 0.9, homography_pretreined = True, 
                 team_detection_on = True, 
                 deep_sort_model_path = "models/market_bot_R50.pth",
                 deep_sort_model_config = "deep_sort_pytorch/thirdparty/fast-reid/configs/Market1501/bagtricks_R50.yml", tracker_on = True, no_gui = False,
                 pitch_2D_width = 320, pitch_2D_height = 180, display_pitch_2D = True): 
        
        # No gui mean will not disaplay opencv window good option on server if we want only export data file without process output video or image/s.
        self.no_gui =  no_gui
        
        self.pitch_2D_width = pitch_2D_width
        self.pitch_2D_height = pitch_2D_height
        self.display_pitch_2D = display_pitch_2D
        self.export_file_path = None
        
        if detector_type == TV2D.DetectorType.DETECTRON: 
            self.objectDetector = detector_detectron2.DetectronDetector(model_threshold=object_detection_threshold, model_path=object_detection_model_path, 
                                                                        model_config_path=object_detection_config_path)
        
        self.homography = None
        if homography_on:
            if not homography_pretreined:
                assert homography_keypoint_path, "Homography keypoint path not set!"
                assert homography_deephomo_path, "Homography deephomo path not set!"
            # assert os.path.isfile(homography_keypoint_path), "Homography keypoint file not exists!"
            # assert os.path.isfile(homography_deephomo_path), "Homography deephomo file not exists!"
            self.homography = homography.Homography(weights_keypoint_path=(homography_keypoint_path), weights_deephomo_path=(homography_deephomo_path), pretreined=homography_pretreined, threshold=homography_threshold)
          
        self.teamDetector = None  
        if team_detection_on:
            self.teamDetector = team_detector.TeamDetector()
        
        self.objectTracker = None
        if tracker_on:    
            self.objectTracker = tracker.Tracker(model_path=deep_sort_model_path, model_config=deep_sort_model_config)
            
    def __call__(self, run_on, file_to_process_path, export_output_path = "", object_detection_threshold = 0.3, homography_threshold = 0.9, export_data_file_path = ""):
        """_summary_

        Args:
            run_on (RunOn): RunOn.VIDEO for video or RunOn.IMAGE for image
            path (String): path to video or (image or images directory) depends which function choosed
            export_output_path (str, optional): Path to file where output (proccessed video or image) will be saved. If passed directory then saves in directory with source file name or names for images directory source. Defaults to "" mean no export.
            object_detection_threshold (float, optional): _description_. Defaults to 0.3.
            homography_threshold (float, optional): _description_. Defaults to 0.9.
            export_data_file_path (str, optional): Path to. Defaults to "".
        """
        assert os.path.exists(file_to_process_path), f"{file_to_process_path} not exists!"
        if export_output_path:
            if os.path.isdir(export_output_path):
                assert os.path.exists(export_output_path), f"Export {export_output_path} not exists!"
        
        if self.homography is not None:
            self.homography.threshold = homography_threshold
            
        if export_data_file_path:
            self.export_file_path = export_data_file_path
            os.makedirs(os.path.dirname(export_data_file_path), exist_ok=True)
            export_file = open(export_data_file_path, 'w')
        else:
            export_file = None
                    
        # -------------------------- Video preparing and processing ----------------------- #
        if run_on == TV2D.RunOn.VIDEO:
            assert not os.path.isdir(file_to_process_path), "For video process acceptable is only video file not directory!"
            video = cv2.VideoCapture(file_to_process_path)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(file_to_process_path)
            codec, file_ext = (
                ("x264", ".mkv") if TV2D.test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                print("x264 codec not available, switching to mp4v")
            if export_output_path:
                if os.path.isdir(export_output_path):
                    output_fname = os.path.join(export_output_path, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = export_output_path
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(file_to_process_path)
            for vis_frame in tqdm.tqdm(self.run_on_video(video, objectDetector=self.objectDetector, export_file=export_file, 
                              homography=self.homography, teamDetector=self.teamDetector, objectTracker=self.objectTracker), total=num_frames):
                if export_output_path:
                    output_file.write(vis_frame)
                elif not self.no_gui:
                    cv2.namedWindow(f"{TV2D.WINDOW_NAME} {basename}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"{TV2D.WINDOW_NAME} {basename}", vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            if export_output_path:
                output_file.release()
            else:
                cv2.destroyAllWindows()
        # -------------------------- End of video preparing and processing ----------------------- #
        
        # --------------------------   Image or images processing -------------------------------- #
        if run_on == TV2D.RunOn.IMAGE:
            images = []
            if os.path.isdir(file_to_process_path):
                if file_to_process_path[-1] != "/":
                    file_to_process_path += "/"
                for image_path in glob.glob(f"{file_to_process_path}*.*"):
                    images.append(image_path)
            else:
                images.append(file_to_process_path)
                
            for index, path in enumerate(tqdm.tqdm(images, total=len(images))):
                image = cv2.imread(path)
                image = self.run_on_image(image, os.path.basename(path), objectDetector=self.objectDetector, export_file=export_file, 
                              homography=self.homography, teamDetector=self.teamDetector)
            
                if export_output_path:
                    if os.path.isdir(export_output_path):
                        assert os.path.isdir(export_output_path), export_output_path
                        out_filename = os.path.join(export_output_path, os.path.basename(path))
                    else:
                        out_filename = export_output_path
                    cv2.imwrite(out_filename, image)
                elif not self.no_gui:
                    cv2.namedWindow(TV2D.WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(TV2D.WINDOW_NAME, image)
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
        # --------------------------  End image or images processing -------------------------------- #
        
        if export_file is not None and not export_file.closed:
            export_file.close()
        

    @staticmethod
    def export_parser(detecedObject, frame_id = None, file_name = None, all_data = True, file_format = "csv"):
        object_class = detecedObject.object_class
        score = detecedObject.score
        bbox = detecedObject.bbox
        seperator = "\t"
        if file_format.lower() == "csv":
            seperator = ";"
            
        first_row = ""
        if frame_id:
            first_row = frame_id
        elif file_name:
            first_row = file_name
            
        if object_class == TV2D.BALL_OBJECT_CLASS:
            object_class = 1
        if all_data:
            # frame_id  object_class    score   bb_cx   bb_cy   bb_height   bb_width    trackId colorRED    colorGREEN  colorBlue   pitchx  pitchy
            trackId = '' if detecedObject.trackId is None else str(detecedObject.trackId)
            colorR = '' 
            colorG = '' 
            colorB = ''                 
            if detecedObject.color is not None:
                colorR = detecedObject.color[0]
                colorG = detecedObject.color[1]
                colorB = detecedObject.color[2]
            pitchx = ''
            pitchy = ''
            if detecedObject.pitchxy is not None:
                pitchx = detecedObject.pitchxy[0]
                pitchy = detecedObject.pitchxy[1]
            return f"{first_row} {object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]} {trackId} {colorR} {colorG} {colorB} {pitchx} {pitchy}\n".replace(" ", seperator)
        else:
            # frame_id	object_class	score   bb_cx	bb_cy	bb_height	bb_width
            return f"{first_row} {object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]}\n".replace(" ", seperator)

    @staticmethod
    def test_opencv_video_format(codec, file_ext):
        with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
            filename = os.path.join(dir, "test_file" + file_ext)
            writer = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(30),
                frameSize=(10, 10),
                isColor=True,
            )
            [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
            writer.release()
            if os.path.isfile(filename):
                return True
            return False

    def export_to_file(self, deteced_objects, export_file, frame_id = None, all_data = False):
        file_format = os.path.basename(self.export_file_path).split(".")[-1].lower()
        for index, detecedObject in enumerate(deteced_objects):
            if not detecedObject.isout:
                if frame_id:      
                    export_file.write(TV2D.export_parser(detecedObject, frame_id=frame_id, all_data=all_data, file_format=file_format))
                else:
                    export_file.write(TV2D.export_parser(detecedObject, all_data=all_data, file_format=file_format))

    def __frame_from_video(self, video):
        frame_id = 0
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame_id += 1
                yield frame, frame_id
            else:
                break
            
    @staticmethod
    def draw_bboxes_on_image(deteced_objects, image):                
        for index, detecedObject in enumerate(deteced_objects):                     
            color = (255,0,0)
            if detecedObject.object_class == TV2D.BALL_OBJECT_CLASS:
                color = (0,0,255)
            # if detecedObject.isout:
            #     color = (123,123,255)
            box = detecedObject.bbox
            if detecedObject.color is not None:
                color = (int(detecedObject.color[0]), int(detecedObject.color[1]), int(detecedObject.color[2]))
            if not detecedObject.isout:                
                cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), color, 2) 
                # cv2.circle(image, (detecedObject.get_xcycwh()[0], detecedObject.get_xcycwh()[1]), radius=1, color=(0,0,255), thickness=-1)
                # cv2.circle(image, (box[0]+detecedObject.get_xcycwh()[2], box[1]+detecedObject.get_xcycwh()[3]), radius=1, color=(0,0,255), thickness=-1)
            if detecedObject.trackId is not None and not detecedObject.isout:
                fontScale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 0)
                thickness = 2
                cv2.putText(image, f"{detecedObject.trackId}", (box[0],box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
            
        return image
    
    @staticmethod
    def merge_frame_with_pitch(frame, pitch):
        x_offset = 0
        y_offset = frame.shape[0]-pitch.shape[0]
        x_end = x_offset + pitch.shape[1]
        y_end = y_offset + pitch.shape[0]
        frame[y_offset:y_end,x_offset:x_end] = pitch
        return frame
            
    def run_engine(self, frame, frame_id, detected_objects, objectDetector, export_file = None, homographyDetector = None, teamDetector = None, objectTracker = None):
        """_summary_

        Args:
            frame (Mat): cv2.imread() or any array with image with compatibile format like cv2
            frame_id (str): frame_id or file name
            detected_objects (_type_): _description_
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.
            teamDetector (TeamDetector, optional): _description_. Defaults to None.
            objectTracker (Tracker, optional): _description_. Defaults to None.

        Returns:
            Mat: cv2.imread() array
        """
        
        if homographyDetector is not None:
            predicted_homography = homographyDetector.predict_homography(frame)
            
            homographyDetector.check_object_isout(frame.shape, detected_objects, predicted_homography)
        
        if objectTracker is not None:
            objectTracker.update_all(detected_objects, frame)
        
        if teamDetector is not None:        
            teamDetector.detectMainColors(frame, detected_objects, TV2D.PERSON_OBJECT_CLASS)                
            for detectedObject in detected_objects:
                if(detectedObject.object_class == TV2D.PERSON_OBJECT_CLASS and not detectedObject.isout):
                    detectedObject.color = teamDetector.assignTeam(frame, detectedObject)
            
        if homographyDetector is not None:
            pitch = homographyDetector.generate_pitch(frame.shape, detected_objects, predicted_homography, 
                                                      pitch_2D_width=self.pitch_2D_width, pitch_2D_height=self.pitch_2D_height)
            
        frame = TV2D.draw_bboxes_on_image(detected_objects, frame)
        
        if homographyDetector is not None and self.display_pitch_2D:
            frame = TV2D.merge_frame_with_pitch(frame, pitch)
            
        if export_file != None and not export_file.closed:
            self.export_to_file(detected_objects, export_file, frame_id, all_data = True)
        
        return frame

    def run_on_video(self, video, objectDetector, export_file = None, homography = None, teamDetector = None, objectTracker = None):
        """_summary_

        Args:
            video (VideoCapture): cv2.VideoCapture() object only 
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.
            teamDetector (TeamDetector, optional): _description_. Defaults to None.
            objectTracker (Tracker, optional): _description_. Defaults to None.

        Yields:
            Mat: cv2.imread() array
        """
        for frame, frame_id in self.__frame_from_video(video):        
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detected_objects = objectDetector.detect_objects(frame)
            frame = self.run_engine(frame, str(frame_id), detected_objects, objectDetector, export_file=export_file, homographyDetector=homography, teamDetector=teamDetector, objectTracker=objectTracker)            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame

    def run_on_image(self, image, file_name, objectDetector, export_file = None, homography = None, teamDetector = None):
        """_summary_

        Args:
            image (Mat): cv2.imread() or any array with image with compatibile format like cv2
            objectDetector (Detector): _description_
            export_file (File, optional): file object. Defaults to None.
            homography (Homography, optional): _description_. Defaults to None.
            teamDetector (TeamDetector, optional): _description_. Defaults to None.
            objectTracker (Tracker, optional): _description_. Defaults to None.
            
        Returns:
            Mat: cv2.imread() array
        """
        start_time = time.time()
        
        detected_objects = objectDetector.detect_objects(image)
        print("{}: {} in {:.2f}s".format(
            file_name,
            "detected {} instances".format(len(detected_objects))
            if len(detected_objects)
            else "finished",
            time.time() - start_time,
            ))

        image = self.run_engine(image, file_name, detected_objects, objectDetector, export_file=export_file, homographyDetector=homography, teamDetector=teamDetector)            

        return image
        

if __name__ == "__main__":
    args = get_parser().parse_args()  
    # Set path to the model or Detectron2 Model Zoo and Baselines models file convention name like as default. 
    ## The model with the Detectron2 convention will be downloaded automatically
    object_detection_model_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    # object_detection_model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # Same as model by for now for config. If you use model from local sotrage you can still use config 
    ## from Detectron2 Model Zoo and Baselines.
    object_detection_config_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    homography_keypoint_path = "models/FPN_efficientnetb3_0.0001_4.h5"
    homography_deephomo_path = "models/HomographyModel_0.0001_4.h5"
    deep_sort_model = "models/market_bot_R50.pth"
    deep_sort_model_config = "deep_sort_pytorch/thirdparty/fast-reid/configs/Market1501/bagtricks_R50.yml"
    
    export_output_path = None
    export_data_path = None
    no_gui = False
    if args.output:
        export_output_path = args.output
    
    if args.export_data_path:
        export_data_path = args.export_data_path
        
    if args.no_gui:
        no_gui = True
        
    if args.input:
        tv2d = TV2D(object_detection_model_path, object_detection_config_path=object_detection_config_path, no_gui=no_gui, homography_on=True, team_detection_on=True, 
                    tracker_on=True, homography_pretreined=False, 
                    homography_deephomo_path=homography_deephomo_path, homography_keypoint_path=homography_keypoint_path, deep_sort_model_path=deep_sort_model, 
                    deep_sort_model_config=deep_sort_model_config)
        tv2d(TV2D.RunOn.IMAGE, args.input, export_output_path=export_output_path, export_data_file_path=export_data_path)
    elif args.video_input:
        tv2d = TV2D(object_detection_model_path, object_detection_config_path=object_detection_config_path, homography_on=True, team_detection_on=True, 
                    tracker_on=True, no_gui=no_gui, 
                    homography_pretreined=False, homography_deephomo_path=homography_deephomo_path, homography_keypoint_path=homography_keypoint_path, 
                    deep_sort_model_path=deep_sort_model, deep_sort_model_config=deep_sort_model_config)
        tv2d(TV2D.RunOn.VIDEO, args.video_input, export_output_path=export_output_path, export_data_file_path=export_data_path)
