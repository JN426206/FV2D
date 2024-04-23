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
from accuracer import Accuracer

PERSON_OBJECT_CLASS = 0
BALL_OBJECT_CLASS = 32

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 football detector.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument("--export-file", help="A file to export data (frame_id, bounding boxes, score).")


    parser.add_argument("--benchmark", action='store_true', help="A file to export data (frame_id, bounding boxes, score).")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def export_parser(detecedObject, frame_id = None, file_name = None, all_data = False):
    # frame_id	object_class	bb_cx	bb_cy	bb_height	bb_width	score
    object_class = detecedObject.object_class
    score = detecedObject.score
    bbox = detecedObject.bbox
    if frame_id:
        if object_class == BALL_OBJECT_CLASS:
            object_class = 1
        if all_data:
            return f"{frame_id} {object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]} {detecedObject.trackId} {detecedObject.color} {detecedObject.pitchxy}\n"
        else:
            return f"{frame_id} {object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]}\n"
        # return f"{frame_id}\t{object_class}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]-bbox[0]}\t{bbox[3]-bbox[1]}\t{score}\n"
    elif file_name:
        return f"{file_name}\t{object_class}\t{bbox[0]}\t{bbox[1]}\t{bbox[2]-bbox[0]}\t{bbox[3]-bbox[1]}\t{score}\n"
    else:        
        if object_class == BALL_OBJECT_CLASS:
            object_class = 1
        return f"{object_class} {score:0.2} {bbox[0]} {bbox[1]} {bbox[2]-bbox[0]} {bbox[3]-bbox[1]}\n"

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

def _frame_from_video(video):
    frame_id = 0
    while video.isOpened():
        success, frame = video.read()
        if success:
            frame_id += 1
            yield frame, frame_id
        else:
            break
        
def draw_bboxes_on_image(deteced_objects, image):                
    for index, detecedObject in enumerate(deteced_objects):                     
        color = (255,0,0)
        if detecedObject.object_class == BALL_OBJECT_CLASS:
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

def run_on_video(video, objectDetector, export_file = None, teamDetector = None, objectTracker = None):
    for frame, frame_id in _frame_from_video(video):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        
        detected_objects = objectDetector.detect_objects(frame)
        
        if homographyDetector is not None:
            predicted_homography = homographyDetector.predict_homography(frame)
            
            homographyDetector.check_object_isout(frame.shape, detected_objects, predicted_homography)
        
        if objectTracker is not None:
            objectTracker.update_all(detected_objects, frame)
        
        if teamDetector is not None:        
            teamDetector.detectMainColors(frame, detected_objects, PERSON_OBJECT_CLASS)                
            for detectedObject in detected_objects:
                if(detectedObject.object_class == PERSON_OBJECT_CLASS and not detectedObject.isout):
                    detectedObject.color = teamDetector.assignTeam(frame, detectedObject)
            
        if homographyDetector is not None:
            pitch = homographyDetector.generate_pitch(frame.shape, detected_objects, predicted_homography)
        
        # print(out_boxes)
            
        frame = draw_bboxes_on_image(detected_objects, frame)
        
        if homographyDetector is not None:
            frame = merge_frame_with_pitch(frame, pitch)
            
        if export_file != None and not export_file.closed:
            export_to_file(detected_objects, export_file, frame_id, all_data = True)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        yield frame

def export_to_file(deteced_objects, export_file, frame_id = None, all_data = False):
    for index, detecedObject in enumerate(deteced_objects):
        if not detecedObject.isout:
            if frame_id:      
                export_file.write(export_parser(detecedObject, frame_id=frame_id, all_data=all_data))
            else:
                export_file.write(export_parser(detecedObject, all_data=all_data))

def export_to_file_close(deteced_objects, export_file):
    export_to_file(deteced_objects, export_file)
    export_file.close()
    
def merge_frame_with_pitch(frame, pitch):
    x_offset = 0
    y_offset = frame.shape[0]-pitch.shape[0]
    x_end = x_offset + pitch.shape[1]
    y_end = y_offset + pitch.shape[0]
    frame[y_offset:y_end,x_offset:x_end] = pitch
    return frame

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

    objectDetector = detector_detectron2.DetectronDetector(model_threshold=0.3, model_path=object_detection_model_path, 
                                                           model_config_path=object_detection_config_path)
    homographyDetector = homography.Homography(threshold=0.9, pretreined=False, 
                    weights_deephomo_path=homography_deephomo_path, weights_keypoint_path=homography_keypoint_path)
    # homographyDetector = None
    teamDetector = team_detector.TeamDetector()
    # teamDetector = None
    objectTracker = tracker.Tracker(model_path=deep_sort_model, 
                                       model_config=deep_sort_model_config)
    # objectTracker = None

    export_file = None

    if args.export_file:
        export_file = open(args.export_file, "w")

    if args.benchmark:
        accuracer = Accuracer()
        for file_id in accuracer.get_files_id():
            export_file = open(f"data/jsladjf1_pred_good_mask_labels/{file_id}.txt", "w")
            image = accuracer.open_image_file(file_id)
            start_time = time.time()

            deteced_objects = objectDetector.detect_objects(image)

            if homographyDetector is not None:
                predicted_homography = homographyDetector.predict_homography(image)
            
                homographyDetector.check_object_isout(image.shape, deteced_objects, predicted_homography)
            
            print("{}: {} in {:.2f}s".format(
                file_id,
                "detected {} instances".format(len(deteced_objects))
                if len(deteced_objects)
                else "finished",
                time.time() - start_time,
                ))
                        
            # accuracer.create_labeled_image_from_prediction(deteced_objects, file_id, "labeled_pred_mask")
            export_to_file_close(deteced_objects, export_file)
            # accuracer.check_accuracy(bboxes, pred_classes, file_id)
        
            
    elif args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for index, path in enumerate(tqdm.tqdm(args.input, disable=not args.output)):
            # use PIL, to be consistent with evaluation
            image = cv2.imread(path)
            start_time = time.time()

            detected_objects = objectDetector.detect_objects(image)            
            
            print("{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(detected_objects))
                if len(detected_objects)
                else "finished",
                time.time() - start_time,
                ))
            
            if homographyDetector is not None:
                predicted_homography = homographyDetector.predict_homography(image)
                print(predicted_homography)
                homographyDetector.check_object_isout(image.shape, detected_objects, predicted_homography)
            
            if teamDetector is not None:  
                teamDetector.detectMainColors(image, detected_objects, PERSON_OBJECT_CLASS)
            
                for detectedObject in detected_objects:
                    if(detectedObject.object_class == PERSON_OBJECT_CLASS):
                        detectedObject.color = teamDetector.assignTeam(image, detectedObject)

            if homographyDetector is not None:
                pitch = homographyDetector.generate_pitch(image.shape, detected_objects, predicted_homography)

            image = draw_bboxes_on_image(detected_objects, image)


            if homographyDetector is not None:
                image = merge_frame_with_pitch(image, pitch)
            
            if export_file != None and not export_file.closed:
                export_to_file_close(detected_objects, export_file)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                cv2.imwrite(out_filename, image)
            else:
                cv2.namedWindow("Detectron2 football detector.", cv2.WINDOW_NORMAL)
                cv2.imshow("Detectron2 football detector.", image)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            print("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
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
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(run_on_video(video, objectDetector, export_file, teamDetector=teamDetector, objectTracker=objectTracker), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    if args.export_file and not export_file.closed:
        export_file.close()
