# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import time
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

# Add ROOT to sys.path if it's not already included
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Adjust ROOT to be relative to the current working directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


class SonarSubscriber(Node):
    def __init__(self):
        super().__init__('sonar_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ping_topic',
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        global ping_x,ping_y,conf_x,conf_y
        if len(msg.data) >= 2:  # Ensure we received valid data
            ping_x = msg.data[0]
            conf_x = msg.data[1]
            ping_y = msg.data[2]
            conf_y = msg.data[3]
        else:
            self.get_logger().warn('Received malformed data!')

class YoloPub(Node):
    def __init__(self):
        super().__init__("Tub_Front_Pub_Node")

        self.declare_parameter("topic", value="command")
        topic_name = self.get_parameter("topic").get_parameter_value().string_value
        self.publisher = self.create_publisher(Int32, topic_name, 10)
        self.shutdown_flag = False
    def publish_detection(self, detection_msg):
        msg = Int32()
        msg.data = detection_msg
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
    def kill_node(self):  
        self.get_logger().info("Killing Node")
        rclpy.shutdown()

def init_ros2():
    rclpy.init()
    return YoloPub()

directon_flag=None
@smart_inference_mode()
def run(
    # weights=ROOT / "/home/agastya/sauvc25/src/testing/testing/sauvc_tub.pt",  # model path or triton URL
    weights=ROOT / "/home/agastya/sauvc25/src/testing/testing/Gulmohar_new_front.pt",  # model path or triton URL
    source="2",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.05,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="/home/agastya/sauvc25/src/auv/auv",  # save results to project/name
    name="Front_dataset",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    global ping_x,ping_y,conf_x,conf_y,flag_dir_x,lock_x,flag_dir
    flag_dir = 0
    flag_sur=0
    flag_dir_x= 0
    ping_x=0
    lock_x = 0
    ping_y=0
    conf_x=0
    conf_y=0
    flag_f=0
    flag_l=0
    flag_r=0
    FORWARD = 1
    BACKWARD = 4
    LEFT = 2
    RIGHT = 3
    STOP = 5
    KILL = 123
    DEPTH_DOWN = 6
    SPEED_DOWN = 93
    DROP = 76
    
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    frames_dir=save_dir/'Blue_front'
    frames_dir.mkdir(parents=True,exist_ok=True)
    cont_dir=save_dir/'Blue_front_cont'
    cont_dir.mkdir(parents=True,exist_ok=True)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    command_pub = init_ros2()
    sonar_sub = SonarSubscriber()

    command_pub.publish_detection(DEPTH_DOWN)
    time.sleep(0.5)
    
    # command_pub.publish_detection(DEPTH_DOWN)
    # time.sleep(0.5)

    # command_pub.publish_detection(DEPTH_DOWN)
    # time.sleep(2)
    
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)
        # Process predictions
        for i, det in enumerate(pred):
            rclpy.spin_once(sonar_sub, timeout_sec=0.1)
              # per image
        # Check if there are any detections
              # Example command, replace with appropriate terminal command  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string

            # if ping_y>12 and conf_y>90:
            #     corr_lockin()

            # def corr_lockin():
            #     return
            def x_corr_locking(lock_x):
                global ping_x,ping_y,conf_x,conf_y
                global flag_dir_x
                error_x = (ping_x - lock_x)
                print("ping_x",ping_x)
                print("lock_x",lock_x)
                print("conf_x",conf_x)
                print("error_x",error_x)
                if 30 > error_x > 0.2 and conf_x > 90: 
                    command_pub.publish_detection(LEFT)
                    print(LEFT)
                    flag_dir_x = 1
                elif -30 < error_x < -0.2 and conf_x > 90:
                    command_pub.publish_detection(RIGHT)
                    print(RIGHT)
                    flag_dir_x = 2
                else:
                    if flag_dir_x == 1:
                        command_pub.publish_detection(RIGHT)
                        print(RIGHT)
                        flag_dir_x = 0 
                    elif flag_dir_x == 2:
                        command_pub.publish_detection(LEFT)
                        print(LEFT)
                        flag_dir_x = 0
                    else:
                        print(FORWARD)
                        command_pub.publish_detection(FORWARD)
                        return True
            if save_img:
                frame_save_path=frames_dir/f"Blue_{seen:04d}.jpg"
                cv2.imwrite(str(frame_save_path),im0)
            def align_x_coordinate(center, rectangle):
                global flag_dir
                x_min, x_max = rectangle[0], rectangle[0] + rectangle[2]

                # Extract the x-coordinate from the single center tuple
                c_x = center[0]

                
                if x_min < c_x < x_max and flag_dir == 0:
                    return "Forward"
                elif c_x <= x_min:
                    flag_dir = 1
                    return "Left"
                elif c_x >= x_max:
                    flag_dir = 2
                    return "Right"
                elif flag_dir == 2:
                    flag_dir = 0
                    return "Right"
                elif flag_dir == 1:
                    flag_dir = 0
                    return "Left"
                        
            fixed_point = (im0.shape[1] // 2, im0.shape[0] // 2)
            square_size = 150
            rectangle = (fixed_point[0] - square_size // 2, fixed_point[1] - square_size // 2, square_size, square_size)

            cv2.rectangle(im0, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 0, 0), 2)


            

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
         
            if 21>ping_y>19 and conf_y>85:
                x_corr_locking(lock_x)
            elif 25>ping_y>=21 and conf_y>85:
                print("Switch To Bottom Cam")
                command_pub.publish_detection(STOP)
                command_pub.destroy_node()  #Kill the flare code......
                rclpy.shutdown()
            
            elif len(det):
                flag_sur = 4
                lock_x = ping_x
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    center_y = int((xyxy[1] + xyxy[3]) / 2)
                    center=(center_x,center_y)
                    
                    # printed_once = False
                    if center:
                        x_alignment_result = align_x_coordinate(center, rectangle)
                        if x_alignment_result  == "Left" and flag_l==0:
                            print("Go Left")
                            flag_l=1
                            flag_f=0
                            flag_r=0
                            printed_once=True
                            command_pub.publish_detection(LEFT)
                        if x_alignment_result  == "Right" and flag_r==0:
                            print("Go Right")
                            flag_r=1
                            flag_l=0
                            flag_f=0
                            printed_once=True
                            command_pub.publish_detection(RIGHT)
                        if x_alignment_result  == "Forward" and flag_f==0:
                            if flag_f==0:
                                print("Forward")
                                printed_once=True
                                flag_f=1
                                flag_l=0
                                flag_r=0
                                command_pub.publish_detection(FORWARD)

            
                
                    # Draw a circle at the center of the detected object
                    cv2.circle(im0, (center_x, center_y), radius=30, color=(0, 0, 0), thickness=2)  # Adjust radius and color as needed
                    
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
            
            if len(det)==0 and flag_sur==0 :
                    print("lateral left")
                    command_pub.publish_detection(LEFT)
                    flag_sur=1
                    
            if len(det) == 0 and flag_sur==1 :
                if ping_x<28 and ping_x<50 and conf_x>85:
                    print("Lateral Right")
                    command_pub.publish_detection(RIGHT)
                    flag_sur=2
                
            if len(det) == 0 and flag_sur==2:
                if 50>ping_x>47 and conf_x>85:
                    print('Surface')
                    command_pub.publish_detection(KILL) 
                    flag_sur=3
            # else:
            #     if len(det)==0 and flag_sur==0:
            #         print("lateral Right")
            #         flag_sur=1
            #     elif ping_x<3000 and conf_x>85 and flag_sur==1:
            #         print("Lateral Left")
            #         flag_sur=2
            #     elif ping_x>27000 and conf_x>85 and flag_sur==2:
            #         print('Surface')
                

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
            if save_img:
                frame_save_path_contour=cont_dir/f"Flare_{seen:04d}.jpg"
                cv2.imwrite(str(frame_save_path_contour),im0)

       

def main():
    run()

if __name__ == '__main__':
    main()
