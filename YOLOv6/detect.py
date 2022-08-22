
import os
import sys
from pathlib import Path
import torch
import math
from PIL import ImageFont
import requests
import cv2
import numpy as np
import time
import logging
import pafy
from urllib.parse import urlparse,parse_qs
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv6 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional
from include.decode_utils.utils import encodeImageIntoBase64,check_file
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

class Detector():

    def __init__(self, filename,fileformat):
        # https://github.com/meituan/YOLOv6/blob/main/tools/infer.py
        self.weights="./include/predictor_yolo_detector/best_ckpt.pt"  # model.pt path(s)
        if fileformat:
            self.source = fileformat
        else:
            self.source="./include/predictor_yolo_detector/inference/images/"+filename  # file/dir/URL/glob, 0 for webcam
        self.yaml='./data/data.yaml'  # dataset.yaml path
        # self.data=None
        self.img_size=int(640)# inference size (height, width)
        self.conf_thres=float(0.25)  # confidence threshold
        self.iou_thres=float(0.45)  # NMS IOU threshold
        self.max_det=int(1000)  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_txt=False  # save results to *.txt
        self.save_img=False  # save visuallized inference results.
        self.save_dir=False  # directory to save predictions in. See --save-txt
        self.view_img=False  # show inference results
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.project="./include/predictor_yolo_detector/inference/output" # save results to project/name
        self.name='output_image.jpg'  # save results to project/name
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference

    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
     
        image = letterbox(img_src, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src
        
    def check_img_size(img_size, s=32, floor=0):
       
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(Detector.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(Detector.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def infer(img, img_src,device,model,conf_thres,iou_thres,classes,agnostic_nms,max_det,yaml):
        ''' Model Inference and results visualization '''

            
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
                # expand for batch dim
        pred_results = model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src

        # check image and font
        assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
        Detector.font_check()

        if len(det):
            det[:, :4] =Detector.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
       
        return det  
    
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), fps= None):

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        # Add one xyxy box to image with label
        
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
          
            cv2.putText(image, "fps:" + fps, (30,30), 0, lw / 3, (0,50,255),
                        thickness=tf, lineType=cv2.LINE_AA)

    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert os.path.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color


    def run(self):
        weights= self.weights
        source=self.source
        yaml=self.yaml
        img_size=self.img_size
        conf_thres=self.conf_thres
        iou_thres=self.iou_thres
        max_det=self.max_det
        device=self.device
        save_txt= self.save_txt
        save_img=self.save_img
        save_dir= self.save_dir
        view_img= self.view_img
        classes=self.classes
        agnostic_nms=self.agnostic_nms
        project=self.project
        name=self.name
        line_thickness=self.line_thickness
        hide_labels=self.hide_labels
        hide_conf=self.hide_conf
        half=self.half
        source = str(source)
          #Set-up hardware options
        cuda = device != 'cpu' and torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')
        model = DetectBackend(weights, device=device)
        stride = int(model.stride)
       
        img_size = Detector.check_img_size(img_size, s=stride)
        class_names = load_yaml(yaml)['names']
         # Half precision

        if half & (device.type != 'cpu'):
            model.model.half()
            
        else:
            model.model.float()
            half = False
           

        if device.type != 'cpu':
           model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters()))) 
       
    
        save_img = not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download
        # Directories
        save_dir=Path(f"{project}")
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

      
        # Dataloader
        if is_file:
            try:
                img_src = cv2.imread(source)
                assert img_src is not None, f'Invalid image'
            except Exception as e:
                print("Invalid Image Path or the image is empty cannot run inference")  
            start = time.time()
            img, img_src = Detector.precess_image(img_src,img_size, model.stride, half)
            img_detect = Detector.infer(img, img_src,device,model,conf_thres,iou_thres,classes,agnostic_nms,max_det,yaml)
            end = time.time() - start
    
            fps_txt =  "{:.2f}".format(1/end)
            
            for *xyxy, conf, cls in reversed(img_detect):
                    class_num = int(cls)  # integer class
                    label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                    Detector.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=Detector.generate_colors(class_num, True), fps = fps_txt)
                    image = np.asarray(img_src) 
           
            if save_img:
                 save_path = str(save_dir / name) 
                 cv2.imwrite(save_path,image)

           
        if is_url:
            
            #Getting video id from the url string
            print("URL Link : ",source)
            print( "Input Video saved :",save_dir)
            url_data = urlparse(source)
            query = parse_qs(url_data.query)
            id = query["v"][0]
            video = 'https://youtu.be/{}'.format(str(id))
       
          
           #Using the pafy library for youtube videos
            urlPafy = pafy.new(video)
            video_path = urlPafy.getbest(preftype="mp4")
            # video_path =r'C:\\YOLOV6_02072022\Trafic.mp4'
           

            video = cv2.VideoCapture(video_path)
            ret, img_src = video.read()

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            output = cv2.VideoWriter(os.path.join(save_dir,'output.mp4'), fourcc,30 , (img_src.shape[1],img_src.shape[0]))
            print("File saved into local computer")
            while True:
                
                if ret:
                   
                    start = time.time()
                    img, img_src = Detector.precess_image(img_src,img_size,model.stride, half)
                    det = Detector.infer(img, img_src,device,model,conf_thres,iou_thres,classes,agnostic_nms,max_det,yaml)
                    end = time.time() - start
                    fps_txt =  "{:.2f}".format(1/end)
                    
                    for *xyxy, conf, cls in reversed(det):

                        class_num = int(cls)  # integer class
                        label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                        Detector.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=Detector.generate_colors(class_num, True), fps = fps_txt)

                    image = np.asarray(img_src)
                    output.write(image)
                    ret, img_src = video.read()
                    print(" Still Processing,Please wait :",datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'))
                
                else:
                    break

            output.release()
            video.release()
            #Save video
            name = output.mp4
            save_path = str(save_dir / name) 

    def detect_action(self):
        with torch.no_grad():
            self.run()
        bgr_image = cv2.imread("./include/predictor_yolo_detector/inference/output/output_image.jpg")
        cv2.imwrite('predicted_output_image.jpg', bgr_image)
        opencodedbase64 = encodeImageIntoBase64('./include/predictor_yolo_detector/inference/output/output_image.jpg')
        result = {"image": opencodedbase64.decode('utf-8')}
        return result


