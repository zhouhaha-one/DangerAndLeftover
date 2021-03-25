#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys, os

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import base64
import cv2

from configRetrive import *
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "DangerAndLeftover.settings")
django.setup()

last_pic = []
cur_pic = []


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


# 计算两个bbox的iou
def compare_iou(x1, y1, x2, y2, x1_hat, y1_hat, x2_hat, y2_hat, iou_threshold=0.50):
    # Intersection area
    inter = (min(x2, x2_hat) - max(x1, x1_hat)) * (min(y2, y2_hat) - max(y1, y1_hat))

    eps = 1e-9
    # Union Area
    w1, h1 = x2 - x1, y2 - y1 + eps
    w2, h2 = x2_hat - x1_hat, y2_hat - y1_hat + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou > iou_threshold


def detect(save_img=False):
    global cur_pic
    global last_pic

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        print("load from webcam")
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        print("load from picture")
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        # 对每一张图片做处理：
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 保存图片的副本，留作截取目标bbox用
            im0_copy = im0.copy()
            # 每次循环将cur_pic中的内容给last_pic，并将cur_pic置空
            if cur_pic:
                last_pic = cur_pic
            cur_pic = []
            # 定义一个list存储一张图片的所有类别，并一起上传到前端
            abnormal_list = []
            abnormal_list2 = []
            # 定义一个比较数量，将当前图片与last_pic中每一张比较，最后退出检测compare_num若等于last_pic张数，表明是新目标；否则是重复目标
            compare_num = 0
            new_object_flag = False
            # 若图片上检测到的目标不为空
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                # 保存预测结果，此处是针对一张图上的每个检测框来说的
                for *xyxy, conf, cls in reversed(det):
                    compare_num = 0
                    # 定义一个cur_pic1，字典类型，保留此张图片一个bbox的信息
                    cur_pic1 = dict()
                    cur_pic1['cls'] = torch.tensor(cls).view(-1).tolist()[0]
                    cur_pic1['xyxy'] = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    cur_pic1['conf'] = torch.tensor(conf).view(-1).tolist()[0]
                    cur_pic.append(cur_pic1)
                    # 当前帧一个bbox的坐标信息
                    x1, y1, x2, y2 = cur_pic1['xyxy'][0], cur_pic1['xyxy'][1], \
                                     cur_pic1['xyxy'][2], cur_pic1['xyxy'][3]
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=3)
                    # 去重方法：与上一帧相比，判断是否有新类别出现，若有，发送新的一帧图的目标检测结果；若无，不发送
                    # 具体步骤：
                    # 1.判断上一帧是否为空：
                    #   1.1.若为空,发送当前帧的所有目标
                    #   1.2.若不为空，设置new_object_flag=False
                    #       1.2.1.此帧中每个结果与上一帧每个结果相比较，比较依据为iou。
                    #           若当前帧某一个bbox与上一帧所有结果比较后iou均小于阈值，表明这个为新目标，将new_object_flag设置为True，并发送这一帧所有结果至前端
                    #           否则，break;
                    if last_pic:
                        for last_pic1 in last_pic:
                            # 上一帧某一个bbox的坐标信息
                            x1_hat, y1_hat, x2_hat, y2_hat = last_pic1['xyxy'][0], last_pic1['xyxy'][1], \
                                                             last_pic1['xyxy'][2], last_pic1['xyxy'][3]
                            if not compare_iou(x1, y1, x2, y2, x1_hat, y1_hat, x2_hat, y2_hat):
                                print("两张图片的Iou值没有超过阈值,可以保留:{}".format(cur_pic1['conf']))
                                compare_num += 1
                            else:
                                print("两张图片的Iou值太大,compare_num:{}".format(compare_num))
                                continue
                            if compare_num == len(last_pic):
                                new_object_flag = True
                                break
                    else:
                        new_object_flag = True
                        break

                if new_object_flag:
                    cur_pic.clear()
                    for *xyxy, conf, cls in reversed(det):
                        # 在原图上画框
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=3)
                        cur_pic1 = dict()
                        cur_pic1['cls'] = torch.tensor(cls).view(-1).tolist()[0]
                        cur_pic1['xyxy'] = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        cur_pic1['conf'] = torch.tensor(conf).view(-1).tolist()[0]
                        cur_pic.append(cur_pic1)

                        label = f'{names[int(cls)]} {conf:.2f}'
                        label_name = label.split(' ')[0]
                        cropped_image = im0_copy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        if cropped_image.shape[0] and cropped_image.shape[1]:
                            cropped_image = base64.b64encode(
                                cv2.imencode('.jpg', cropped_image)[1].tostring()).decode("utf-8")
                            abnormal_list.append((label_name, cropped_image), )
                            abnormal_list2.append(label_name, )
                    s = json.dumps(abnormal_list)
                    dangerousClient.publish("abnormal", payload=s)
                    print("------>发布一条abnormal_list的消息")
                    dangerousClient.publish(topic="warning",
                                            payload="Hazardous material detected in security area")
                    abnormal_list.clear()
                    abnormal_list2.clear()

                flow_danger = base64.b64encode(cv2.imencode('.jpg', im0)[1].tostring()).decode(
                    "utf-8")
                dangerousClient.publish(topic="flowDangerous", payload=flow_danger)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond


if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
        """
    dangerousClient = mqtt.Client()
    dangerousClient.connect(host="127.0.0.1", port=1883)
    # topic = 'dangerous'
    # agent = RTMP_AGENT(topic=topic)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/detect_data/test2.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.70, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.40, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # use GPU 0
    parser.add_argument('--view-img', action='store_true', default='True', help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    # check_requirements()
    # if len(sys.argv) == 2:
    #     topic = sys.argv[1]
    # elif len(sys.argv) == 3:
    #     topic = sys.argv[1]
    #     video_path = sys.argv[2]
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # 保留上次及此次检测结果的列表，列表里的类型为字典
            detect()
