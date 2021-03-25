IgnoreExtensions = True
import json
import threading
from datetime import date, timedelta
import logging
import paho.mqtt.client as mqtt
from django.http import HttpResponse
from dwebsocket.decorators import accept_websocket
from .config import *
import torch
import cv2
import base64
import paho.mqtt.publish as publish
from myAPP.models import DangerInfo


# from .configRetrive import ConfigRetrive
#
# logging.basicConfig(filename='logger.log', level=logging.INFO,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S', )
# logging.info("initiate configRetrive")
# config = ConfigRetrive()


@accept_websocket
def warning(request):
    if request.is_websocket:
        lock = threading.RLock()
        try:
            lock.acquire()
            logging.info("receive a new ws about warning")
            wsclients = request.websocket

            def on_connect(clent, userdata, flags, rc):
                print("Connected with result code " + str(rc))
                client.subscribe(topic="warning")
                print('subscribe warning successfully')

            def on_message(client, userdata, msg):
                wsclients.read()
                if wsclients.is_closed():
                    client.disconnect()
                wsclients.send(msg.payload)

            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(host=msg_middleware_host, port=msg_middleware_port)
            client.loop_forever()
        finally:
            lock.release()


@accept_websocket
def flowDangerous(request):
    if request.is_websocket:
        lock = threading.RLock()
        try:
            lock.acquire()
            logging.info("receive a new ws about offlineImage")
            wsclients = request.websocket

            def on_connect(clent, userdata, flags, rc):
                # print("Connected with result code " + str(rc))
                client.subscribe(topic="flowDangerous")
                # print('subscribe offlineImage successfully')

            def on_message(client, userdata, msg):
                wsclients.read()
                if wsclients.is_closed():
                    client.disconnect()
                wsclients.send(msg.payload)
                print("send flowDangerous suncessfully!")

            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(host=msg_middleware_host, port=msg_middleware_port)
            client.loop_forever()
        finally:
            lock.release()


# 判断在指定时间间隔内是否重复显示,默认时间间隔为10min
# def is_repeated(self, dclass, time_gap=10):
#     danger_object = DangerInfo.objects.filter(dangerClass=dclass).last()
#     now = datetime.now()  # 返回当前日期和时间
#     print("now=" + str(now))
#     max_time = danger_object.datetime + timedelta(minutes=time_gap)
#     if max_time > now:
#         print("指定时间间隔内图片没有重复")
#         return False
#     else:
#         print("指定时间间隔内图片重复")
#         return True

# 计算两个bbox的iou
def compare_iou(x1, y1, x2, y2, x1_hat, y1_hat, x2_hat, y2_hat, iou_threshold=0.65):
    # Intersection area
    inter = (torch.min(x2, x2_hat) - torch.max(x1, x1_hat)).clamp(0) * \
            (torch.min(y2, y2_hat) - torch.max(y1, y1_hat)).clamp(0)

    eps = 1e-9
    # Union Area
    w1, h1 = x2 - x1, y2 - y1 + eps
    w2, h2 = x2_hat - x1_hat, y2_hat - y1_hat + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou < iou_threshold


@accept_websocket
def abnormal(request):
    """
    展示危险物品类别及图片信息
    :param request:
    :return: {'img':base64img,'name':knife}
    """
    if request.is_websocket:
        lock = threading.RLock()
        try:
            lock.acquire()
            logging.info("receive a new ws about abnormal")
            wsclients = request.websocket

            def on_connect(clent, userdata, flags, rc):
                # print("Connected with result code " + str(rc))
                client.subscribe(topic="abnormal")
                # print('subscribe abnormal successfully')

            def on_message(client, userdata, msg):
                wsclients.read()
                if wsclients.is_closed():
                    client.disconnect()
                wsclients.send(msg.payload)
                print("send abnormal suncessfully!")

            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(host=msg_middleware_host, port=msg_middleware_port)
            client.loop_forever()
        finally:
            lock.release()






