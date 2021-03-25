import paho.mqtt.client as mqtt
import threading
import logging
import json
import time
import socket
import random
import paho.mqtt.publish as publish

MQTT_URL = '211.67.21.65'
# MQTT_URL = '127.0.0.1'
PORT = 1883
TOPIC = 'config'
# SERVER_IP='127.0.0.1'
SERVER_IP = '211.67.21.65'
SERVER_PORT = 10086
logging.basicConfig(filename='./logger.log', level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


class ConfigRetrive:
    # initialize config
    def __init__(self, url=MQTT_URL, port=PORT, topic=TOPIC):
        self.topic = topic
        self.port = port
        self.url = url
        self.config = dict()
        self.client = mqtt.Client()
        self.getConfigFromServer()

        def task(client, config):
            def on_connect(client, userdata, flags, rc):
                client.subscribe(topic=self.topic)
                logging.info('subscribe ' + self.topic + " OK!")

            def on_message(client, userdata, msg):
                logging.info('get new config: ' + str(msg.payload))
                kv = json.loads(msg.payload)
                config[list(kv.keys())[0]] = list(kv.values())[0]

            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(self.url, self.port, 60)
            client.loop_forever()

        self.thread_ = threading.Thread(target=task, args=(self.client, self.config))
        self.thread_.start()

    # get the config by key
    def get(self, key, default_value):
        if key in self.config.keys():
            return self.config[key]
        else:
            logging.warning(key + " not in config map")
            return default_value

    def getConfigFromServer(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('connecting to server')
        s.connect((SERVER_IP, SERVER_PORT))
        print('connected server sucessfully')
        cfg_byte = s.recv(1024 * 1024)
        # print(cfg_byte)
        # print('parsing')
        cfg_str = cfg_byte.decode()
        # print(cfg_str)
        self.config = json.loads(cfg_str)
        # print(self.config)

    def getConfigFromLocal(self):
        print("generate local parameters...")
        x0 = round(random.uniform(0, 0.5), 2)
        y0 = round(random.uniform(0, 0.5), 2)

        x1 = round(random.uniform(0.5, 1), 2)
        y1 = round(random.uniform(0, 0.5), 2)

        x2 = round(random.uniform(0.5, 1), 2)
        y2 = round(random.uniform(0.5, 1), 2)

        x3 = round(random.uniform(0, 0.5), 2)
        y3 = round(random.uniform(0.5, 1), 2)

        area_cfg = dict()
        area_cfg['dangerous_area'] = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        print("area_cfg['dangerous_area']=" + str(area_cfg['dangerous_area']))
        s = json.dumps(area_cfg)
        publish.single(topic='config', hostname=MQTT_URL, payload=s)


# -------------------------------------------以下为操作示例--------------------

import paho.mqtt.publish as publish
import numpy as np
import threading


def select_roi_random():
    print("generate local parameters...")
    x0 = round(random.uniform(0, 0.5), 2)
    y0 = round(random.uniform(0, 0.5), 2)

    x1 = round(random.uniform(0.5, 1), 2)
    y1 = round(random.uniform(0, 0.5), 2)

    x2 = round(random.uniform(0.5, 1), 2)
    y2 = round(random.uniform(0.5, 1), 2)

    x3 = round(random.uniform(0, 0.5), 2)
    y3 = round(random.uniform(0.5, 1), 2)

    area_cfg = dict()
    area_cfg['dangerous_area'] = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    # print("area_cfg['dangerous_area']=" + str(area_cfg['dangerous_area']))
    s = json.dumps(area_cfg)
    publish.single(topic='config', hostname=MQTT_URL, payload=s)
    return s


def select_roi_test_mode():
    x0, y0 = 0.2, 0.1
    x1, y1 = 0.3, 0.9
    x2, y2 = 0.8, 0.9
    x3, y3 = 0.9, 0.1
    area_cfg = dict()
    area_cfg['dangerous_area'] = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    s = json.dumps(area_cfg)
    publish.single(topic='config', hostname=MQTT_URL, payload=s)
    return s

# # 构造配置生成器
# bg = ConfigRetrive()
# # 通过get函数获取对应的值，key需要获取值对应的键，default_value为该key对应的默认值（自己写一个同类型的）
# val = bg.get(key='dangerous_area', default_value=[])
# # 获取到的val有规范的格式，可根据此格式进行后续操作
# print("dangerous_area = ", str(val))
# # print('val2 = ', val2)
# time.sleep(3)
