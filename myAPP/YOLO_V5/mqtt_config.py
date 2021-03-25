import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import json
import base64

dangerousClient = mqtt.Client()
dangerousClient.connect(host="211.67.21.65", port=1883)

