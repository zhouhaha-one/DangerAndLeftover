from django.db import models
import torch
import time
from datetime import datetime, timedelta


# Create your models here.
class DangerInfo(models.Model):
    frame_id = models.IntegerField(primary_key=True, default=0)
    class_name = models.CharField(max_length=32)  # 检测出来的危险品类别
    x1 = models.FloatField(default=0)
    y1 = models.FloatField(default=0)
    x2 = models.FloatField(default=0)
    y2 = models.FloatField(default=0)

    # @property
    # # 计算两个bbox的iou
    # def compute_iou(self, x1_hat, y1_hat, x2_hat, y2_hat):
    #     # Intersection area
    #     inter = (torch.min(self.x2, x2_hat) - torch.max(self.x1, x1_hat)).clamp(0) * \
    #             (torch.min(self.y2, y2_hat) - torch.max(self.y1, y1_hat)).clamp(0)
    #
    #     eps = 1e-9
    #     # Union Area
    #     w1, h1 = self.x2 - self.x1, self.y2 - self.y1 + eps
    #     w2, h2 = x2_hat - x1_hat, y2_hat - y1_hat + eps
    #     union = w1 * h1 + w2 * h2 - inter + eps
    #
    #     iou = inter / union
    #     return iou
    #
    # # 比较两帧之间的相似度,比较的两帧的最大间距默认为50帧,
    # def compare_2frame(self, c_name, x1_hat, y1_hat, x2_hat, y2_hat, iou_threshold=0.65):
    #     if self.class_name == c_name:
    #         return False
    #     else:
    #         if self.compute_iou(self, x1_hat, y1_hat, x2_hat, y2_hat) >= iou_threshold:
    #             return False
    #         return True
