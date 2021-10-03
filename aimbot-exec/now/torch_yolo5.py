from utils.general import non_max_suppression, scale_coords, xyxy2xywh, xywhn2xyxy
from two_class_threat import threat_handling
from models.experimental import attempt_load
from utils.augmentations import letterbox
from math import pow, sqrt
from util import check_gpu
from math import sqrt, pow
# from numba import njit
import numpy as np
import win32gui
import torch
import cv2


# 分析类
class FrameDetection5:
    # 类属性
    conf_thres = 0.5  # 置信度阀值
    iou_thres = 0.45  # 非极大值抑制
    win_class_name = None  # 窗口类名
    class_names = None  # 检测类名
    total_classes = 1  # 模型类数量
    COLORS = []
    WEIGHT_FILE = ['./']
    input_shape = (352, 352)  # 输入尺寸
    errors = 0  # 仅仅显示一次错误
    model, stride, names = None, None, None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'

    # 构造函数
    def __init__(self, hwnd_value):
        self.win_class_name = win32gui.GetClassName(hwnd_value)
        load_file('cf_best', self.WEIGHT_FILE)

        self.model = attempt_load(self.WEIGHT_FILE[0], map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # 检测是否在GPU上运行图像识别
        if self.half:
            # self.model.half()  # to FP16
            self.model(torch.zeros(1, 3, self.input_shape[0], self.input_shape[1]).to(self.device).type_as(next(self.model.parameters())))
            gpu_eval = check_gpu()
            gpu_message = {
                2: '小伙电脑顶呱呱啊',
                1: '战斗完全木得问题',
            }.get(gpu_eval, '您的显卡配置不够')
            print(gpu_message)
        else:
            print('中央处理器烧起来')
            print('PS:注意是否安装CUDA')

        try:
            with open('classes.txt', 'r') as f:
                self.class_names = [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            self.class_names = ['human-head', 'human-body']
        for i in range(len(self.class_names)):
            self.COLORS.append(tuple(16*x-1 for x in np.random.randint(16, size=3).tolist()))

    def detect(self, frames, recoil_coty, windoww=1600, adv_move=0):
        try:
            if frames.any():
                frame_height, frame_width = frames.shape[:2]
            frame_height += 0
            frame_width += 0
        except (AttributeError, UnboundLocalError) as e:  # cv2.error
            if self.errors < 2:
                print(str(e))
                self.errors += 1
            return 0, 0, 0, 0, 0, frames

        # 预处理
        img = letterbox(frames, self.input_shape, stride=self.stride)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
        img = torch.from_numpy(img).to(self.device)
        # img = img.half() if self.half else img.float()
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]  # img = img.unsqueeze(0)

        # 检测
        pred = self.model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=False)

        # 画框
        threat_list = []
        for i, dets in enumerate(pred):
            if len(dets):
                gn = torch.tensor(frames.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], frames.shape).round()
                for *xyxy, final_score, final_cls_ind in reversed(dets):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    box = xywhn2xyxy(torch.tensor(xywh).view(1, 4), frame_width, frame_height)[0]
                    cv2.rectangle(frames, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.COLORS[0], 2)
                    label = str(round(final_score.item(), 3))
                    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                    cv2.putText(frames, label, (int(x + w/2 - 4*len(label)), int(y + h/2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    if final_cls_ind == self.total_classes:
                        self.total_classes += 1

                    # 计算威胁指数(正面画框面积的平方根除以鼠标移动到目标距离)
                    h_factor = (0.5 if w >= h or (self.total_classes > 1 and final_cls_ind == 0) else 0.25)
                    dist = sqrt(pow(frame_width / 2 - (x + w / 2), 2) + pow(frame_height / 2 - (y + h * h_factor), 2))
                    threat_var = -(pow(w * h, 1/2) / dist * final_score if dist else 9999)
                    if final_cls_ind == 0 and self.total_classes > 1:
                        threat_var *= 4
                    threat_list.append([threat_var, [x, y, w, h], final_cls_ind])

        x0, y0, fire_pos, fire_close, fire_ok, frames = threat_handling(frames, windoww, threat_list, recoil_coty, frame_height, frame_width, self.total_classes, adv_move)

        return len(threat_list), int(x0), int(y0), fire_pos, fire_close, fire_ok, frames


# 加载配置与权重文件
def load_file(file, weight_filename):
    weights_filename = file + '.pt'
    weight_filename[0] += weights_filename
    return
