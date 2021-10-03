from two_class_threat import threat_handling
from math import pow, sqrt
from util import check_gpu
from math import sqrt, pow
# from numba import njit
from time import time
import torch.nn as nn
import numpy as np
import torchvision
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


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 加载配置与权重文件
def load_file(file, weight_filename):
    weights_filename = file + '.pt'
    weight_filename[0] += weights_filename
    return


# yolov5的官方nms函数
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=100):
    """
    Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
    list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


# 坐标(0到1)xywh转xyxy(左上角右下角)
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 坐标(0到1)xyxy(左上角右下角)转xywh
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# 坐标(0到1)xywh转实际xywh
def xywhn2xyxy(x, w=416, h=416, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


# 坐标(处理后的图)xyxy转实际原图xyxy
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


# 坐标(处理后的图)xyxy转实际图xyxy
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


# 计算iou
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# 加载模型
def attempt_load(weights, map_location=None, inplace=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(weights, map_location=map_location)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


# yolov5加载模型类
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# yolov5自动填充
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# yolov5模型集合
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
