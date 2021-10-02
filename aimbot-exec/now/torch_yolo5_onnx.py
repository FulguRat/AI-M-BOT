from two_class_threat import threat_handling
from util import check_gpu, millisleep
from math import sqrt, pow
# from numba import njit
import numpy as np
import onnxruntime
import torchvision
import win32gui
import torch
import cv2


# 分析类
class FrameDetection5:
    # 类属性
    std_confidence = 0.5  # 置信度阀值
    nms_thd = 0.45  # 非极大值抑制
    win_class_name = None  # 窗口类名
    class_names = None  # 检测类名
    total_classes = 1  # 模型类数量
    COLORS = []
    WEIGHT_FILE = ['./']
    input_shape = (352, 352)  # 输入尺寸
    EP_list = onnxruntime.get_available_providers()  # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] Tensorrt优先于CUDA优先于CPU执行提供程序
    session, device_name = None, None
    input_name, output_name = None, None
    stride=[8, 16, 32]
    anchor_list= [[10, 13, 16,30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156,198, 373, 326]]
    anchor = np.array(anchor_list).astype(np.float).reshape(3, -1, 2)
    errors = 0  # 仅仅显示一次错误

    # 构造函数
    def __init__(self, hwnd_value):
        self.win_class_name = win32gui.GetClassName(hwnd_value)
        load_file('yolov5s', self.WEIGHT_FILE)

        # 检测是否在GPU上运行图像识别
        self.device_name = onnxruntime.get_device()
        try:
            self.session = onnxruntime.InferenceSession(self.WEIGHT_FILE[0], providers=['CUDAExecutionProvider'])  # 默认推理构造
        except RuntimeError:
            self.session = onnxruntime.InferenceSession(self.WEIGHT_FILE[0], providers=['CPUExecutionProvider'])  # 使用cpu推理
            self.device_name = 'CPU'
        if self.device_name == 'GPU':
            gpu_eval = check_gpu()  # 简单检查显卡性能
            gpu_message = {
                2: '小伙电脑顶呱呱啊',
                1: '战斗完全木得问题',
            }.get(gpu_eval, '您的显卡配置不够')
            print(gpu_message)
        else:
            print('中央处理器烧起来')
            print('PS:注意是否安装CUDA')

        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

        try:
            with open('classes.txt', 'r') as f:
                self.class_names = [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            self.class_names = ['human-head', 'human-body']
        for i in range(len(self.class_names)):
            self.COLORS.append(tuple(16*x-1 for x in np.random.randint(16, size=3).tolist()))

    # def sigmoid(self, x):  # 生成深度神经网络的构建块@#￥%……&*
    #     return 1 / (1 + np.exp(-x))

    def get_input_name(self):  # 获取输入节点名称
        input_name=[]
        for node in self.session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):  # 获取输出节点名称
        output_name=[]
        for node in self.session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_tensor):  # 获取输入tensor
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed

    def detect(self, frames, recoil_coty, windoww=1600, adv_move=0):
        try:
            if frames.any():
                frame_height, frame_width = frames.shape[:2]
            frame_height += 0  # 这是为了检查宽高是否为正常值
            frame_width += 0
        except (AttributeError, UnboundLocalError) as e:  # cv2.error
            if self.errors < 20:
                print(str(e))
                self.errors += 1
            millisleep(1)
            return 0, 0, 0, 0, 0, frames

        # 超参数设置
        area = self.input_shape[0] * self.input_shape[1]
        size = [int(area / self.stride[0] ** 2), int(area / self.stride[1] ** 2), int(area / self.stride[2] ** 2)]
        feature = [[int(j / self.stride[i]) for j in self.input_shape] for i in range(3)]

        # 读取图片
        src_size = frames.shape[:2]

        # 图片填充并归一转化
        img = letterbox(frames, self.input_shape, stride=32)[0].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # 归一化
        img = img.astype(dtype=np.float32) / 255.0

        # 维度扩张
        img=np.expand_dims(img, axis=0)

        # 前向推理
        input_feed = self.get_input_feed(img)

        # 检测
        output = self.session.run(output_names=self.output_name, input_feed=input_feed)

        dets = infer(img, output, size, feature, self.total_classes, self.std_confidence, self.nms_thd, src_size, self.stride, self.anchor)

        # 画框
        threat_list = []
        if not (dets is None):
            for *box, final_score, final_cls_ind in dets:
                cv2.rectangle(frames, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.COLORS[0], 2)
                label = str(round(final_score.item(), 3))
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                cv2.putText(frames, label, (int(x + w/2 - 4*len(label)), int(y + h/2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if final_cls_ind == self.total_classes:
                    self.total_classes += 1

                # 计算威胁指数(正面画框面积的平方根除以鼠标移动到目标距离乘以置信度)
                h_factor = (0.5 if w >= h or (self.total_classes > 1 and final_cls_ind == 0) else 0.25)
                dist = sqrt(pow(frame_width / 2 - (x + w / 2), 2) + pow(frame_height / 2 - (y + h * h_factor), 2))
                threat_var = -(pow(w * h, 1/2) / dist * final_score if dist else 9999)  # 用负数因为一会排序方便
                if final_cls_ind == 0 and self.total_classes > 1:  # 识别为头威胁加倍
                    threat_var *= 4
                threat_list.append([threat_var, [x, y, w, h], final_cls_ind])

        x0, y0, fire_pos, fire_close, fire_ok, frames = threat_handling(frames, windoww, threat_list, recoil_coty, frame_height, frame_width, self.total_classes, adv_move)

        return len(threat_list), int(x0), int(y0), fire_pos, fire_close, fire_ok, frames


# 执行前向操作预测输出
def infer(img, pred, size, feature, class_num, conf_thres, iou_thres, src_size, stride, anchor):
    #提取出特征
    y = []
    y.append(torch.tensor(pred[0].reshape(-1, size[0]*3, 5+class_num)).sigmoid())
    y.append(torch.tensor(pred[1].reshape(-1, size[1]*3, 5+class_num)).sigmoid())
    y.append(torch.tensor(pred[2].reshape(-1, size[2]*3, 5+class_num)).sigmoid())

    grid = []
    for k, f in enumerate(feature):
        grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

    z = []
    for i in range(3):
        src = y[i]

        xy = src[..., 0:2] * 2. - 0.5
        wh = (src[..., 2:4] * 2) ** 2
        dst_xy = []
        dst_wh = []
        for j in range(3):
            dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + torch.tensor(grid[i])) * stride[i])
            dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
        src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
        src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
        z.append(src.view(1, -1, 6))

    results = torch.cat(z, 1)
    results = nms(results, conf_thres, iou_thres)

    #映射到原始图像
    img_shape=img.shape[2:]

    for det in results:  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_shape, det[:, :4], src_size).round()

    return det


# 标注坐标转换
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


# 图片归一化
def letterbox(img, new_shape=(352, 352), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# 坐标对应到原始图像上
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''
    坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
    :param img1_shape: 输入尺寸
    :param coords: 输入坐标
    :param img0_shape: 映射的尺寸
    :param ratio_pad:
    :return:
    '''

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
    coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
    coords[:, :4] /= gain  # 将box坐标对应到原始图像上
    clip_coords(coords, img0_shape)  # 边界检查
    return coords


# 单类非极大值抑制函数
def nms(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    xc = prediction[..., 4] > conf_thres  # candidates
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

    return output


# 查看是否越界
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# 加载配置与权重文件
def load_file(file, weight_filename):
    weights_filename = file + '.onnx'
    weight_filename[0] += weights_filename
    return
