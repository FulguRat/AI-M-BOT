from time import time, sleep
from tqdm import tqdm
from sys import exit
import numpy as np
import onnxruntime
import cv2
import os


# 分析预测数据
# @njit(fastmath=True)
def analyze(predictions, ratio):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    return boxes_xyxy, scores


# 从yolox复制的预处理函数
def preprocess(img, input_size, swap=(2, 0, 1)):
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


# 从yolox复制的单类非极大值抑制函数
# @njit(fastmath=True)
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


# 从yolox复制的多类非极大值抑制函数
def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


# 从yolox复制的多类非极大值抑制函数(class-agnostic方式)
def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


# 从yolox复制的多类非极大值抑制函数(class-aware方式)
def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


# 从yolox复制的后置处理函数
def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


prep_label_path = './测试标注/'  # 自己改位置名
if not os.path.isdir(prep_label_path):
    os.makedirs(prep_label_path)

pics_path = './预先处理/'  # 自己改位置名
if not os.path.isdir(pics_path):
    os.makedirs(pics_path)


if __name__ == '__main__':
    if not os.listdir(pics_path):
        exit(0)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 设置工作路径

    WEIGHT_FILE = './yolox_tiny.onnx'

    try:
        session = onnxruntime.InferenceSession(WEIGHT_FILE, providers=['CUDAExecutionProvider'])  # 推理构造
    except RuntimeError:
        session = onnxruntime.InferenceSession(WEIGHT_FILE, providers=['CPUExecutionProvider'])  # 推理构造

    start_time = time()

    for pics in tqdm(os.listdir(pics_path)):
        pic_names, pic_ext = os.path.splitext(pics)
        file_create = open(prep_label_path + pic_names + '.txt', 'w')
        pictures = cv2.imdecode(np.fromfile(pics_path + pics, dtype=np.uint8), -1)
        pictures = pictures[..., :3]
        # cv2.imshow('Show frame', pictures)
        # cv2.waitKey(1)

        frame_height, frame_width = pictures.shape[:2]
        
        # 预处理
        img, ratio = preprocess(pictures, (512, 320))
        
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)[0]
        
        predictions = demo_postprocess(output, (512, 320))[0]
        boxes_xyxy, scores = analyze(predictions, ratio)
        dets = multiclass_nms(boxes_xyxy, scores, 0.3, 0.3)
        
        # 画框
        if not (dets is None):
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for (box, final_score, final_cls_ind) in zip(final_boxes, final_scores, final_cls_inds):
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                x = x + w/2
                y = y + h/2
                file_create.write(f'{final_cls_ind:.0f} {(x/frame_width):.6f} {(y/frame_height):.6f} {(w/frame_width):.6f} {(h/frame_height):.6f}\n')

        file_create.close()

    end_time = time() - start_time

    print(f'Time used: {end_time:.3f} s')
    sleep(3)
