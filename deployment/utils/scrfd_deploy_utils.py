"""Shared utilities for SCRFD deployment analysis."""
import cv2
import numpy as np


def preprocess_full(img_path, size=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = size[0] - nh, size[1] - nw
    img_rgb = cv2.copyMakeBorder(img_rgb, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    inp = img_rgb.astype(np.float32)
    inp = (inp - 127.5) / 128.0
    inp = inp.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return inp, scale, (h, w)


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def decode_single(scores, bboxes, stride, score_thresh=0.1, input_size=(640, 640)):
    h, w = input_size
    s_h, s_w = h // stride, w // stride
    x_range = np.arange(0, s_w, dtype=np.float32)
    y_range = np.arange(0, s_h, dtype=np.float32)
    yv, xv = np.meshgrid(y_range, x_range, indexing='ij')
    points = np.stack([xv.ravel(), yv.ravel()], axis=-1) * stride
    n_anchors = 2
    n_loc = len(points)
    scores = scores.reshape(-1, n_anchors)
    bboxes = bboxes.reshape(-1, n_anchors, 4) * stride
    dets = []
    for a in range(n_anchors):
        keep = scores[:, a] > score_thresh
        if not keep.any():
            continue
        s_a = scores[keep, a]
        b_a = bboxes[keep, a]
        loc_idx = np.where(keep)[0] % n_loc
        p_a = points[loc_idx]
        boxes = distance2bbox(p_a, b_a, max_shape=input_size)
        for i in range(len(s_a)):
            dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(s_a[i])])
    return np.array(dets, dtype=np.float32) if dets else np.zeros((0, 5), dtype=np.float32)


def nms(dets, iou_thresh=0.45):
    if len(dets) < 2:
        return dets
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        if not keep[order[i]]:
            continue
        idx = order[i]
        xx1 = np.maximum(x1[idx], x1[order[i+1:]])
        yy1 = np.maximum(y1[idx], y1[order[i+1:]])
        xx2 = np.minimum(x2[idx], x2[order[i+1:]])
        yy2 = np.minimum(y2[idx], y2[order[i+1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[idx] + areas[order[i+1:]] - inter + 1e-10)
        keep[order[i+1:][iou > iou_thresh]] = False
    return dets[keep]


def run_full_decode(ort_outs, scale, orig_size, score_thresh=0.1, input_size=(640, 640)):
    strides = [8, 16, 32]
    all_dets = []
    for i, stride in enumerate(strides):
        scores = ort_outs[i][0].ravel()
        bboxes = ort_outs[3+i][0]
        dets = decode_single(scores, bboxes, stride, score_thresh, input_size)
        all_dets.append(dets)
    detections = np.concatenate(all_dets, axis=0) if all_dets else np.zeros((0, 5))
    if len(detections) > 0:
        detections = nms(detections, 0.45)
        detections[:, :4] /= scale
        detections[:, :4] = np.clip(detections[:, :4], 0,
            [orig_size[1], orig_size[0], orig_size[1], orig_size[0]])
    return detections
