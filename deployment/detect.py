"""SCRFD 2.5G Face Detection — single script inference."""
import os, sys, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from scrfd_deploy_utils import preprocess_full, run_full_decode

def detect(model_path, img_path, thresh=0.10, output=None):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    prep = preprocess_full(img_path)
    if prep is None:
        print(f"Failed to load {img_path}")
        return np.empty((0, 5))
    inp, scale, orig = prep
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: inp})
    dets = run_full_decode(ort_outs, scale, orig, score_thresh=thresh)

    if output:
        img = cv2.imread(img_path)
        for d in dets:
            x1, y1, x2, y2 = map(int, [d[0], d[1], d[2], d[3]])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d[4]:.2f}", (x1, max(y1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(output, img)
        print(f"Saved: {output}")

    return dets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCRFD 2.5G Face Detection')
    parser.add_argument('input', help='Path to image or directory')
    parser.add_argument('--model', default='models/scrfd_2.5g_int8.onnx',
                        help='ONNX model path')
    parser.add_argument('--thresh', type=float, default=0.10,
                        help='Detection threshold (default: 0.10)')
    parser.add_argument('--output', default='output',
                        help='Output directory (default: output/)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        img_paths = [os.path.join(args.input, f) for f in os.listdir(args.input)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        img_paths = [args.input]

    for p in img_paths:
        out_name = os.path.splitext(os.path.basename(p))[0] + '_result.jpg'
        out_path = os.path.join(args.output, out_name)
        dets = detect(args.model, p, args.thresh, out_path)
        print(f"{os.path.basename(p)}: {len(dets)} face(s)")
