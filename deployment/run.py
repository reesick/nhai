"""Run SCRFD face detection on any image. Change IMAGE_PATH below."""
IMAGE_PATH = r"test\0_Parade_0_Parade_Parade_0_102.jpg"  # <-- PUT YOUR IMAGE PATH HERE

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
import onnxruntime
import cv2
from scrfd_deploy_utils import preprocess_full, run_full_decode

sess = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'models', 'scrfd_2.5g_int8.onnx'))
prep = preprocess_full(IMAGE_PATH)
if prep is None:
    print("Failed to load image"); exit(1)
inp, scale, orig = prep
outs = sess.run(None, {sess.get_inputs()[0].name: inp})
dets = run_full_decode(outs, scale, orig, score_thresh=0.10)

img = cv2.imread(IMAGE_PATH)
for d in dets:
    x1, y1, x2, y2 = map(int, [d[0], d[1], d[2], d[3]])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{d[4]:.2f}", (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

out_path = os.path.splitext(IMAGE_PATH)[0] + "_result.jpg"
cv2.imwrite(out_path, img)
print(f"{len(dets)} face(s) detected. Saved: {out_path}")
