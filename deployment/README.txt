SCRFD 2.5G Face Detection — Deployment Package
===============================================

Contents:
  models/
    scrfd_2.5g_fp32.onnx   — FP32 ONNX model (validated, recommended)
    scrfd_2.5g_int8.onnx    — INT8 quantized ONNX model
    model.pth                — Original PyTorch checkpoint

  utils/
    scrfd_deploy_utils.py   — Preprocessing, decoding, and NMS utilities

  detect.py                 — Single-script face detection

  test/                     — Sample test images

Quick Start
-----------
  1. Install dependencies:
     pip install -r requirements.txt

  2. Run on a single image:
     python detect.py test/0_Parade_0_Parade_marchingband_1_379.jpg

  3. Run on all test images:
     python detect.py test/

  4. Specify model (FP32 or INT8):
     python detect.py test/ --model models/scrfd_2.5g_int8.onnx

  5. Adjust threshold:
     python detect.py test/ --thresh 0.15

  Results are saved to output/ by default.

Models
------
  scrfd_2.5g_fp32.onnx (recommended)
    - Validated 100% identical to PyTorch checkpoint
    - ~32ms per frame on CPU, ~31 FPS
    - Threshold: 0.10 (sigmoid domain)

  scrfd_2.5g_int8.onnx
    - INT8 dynamic quantization
    - ~600ms per frame on CPU (slower due to integer ops)
    - Use only if FP32 is unavailable

Programmatic Usage
------------------
  import sys
  sys.path.insert(0, 'utils')
  from scrfd_deploy_utils import preprocess_full, run_full_decode

  import onnxruntime
  sess = onnxruntime.InferenceSession('models/scrfd_2.5g_fp32.onnx')

  prep = preprocess_full('test/image.jpg')
  inp, scale, orig_size = prep
  ort_outs = sess.run(None, {sess.get_inputs()[0].name: inp})
  detections = run_full_decode(ort_outs, scale, orig_size, score_thresh=0.10)
  # detections: (N, 5) — [x1, y1, x2, y2, confidence]

Notes
-----
  - Input: any image format supported by OpenCV
  - The model resizes to 640x640 internally with letterbox padding
  - Output boxes are in original image coordinates
  - Scores are QualityFocalLoss quality-aware (IoU), not binary confidence
  - Threshold 0.10 works well; lower = more detections but more false positives
