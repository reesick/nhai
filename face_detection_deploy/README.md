# SCRFD-500M Face Detection Deployment

A minimal, portable deployment package for SCRFD-500M single-stage face detection using ONNX Runtime.

## Folder Structure

```
face_detection_deploy/
├── models/
│   └── det.onnx              # SCRFD-500M ONNX model (~2.4 MB)
├── images/
│   └── sample.jpg            # Sample test image with faces
├── outputs/                  # Annotated detection results (generated)
├── scripts/
│   ├── detect_single_image.py # Detect faces in a single image
│   ├── detect_folder.py       # Batch detect faces in images/
│   └── test_deployment.py     # One-click deployment validation
├── requirements/
│   └── minimal_requirements.txt
├── scrfd_detector.py          # Standalone SCRFD detector class
├── deployment_inventory.txt   # File inventory manifest
└── README.md
```

## Installation

```bash
# 1. Create a fresh Python environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 2. Install runtime dependencies
pip install -r requirements/minimal_requirements.txt
```

### Dependencies

| Package       | Minimum Version | Purpose                      |
|---------------|-----------------|------------------------------|
| numpy         | 1.19.0          | Array operations             |
| opencv-python | 4.5.0           | Image I/O and preprocessing  |
| onnxruntime   | 1.7.0           | ONNX model inference engine  |
| onnx          | 1.8.0           | ONNX model validation        |

Total: 4 packages.

## Usage

### Validate Deployment

```bash
python scripts/test_deployment.py
```

This checks the model, ONNX Runtime, OpenCV, sample image, and output folder.

### Single Image Detection

```bash
python scripts/detect_single_image.py path/to/image.jpg
```

Optional: `--thresh 0.5` to set the confidence threshold.

Output: annotated image saved to `outputs/`.

Prints:
- Number of faces detected
- Confidence scores
- Inference time in ms

### Batch Detection

Place images in `images/`, then:

```bash
python scripts/detect_folder.py
```

Optional: `--thresh 0.5` to set the confidence threshold.

Output: annotated images saved to `outputs/`.

Prints:
- Images processed
- Total faces detected
- Average latency (ms)
- FPS

## Troubleshooting

| Problem                          | Solution                                                    |
|----------------------------------|-------------------------------------------------------------|
| `No module named 'onnxruntime'`  | Run `pip install -r requirements/minimal_requirements.txt`  |
| Model file not found             | Ensure `models/det.onnx` exists (~2.4 MB)                   |
| `cv2.imread` returns None        | Check image path and file format (jpg/png/bmp supported)    |
| CUDA not available               | `detector.prepare(-1)` forces CPU — always works             |
| Low detection accuracy           | Lower threshold: `--thresh 0.3`                             |
| Slow inference                   | Try smaller input size with `detector.prepare(-1, input_size=(640, 640))` |

## Expected Outputs

Each script produces annotated images in `outputs/` with:
- Green bounding boxes around detected faces
- Confidence score labels
- Red keypoint markers (eyes, nose, mouth corners)

### test_deployment.py

```
=======================================================
  SCRFD-500M Deployment Validation
=======================================================

[1] Model file check
  [PASS] Model found at ...\models\det.onnx
       Model size: 2.41 MB

[2] ONNX model load check
  [PASS] ONNX model loaded and validated
...

Result: ALL CHECKS PASSED
```

### detect_single_image.py

```
Number of faces detected: 3
Confidence scores: [0.9234, 0.8876, 0.6543]
Inference time: 45.23 ms
Result saved to: outputs\sample.jpg
```

### detect_folder.py

```
Found 5 images in ...\images\
  [1/5] sample.jpg: 3 face(s), 42.1 ms
  [2/5] group.jpg: 8 face(s), 44.7 ms
...

==================================================
Batch Detection Summary
==================================================
  Images processed:     5
  Total faces detected: 24
  Average latency:      43.50 ms
  FPS:                  22.99
```

## Model Info

- **Architecture**: SCRFD-500M
- **Backbone**: MobileNet-style (0.5M params)
- **Input**: dynamic (default 640x640)
- **Output**: bounding boxes + 5 facial keypoints
- **Format**: ONNX (opset 11)
- **Size**: ~2.4 MB
- **Framework**: ONNX Runtime (CPU)
