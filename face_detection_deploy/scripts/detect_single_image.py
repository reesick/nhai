import sys
import os
import os.path as osp
import time
import cv2

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from scrfd_detector import SCRFD


def resolve(path):
    return osp.abspath(osp.join(osp.dirname(__file__), '..', path))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/detect_single_image.py <image_path> [--thresh 0.5]")
        sys.exit(1)

    image_path = osp.abspath(sys.argv[1])
    det_thresh = 0.5
    if '--thresh' in sys.argv:
        idx = sys.argv.index('--thresh')
        det_thresh = float(sys.argv[idx + 1])

    if not osp.exists(image_path):
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    model_path = resolve('models/det.onnx')
    if not osp.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)

    detector = SCRFD(model_file=model_path)
    detector.prepare(-1, det_thresh=det_thresh)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not read image at {image_path}")
        sys.exit(1)

    in_name = osp.basename(image_path)

    start = time.time()
    bboxes, kpss = detector.detect(img, max_num=0, metric='default')
    elapsed = (time.time() - start) * 1000

    num_faces = bboxes.shape[0]

    for i in range(num_faces):
        x1, y1, x2, y2, score = bboxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if kpss is not None:
            kps = kpss[i].astype(int)
            for kp in kps:
                cv2.circle(img, tuple(kp), 2, (0, 0, 255), -1)

    output_dir = resolve('outputs')
    os.makedirs(output_dir, exist_ok=True)
    out_name = in_name
    out_path = osp.join(output_dir, out_name)
    success = cv2.imwrite(out_path, img)

    print()
    print("Detection Results")
    print("-" * 40)
    print(f"  Input file:    {in_name}")
    print(f"  Input path:    {image_path}")
    print(f"  Faces found:   {num_faces}")
    if num_faces > 0:
        print(f"  Scores:        {[f'{s:.4f}' for s in bboxes[:, 4]]}")
    print(f"  Inference:     {elapsed:.2f} ms")
    print(f"  Saved:         {success}")
    print(f"  Output file:   {out_name}")
    print(f"  Output path:   {out_path}")

    if not success:
        print("  ERROR: cv2.imwrite returned False — file was NOT saved!")
        sys.exit(1)


if __name__ == '__main__':
    main()
