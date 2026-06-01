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
    images_dir = resolve('images')
    output_dir = resolve('outputs')
    model_path = resolve('models/det.onnx')

    det_thresh = 0.5
    if '--thresh' in sys.argv:
        idx = sys.argv.index('--thresh')
        det_thresh = float(sys.argv[idx + 1])

    if not osp.exists(images_dir):
        print(f"Error: images directory not found at {images_dir}")
        sys.exit(1)
    if not osp.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(valid_exts)
    ]
    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    detector = SCRFD(model_file=model_path)
    detector.prepare(-1, det_thresh=det_thresh)

    os.makedirs(output_dir, exist_ok=True)

    total_faces = 0
    total_time = 0.0
    processed = 0
    failed_saves = 0

    for fname in sorted(image_files):
        img_path = osp.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] {fname}: could not read file")
            continue

        start = time.time()
        bboxes, kpss = detector.detect(img, max_num=0, metric='default')
        elapsed = (time.time() - start) * 1000

        num_faces = bboxes.shape[0]
        total_faces += num_faces
        total_time += elapsed
        processed += 1

        for i in range(num_faces):
            x1, y1, x2, y2, score = bboxes[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if kpss is not None:
                kps = kpss[i].astype(int)
                for kp in kps:
                    cv2.circle(img, tuple(kp), 2, (0, 0, 255), -1)

        out_path = osp.join(output_dir, fname)
        saved = cv2.imwrite(out_path, img)
        if not saved:
            failed_saves += 1

        print(f"  [{processed}/{len(image_files)}] {fname}")
        print(f"       Faces: {num_faces}   Latency: {elapsed:.1f} ms")
        print(f"       Saved: {saved}   -> {out_path}")

    avg_latency = total_time / processed if processed > 0 else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    print()
    print("=" * 50)
    print("Batch Detection Summary")
    print("=" * 50)
    print(f"  Images processed:     {processed}")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Average latency:      {avg_latency:.2f} ms")
    print(f"  FPS:                  {fps:.2f}")
    print(f"  Failed saves:         {failed_saves}")
    print(f"  Output directory:     {output_dir}")

    if failed_saves:
        print(f"  WARNING: {failed_saves} file(s) failed to save!")


if __name__ == '__main__':
    main()
