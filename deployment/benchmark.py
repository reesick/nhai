"""Benchmark: annotate every image, encode latency in filename."""
import os, sys, time, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from scrfd_deploy_utils import preprocess_full, run_full_decode

def benchmark(model_path, input_dir, output_dir, thresh=0.10):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    imgs.sort()

    for fname in imgs:
        path = os.path.join(input_dir, fname)
        prep = preprocess_full(path)
        if prep is None:
            print(f"SKIP {fname}")
            continue

        inp, scale, orig = prep
        t0 = time.perf_counter()
        ort_outs = sess.run(None, {sess.get_inputs()[0].name: inp})
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        dets = run_full_decode(ort_outs, scale, orig, score_thresh=thresh)
        img = cv2.imread(path)
        for d in dets:
            x1, y1, x2, y2 = map(int, [d[0], d[1], d[2], d[3]])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{d[4]:.2f}", (x1, max(y1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        stem = os.path.splitext(fname)[0]
        out_name = f"{latency_ms:.1f}ms_{stem}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img)
        print(f"{latency_ms:7.1f}ms  {len(dets):3d} faces  {out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCRFD benchmark + annotation')
    parser.add_argument('input', help='Input image directory')
    parser.add_argument('--model', default='models/scrfd_2.5g_int8.onnx')
    parser.add_argument('--thresh', type=float, default=0.10)
    parser.add_argument('--output', default=None,
                        help='Output dir (default: benchmark_output inside input dir)')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, 'benchmark_output')
    benchmark(args.model, input_dir, output_dir, args.thresh)
