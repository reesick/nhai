import sys
import os
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(condition, message):
    status = PASS if condition else FAIL
    print(f"  [{status}] {message}")
    return condition


def main():
    base_dir = osp.join(osp.dirname(__file__), '..')
    model_path = osp.join(base_dir, 'models', 'det.onnx')
    images_dir = osp.join(base_dir, 'images')
    output_dir = osp.join(base_dir, 'outputs')
    sample_image = osp.join(images_dir, 'sample.jpg')

    print("=" * 55)
    print("  SCRFD-500M Deployment Validation")
    print("=" * 55)
    print()

    all_pass = True

    # 1. Model exists
    print("[1] Model file check")
    all_pass &= check(
        osp.exists(model_path),
        f"Model found at {model_path}"
    )
    if osp.exists(model_path):
        size_mb = osp.getsize(model_path) / (1024 * 1024)
        print(f"       Model size: {size_mb:.2f} MB")
    print()

    # 2. ONNX loads correctly (optional check - onnx package not required for inference)
    print("[2] ONNX model validation check")
    try:
        import onnx
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        all_pass &= check(True, "ONNX model loaded and validated")
    except ImportError:
        all_pass &= check(True, "ONNX package not installed (skip validation - not needed for inference)")
    except Exception as e:
        all_pass &= check(False, f"ONNX validation failed: {e}")
    print()

    # 3. ONNX Runtime works
    print("[3] ONNX Runtime inference check")
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(model_path)
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        all_pass &= check(True, f"ONNX Runtime session created")
        print(f"       Input:  {inputs[0].name} {inputs[0].shape}")
        print(f"       Outputs: {len(outputs)} tensors")
        print(f"       Providers: {sess._providers}")
    except Exception as e:
        all_pass &= check(False, f"ONNX Runtime failed: {e}")
    print()

    # 4. OpenCV works + detection + save to disk
    print("[4] OpenCV + detection + save-to-disk check")
    try:
        import cv2
        all_pass &= check(True, f"OpenCV version {cv2.__version__}")

        img = cv2.imread(sample_image)
        all_pass &= check(img is not None, f"Sample image loaded ({sample_image})")

        detector = __import__('scrfd_detector').SCRFD(model_file=model_path)
        detector.prepare(-1)
        bboxes, kpss = detector.detect(img, max_num=0)
        all_pass &= check(
            bboxes.shape[0] > 0,
            f"Detection ran: {bboxes.shape[0]} face(s) found"
        )
        if bboxes.shape[0] > 0:
            print(f"       Top score: {bboxes[0, 4]:.4f}")
            print(f"       BBox: {bboxes[0, :4].astype(int).tolist()}")
            if kpss is not None:
                print(f"       Keypoints: {kpss.shape[1]} landmarks per face")

        # Draw bounding boxes and save to disk
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if kpss is not None:
                kps = kpss[i].astype(int)
                for kp in kps:
                    cv2.circle(img, tuple(kp), 2, (0, 0, 255), -1)

        os.makedirs(output_dir, exist_ok=True)
        test_out = osp.join(output_dir, 'test_deploy_output.jpg')
        saved = cv2.imwrite(test_out, img)
        all_pass &= check(saved, f"Annotated image saved to disk (cv2.imwrite={saved})")
        all_pass &= check(osp.exists(test_out), f"File exists on disk after save")
        os.remove(test_out)
    except Exception as e:
        all_pass &= check(False, f"OpenCV/detection/save failed: {e}")
    print()

    # 5. Output folder writable
    print("[5] Output folder check")
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = osp.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        all_pass &= check(True, f"Output directory writable: {output_dir}")
    except Exception as e:
        all_pass &= check(False, f"Output directory not writable: {e}")
    print()

    # 6. Sample image exists
    print("[6] Sample image check")
    all_pass &= check(
        osp.exists(sample_image),
        f"Sample image exists at {sample_image}"
    )
    if osp.exists(sample_image):
        size_kb = osp.getsize(sample_image) / 1024
        print(f"       Image size: {size_kb:.1f} KB")
    print()

    print("=" * 55)
    if all_pass:
        print("  Result: \033[92mALL CHECKS PASSED\033[0m")
        print("  Deployment is ready.")
    else:
        print("  Result: \033[91mSOME CHECKS FAILED\033[0m")
        print("  Review the failures above.")
    print("=" * 55)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
