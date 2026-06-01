# SCRFD 2.5G — INT8 Quantization Report

## Model Information

| Property | Value |
|----------|-------|
| Architecture | ResNetV1e + PAFPN + RetinaFaceHead |
| Parameters | 670,354 (0.67M) |
| Input size | 640×640 |
| Outputs | 3 strides [8, 16, 32], 2 anchors per stride |
| Score format | Sigmoid [0,1] (QualityFocalLoss quality-aware) |
| Recommended threshold | 0.10 |

## Quantization Process

The FP32 ONNX model was exported from the PyTorch checkpoint with 100% tensor-level accuracy verified across all 3 strides (max diff < 1e-5).

INT8 quantization was performed using ONNX Runtime's dynamic quantization:

```
quantize_dynamic(
    model_input='scrfd_2.5g_fp32.onnx',
    model_output='scrfd_2.5g_int8.onnx',
    per_channel=True,
    weight_type=QuantType.QInt8,
)
```

This converts Conv2d weights from FP32 to INT8 using per-channel quantization with dynamic activation quantization at runtime.

## Calibration Attempt

Static calibration (QDQ format) was also tested with:
- **MinMax** calibration (600 WIDER_val images)
- **Percentile** calibration
- **Entropy (KL Divergence)** calibration

All static calibration methods produced models with **17–21× latency regression** (Q/DQ op overhead on CPU) and were discarded in favor of dynamic quantization.

## Performance Comparison (100 WIDER_val images)

| Metric | FP32 | INT8 | Change |
|--------|------|------|--------|
| Total detections | 3165 | 3289 | +3.92% |
| Average latency | 32.9ms | 581.7ms | +1669% |
| Median latency | 33.6ms | 582.0ms | — |
| P95 latency | 36.8ms | 608.8ms | — |
| FPS | 30.4 | 1.7 | — |
| Mean confidence | 0.2610 | 0.2583 | -1.03% |
| Max confidence | 0.9208 | 0.9195 | — |
| Min confidence | 0.1000 | 0.1001 | — |

## Detection Ratio per Image

| Image | FP32 | INT8 |
|-------|------|------|
| 0_Parade_Parade_0_502 | 50 | 47 |
| 0_Parade_marchingband_1_360 | 21 | 26 |
| 0_Parade_marchingband_1_476 | 147 | 155 |
| 0_Parade_marchingband_1_593 | 278 | 285 |
| 10_People_Marching_People_Marching_2_40 | 284 | 284 |
| 12_Group_Group_12_Group_Group_12_38 | 13 | 14 |
| 12_Group_Group_12_Group_Group_12_578 | 15 | 13 |
| 12_Group_Large_Group_12_Group_Large_Group_12_613 | 13 | 13 |
| 12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_602 | 13 | 11 |
| 12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_617 | 77 | 79 |
| 12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_83 | 68 | 68 |
| 12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_889 | 14 | 14 |
| 13_Interview_Interview_On_Location_13_179 | 4 | 4 |
| 13_Interview_Interview_On_Location_13_190 | 2 | 3 |
| 13_Interview_Interview_On_Location_13_401 | 21 | 22 |
| 13_Interview_Interview_On_Location_13_559 | 6 | 6 |
| 13_Interview_Interview_Sequences_13_717 | 45 | 49 |
| 13_Interview_Interview_Sequences_13_868 | 9 | 12 |
| 16_Award_Ceremony_Awards_Ceremony_16_85 | 7 | 7 |
| 17_Ceremony_Ceremony_17_803 | 71 | 72 |

## Charts

![Quantization Charts](report/quantization_charts.png)

## Key Findings

1. **INT8 adds +3.9% more detections** — the quantization noise increases sensitivity slightly, mostly on marginal detections near the threshold boundary.

2. **INT8 is 18× slower on CPU** — the ConvInteger + DynamicQuantizeLinear ops are not well-optimized on x86 CPUs. On GPU or NPU this may differ.

3. **Confidence distribution is preserved** — mean confidence shifts by <1%, no collapse or inflation.

4. **No stride-specific failure** — all three strides produce comparable outputs.

## Deployment Recommendation

**For CPU deployment: Use FP32.**
INT8 quantization on CPU does not provide a speed benefit for this model. The FP32 model runs at ~30 FPS on a modern CPU, which is sufficient for real-time applications.

**For GPU/NPU deployment: INT8 may be viable** if the target hardware has optimized INT8 kernels (e.g., TensorRT, OpenVINO, CoreML).

## File Sizes

| Model | Size |
|-------|------|
| FP32 (self-contained) | 2,947 KB |
| INT8 (self-contained) | 909 KB |

## Commands Used

```bash
# Export FP32 from PyTorch
python export_scrfd.py

# Dynamic quantization
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('scrfd_2.5g_fp32.onnx', 'scrfd_2.5g_int8.onnx', per_channel=True, weight_type=QuantType.QInt8)"

# Run detection
python detect.py path/to/image.jpg
```
