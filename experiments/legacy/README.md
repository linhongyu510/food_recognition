# Legacy experiment scripts

These are the **original** single-file scripts from the initial research phase,
preserved verbatim for reference and reproducibility of the historical results.

They are **not** part of the `food_recognition` package: nothing here is
imported by `src/`, and they are excluded from linting and CI.

## Contents

| File | Notes |
| --- | --- |
| `model1_1.py`, `model1_2.py`, `model1_3.py` | Early CNN / transfer-learning iterations |
| `model2_1.py` | Semi-supervised experiment |
| `model3_1.py`, `model3_2.py` | EfficientNet-**B0** + CBAM, with Grad-CAM |
| `myAlexNet.py`, `myalexnet2.py`, `myVGG.py`, `myResNet.py` | From-scratch architecture studies |
| `mylinaer.py`, `softmax.py`, `simple_class_1.py` | Teaching/warm-up scripts |
| `otherdata.py` | Alternative dataset loading |

`../../legacy/model_utils/` holds the shared `data.py` / `model.py` /
`train.py` helpers these scripts import.

## Known limitations

Running these as-is will likely fail on a modern machine:

- **Hard-coded CUDA.** `torch.device("cuda:0")` and `.cuda()` calls mean they
  require an NVIDIA GPU; they will not fall back to CPU or Apple MPS.
- **Hard-coded relative paths** such as `food-11/training/labeled`, expected
  next to the script.
- **Deprecated APIs.** `register_backward_hook` (superseded by
  `register_full_backward_hook`) and `pretrained=` (superseded by `weights=`).
- **Duplication.** The CBAM block, dataset class and training loop are copied
  across several files with small divergences.
- **`model_utils/model.py` imports `timm`**, which is not a project dependency.

The maintained pipeline in `src/food_recognition/` addresses all of the above.
Use these files to understand what was tried historically, not as a starting
point for new work.
