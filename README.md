# [GONet](https://github.com/NHirose/GONET) PyTorch Traversability Estimation

PyTorch reproduction of the **GONet** monocular image-level traversability pipeline using the **GO Stanford** dataset.

This repository trains a GAN-based traversability estimator in three stages:

1. Train a DCGAN on **positive/traversable** images.
2. Train an inverse generator, `InvG`, to map real images into the GAN latent space.
3. Train a final GONet classifier, `FL`, using hand-labelled positive and negative images.

The trained model can then be evaluated on the hand-labelled test split and used to run inference on unlabelled frames, producing annotated videos with **GO / NO-GO** decisions.

---

## 1. Method overview

The GONet-style inference pipeline is:

```text
input image
    ↓
InvG
    ↓
latent vector z
    ↓
Generator
    ↓
reconstructed image
    ↓
Discriminator features from real and reconstructed images
    ↓
Final classifier
    ↓
traversability probability
```

More explicitly:

```python
z_hat = invg(img_real)
img_gen = generator(z_hat)

dis_real = discriminator(img_real)
dis_gen = discriminator(img_gen)

prob = classifier(
    img_real - img_gen,
    dis_real - dis_gen,
    dis_real,
)
```

The final output is a scalar:

```text
probability close to 1 → GO / traversable
probability close to 0 → NO-GO / non-traversable
```

---

## 2. Repository structure

Recommended repository layout:

```text
gonet_pytorch/
├── datasets/
│   ├── __init__.py
│   └── go_stanford.py
├── models/
│   ├── __init__.py
│   └── gonet.py
├── tools/
│   ├── inspect_preprocessing.py
│   ├── sweep_threshold.py
│   └── infer_unlablelled_gs.py
├── train_gan.py
├── train_invg.py
├── train_fl.py
├── evaluate_gonet.py
├── .gitignore
└── README.md
```

Expected generated folders after training:

```text
checkpoints/
├── gonet_gan/
├── gonet_invg/
└── gonet_fl/

outputs/
├── gonet_eval_test/
├── gonet_eval_test_thr085/
└── unlabelled_inference_videos/
```

Do not commit `checkpoints/`, `outputs/`, or the dataset to GitHub.

---

## 3. Step 0 — Create conda environment

Create and activate a clean environment:

```bash
conda create -n gonet_pytorch python=3.10 -y
conda activate gonet_pytorch
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install remaining dependencies:

```bash
pip install opencv-python numpy matplotlib tqdm scikit-learn pillow pandas
```

Verify installation:

```bash
python - << 'EOF'
import torch
import cv2
import numpy as np

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
EOF
```

You should see:

```text
CUDA available: True
```

if your GPU and CUDA setup are working correctly.

---

## 4. Step 1 — Download and arrange GO Stanford dataset

Download the GO Stanford dataset from:

```text
https://cvgl.stanford.edu/gonet/dataset/
```

After extraction, the dataset should look like:

```text
go_stanford_dataset/
├── hand_labelled_dataset
│   ├── data_test_annotation
│   │   ├── negative_L
│   │   ├── negative_R
│   │   ├── positive_L
│   │   └── positive_R
│   ├── data_train_annotation
│   │   ├── negative_L
│   │   ├── negative_R
│   │   ├── positive_L
│   │   └── positive_R
│   └── data_vali_annotation
│       ├── negative_L
│       ├── negative_R
│       ├── positive_L
│       └── positive_R
└── whole_dataset
    ├── data_test
    │   ├── positive_L
    │   ├── positive_R
    │   ├── unlabel_L
    │   └── unlabel_R
    ├── data_train
    │   ├── positive_L
    │   ├── positive_R
    │   ├── unlabel_L
    │   └── unlabel_R
    └── data_vali
        ├── positive_L
        ├── positive_R
        ├── unlabel_L
        └── unlabel_R
```


Change the path to wherever the dataset is.

---

## 5. Important preprocessing note

The released GO Stanford dataset images are already:

```text
128 × 128 × 3
```

Therefore, for this released dataset, the preprocessing is simply:

```text
BGR image
    ↓
RGB image
    ↓
normalize from [0, 255] to approximately [-1, 1]
    ↓
CHW tensor
```

## 6. Step 2 — Inspect preprocessing

Before training, verify that the dataset loader produces normal-looking images.

```bash
PYTHONPATH=. python tools/inspect_preprocessing.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --dataset-type positive \
  --split train \
  --side both \
  --output-dir outputs/preprocessing_positive_debug \
  --num-samples 30
```

Inspect the outputs:

```bash
xdg-open outputs/preprocessing_positive_debug
```

Also inspect the hand-labelled data:

```bash
PYTHONPATH=. python tools/inspect_preprocessing.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --dataset-type labelled \
  --split train \
  --side both \
  --output-dir outputs/preprocessing_labelled_debug \
  --num-samples 30
```

Inspect:

```bash
xdg-open outputs/preprocessing_labelled_debug
```

The images should look normal and not completely dark.

---

## 7. Step 3 — Verify model definitions

Run:

```bash
PYTHONPATH=. python models/gonet.py
```

Expected output:

```text
Device: cuda
Input image: torch.Size([4, 3, 128, 128])
Latent z: torch.Size([4, 100])
Generated image: torch.Size([4, 3, 128, 128])
Encoded z: torch.Size([4, 100])
Discriminator features: torch.Size([4, 512, 8, 8])
GONet probability: torch.Size([4, 1])
```

This confirms that the model tensor shapes are correct.

---

## 8. Step 4 — Train DCGAN on positive traversable images

The DCGAN is trained only on positive images from:

```text
whole_dataset/data_train/positive_L
whole_dataset/data_train/positive_R
```

It does **not** use negative or unlabelled images.

Run a smoke test first:

```bash
PYTHONPATH=. python train_gan.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --output-dir checkpoints/gonet_gan_smoke \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 0 \
  --nz 100 \
  --lr 2e-4 \
  --device cuda
```

Then train:

```bash
PYTHONPATH=. python train_gan.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --output-dir checkpoints/gonet_gan \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 0 \
  --nz 100 \
  --lr 2e-4 \
  --device cuda
```

Outputs:

```text
checkpoints/gonet_gan/
├── config.json
├── gan_latest.pt
├── gan_loss_curve.png
├── gan_epoch_0010.pt
├── gan_epoch_0020.pt
├── gan_epoch_0030.pt
└── samples/
```

Inspect generated samples:

```bash
xdg-open checkpoints/gonet_gan/samples
```

Inspect loss curve:

```bash
xdg-open checkpoints/gonet_gan/gan_loss_curve.png
```

### GAN checkpoint selection

In our run, GAN training became unstable after around epoch 30. The discriminator became too strong and the generator loss shot up.

The selected checkpoint was:

```bash
checkpoints/gonet_gan/gan_epoch_0020.pt
```

Use this checkpoint for the next stages.

---

## 9. Step 5 — Train inverse generator `InvG`

`InvG` learns:

```text
image → latent z → frozen Generator → reconstructed image
```

The generator is frozen. Only `InvG` is trained.

Smoke test:

```bash
PYTHONPATH=. python train_invg.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --gan-checkpoint checkpoints/gonet_gan/gan_epoch_0020.pt \
  --output-dir checkpoints/gonet_invg_smoke \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 0 \
  --nz 100 \
  --lr 1e-4 \
  --device cuda
```

Train:

```bash
PYTHONPATH=. python train_invg.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --gan-checkpoint checkpoints/gonet_gan/gan_epoch_0020.pt \
  --output-dir checkpoints/gonet_invg \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 0 \
  --nz 100 \
  --lr 1e-4 \
  --device cuda
```

Outputs:

```text
checkpoints/gonet_invg/
├── config.json
├── invg_latest.pt
├── invg_loss_curve.png
├── invg_epoch_0005.pt
├── invg_epoch_0010.pt
└── samples/
```

Inspect reconstructions:

```bash
xdg-open checkpoints/gonet_invg/samples
```

Each reconstruction image has:

```text
top row    = real images
bottom row = reconstructed images
```

The reconstructions do not need to be perfect. The goal is to obtain a meaningful projection into the traversable-image manifold.

---

## 10. Step 6 — Train final GONet classifier / FL

The final classifier is trained on the hand-labelled positive and negative images:

```text
hand_labelled_dataset/data_train_annotation/positive_L
hand_labelled_dataset/data_train_annotation/positive_R
hand_labelled_dataset/data_train_annotation/negative_L
hand_labelled_dataset/data_train_annotation/negative_R
```

The generator, discriminator, and `InvG` are frozen. Only the classifier is trained.

Run:

```bash
PYTHONPATH=. python train_fl.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --gan-checkpoint checkpoints/gonet_gan/gan_epoch_0020.pt \
  --invg-checkpoint checkpoints/gonet_invg/invg_latest.pt \
  --output-dir checkpoints/gonet_fl \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 0 \
  --nz 100 \
  --lr 1e-4 \
  --device cuda
```

Outputs:

```text
checkpoints/gonet_fl/
├── config.json
├── fl_best.pt
├── fl_latest.pt
├── fl_loss_curve.png
├── fl_metrics_curve.png
├── fl_epoch_0005.pt
├── fl_epoch_0010.pt
└── ...
```

The most important checkpoint is:

```bash
checkpoints/gonet_fl/fl_best.pt
```

---

## 11. Step 7 — Evaluate on hand-labelled test split

Evaluate using the hand-labelled test split:

```bash
PYTHONPATH=. python evaluate_gonet.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output-dir outputs/gonet_eval_test \
  --split test \
  --batch-size 32 \
  --num-workers 0 \
  --nz 100 \
  --threshold 0.5 \
  --device cuda
```

Outputs:

```text
outputs/gonet_eval_test/
├── metrics.json
├── predictions.csv
├── roc_curve.png
├── precision_recall_curve.png
└── score_histogram.png
```

View metrics:

```bash
cat outputs/gonet_eval_test/metrics.json
```

Initial result at threshold `0.5`:

```json
{
  "threshold": 0.5,
  "accuracy": 0.935625,
  "precision_positive_traversable": 0.9153754469606674,
  "recall_positive_traversable": 0.96,
  "f1_positive_traversable": 0.937156802928615,
  "roc_auc": 0.9756,
  "average_precision": 0.977484493896653,
  "tn_negative_correct": 729,
  "fp_negative_predicted_positive": 71,
  "fn_positive_predicted_negative": 32,
  "tp_positive_correct": 768
}
```

---

## 12. Step 8 — Threshold sweep

For robotics, false positives are dangerous:

```text
negative / no-go image predicted as go
```

So we sweep thresholds:

```bash
python tools/sweep_threshold.py \
  --predictions outputs/gonet_eval_test/predictions.csv
```

This creates:

```text
outputs/gonet_eval_test/threshold_sweep.csv
```

In our run, threshold `0.85` was selected as a safety-biased operating point.

At threshold `0.85`:

```text
Accuracy:       92.81%
ROC-AUC:        97.56%
Avg Precision:  97.75%
Go Precision:   96.60%
Go Recall:      88.75%
F1:             92.51%
```

Confusion matrix:

```text
                 Pred no-go   Pred go
True no-go          775          25
True go              90         710
```

This reduced unsafe no-go → go predictions from:

```text
71 at threshold 0.5
```

to:

```text
25 at threshold 0.85
```

Re-evaluate at the selected threshold:

```bash
PYTHONPATH=. python evaluate_gonet.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output-dir outputs/gonet_eval_test_thr085 \
  --split test \
  --batch-size 32 \
  --num-workers 0 \
  --nz 100 \
  --threshold 0.85 \
  --device cuda
```

---

## 13. Step 9 — Run inference on unlabelled frames and generate video

The unlabelled frames are located at:

```text
whole_dataset/data_test/unlabel_L
whole_dataset/data_test/unlabel_R
```

The inference video script does:

```text
unlabelled frames
    ↓
run GONet inference
    ↓
overlay probability and GO / NO-GO decision
    ↓
write annotated video
    ↓
write per-frame CSV
```

Run a quick test on 200 frames:

```bash
PYTHONPATH=. python tools/infer_unlablelled_gs.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --split test \
  --side L \
  --threshold 0.85 \
  --fps 10 \
  --scale 4 \
  --max-frames 200 \
  --device cuda
```

Outputs are written to:

```text
outputs/unlabelled_inference_videos/
```

Open:

```bash
xdg-open outputs/unlabelled_inference_videos
```

Run full left-camera unlabelled inference video:

```bash
PYTHONPATH=. python tools/infer_unlablelled_gs.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --split test \
  --side L \
  --threshold 0.85 \
  --fps 10 \
  --scale 4 \
  --device cuda
```

Run full right-camera unlabelled inference video:

```bash
PYTHONPATH=. python tools/infer_unlablelled_gs.py \
  --data-root /home/subodh/Downloads/go_stanford_dataset \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --split test \
  --side R \
  --threshold 0.85 \
  --fps 10 \
  --scale 4 \
  --device cuda
```

Each frame shows:

```text
GONet prob: <value>
Decision: GO or NO-GO
threshold = 0.85
```

The CSV contains:

```text
frame index
filename
building number
frame index from filename
side
probability
threshold
decision
```

---

## 14. Optional — Remove `PYTHONPATH=.`

Currently, commands use:

```bash
PYTHONPATH=. python ...
```

because the scripts import local packages:

```python
from datasets.go_stanford import ...
from models.gonet import ...
```

To remove the need for `PYTHONPATH=.`, create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "gonet-pytorch"
version = "0.1.0"
description = "PyTorch reproduction of GONet for GO Stanford traversability estimation"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["datasets*", "models*", "tools*"]
```

Then install in editable mode:

```bash
pip install -e .
```

After that, you can run:

```bash
python train_gan.py ...
python train_invg.py ...
python train_fl.py ...
python evaluate_gonet.py ...
python tools/infer_unlablelled_gs.py ...
```

without `PYTHONPATH=.`.

---

## 15. Troubleshooting

### `ModuleNotFoundError: No module named 'datasets'`

Run from the repository root using:

```bash
PYTHONPATH=. python ...
```

or install the repo in editable mode:

```bash
pip install -e .
```

---

### `RuntimeError: view size is not compatible with input tensor's size and stride`

Use `.reshape(...)` instead of `.view(...)` in `models/gonet.py`.

Inside `GONetClassifier.forward`, use:

```python
h = torch.abs(img_error).reshape(img_error.size(0), -1)
g = torch.abs(dis_error).reshape(dis_error.size(0), -1)
f = dis_real.reshape(dis_real.size(0), -1)
```

---

### `Tcl_AsyncDelete` or Tkinter crash during training

This is usually caused by matplotlib using a GUI backend while PyTorch workers are active.

Ensure scripts use:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Also run with:

```bash
--num-workers 0
```

---

### GAN loss explodes after many epochs

GAN instability is common.

In our run, training became unstable after around epoch 30. The selected checkpoint was:

```bash
checkpoints/gonet_gan/gan_epoch_0020.pt
```

Do not blindly use `gan_latest.pt` if the generated samples have degraded.

---

## 16. Data used by each training stage

| Stage | Dataset folders | Labels? | Purpose |
|---|---|---:|---|
| DCGAN | `whole_dataset/data_train/positive_L`, `positive_R` | No binary labels | Learn traversable image manifold |
| InvG | `whole_dataset/data_train/positive_L`, `positive_R` | No binary labels | Learn image → latent mapping |
| FL classifier | `hand_labelled_dataset/data_train_annotation/positive_*`, `negative_*` | Yes | Learn GO/NO-GO decision |
| Evaluation | `hand_labelled_dataset/data_test_annotation/positive_*`, `negative_*` | Yes | Test performance |
| Inference video | `whole_dataset/data_test/unlabel_L/R` | No | Visualize predictions on unlabelled sequences |

From the DCGAN training log:

```text
Batches per epoch: 2695
Batch size: 64
```

Therefore, the DCGAN consumed:

```text
2695 × 64 = 172,480 images per epoch
```

with `drop_last=True`.

---

## 17. Limitations

This is an image-level traversability classifier.

It does **not** output:

```text
obstacle mask
debris location
drivable segmentation
path-conditioned traversability
```

It answers only:

```text
Is this image likely traversable or not?
```

For warehouse AMRs, future extensions should include:

```text
path-aware inference
temporal smoothing
debris segmentation
uncertainty estimation
fusion with depth or projected planned path
```

---

## 18. Citation / acknowledgement

This project is based on the GONet traversability-estimation idea and the GO Stanford dataset.

Please cite and acknowledge the original [GONet](https://github.com/NHirose/GONET) authors and dataset source when using this repository publicly.