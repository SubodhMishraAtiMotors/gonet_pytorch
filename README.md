# GONet PyTorch Traversability Estimation

PyTorch reproduction of the **GONet** monocular image-level traversability pipeline using the **GO Stanford** dataset.

This repository trains a GAN-based traversability estimator in three stages and then extends it with a temporal LSTM model, **GONet+T**:

1. Train a DCGAN on **positive/traversable** images.
2. Train an inverse generator, `InvG`, to map real images into the GAN latent space.
3. Train a final GONet classifier, `FL`, using hand-labelled positive and negative images.
4. Train a GONet+T-style temporal model on pseudo-labelled unlabelled sequences.

The trained vanilla GONet model can be evaluated on the hand-labelled test split and used to run inference on unlabelled frames, producing annotated videos with **GO / NO-GO** decisions. The GONet+T extension then smooths frame-wise traversability predictions over time using an LSTM.

All results reported here use only the **GO Stanford** dataset.

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

### GONet+T temporal extension

The temporal extension keeps the trained GONet feature extraction idea, but applies it over image sequences.

For each frame, the following three quantities are computed:

```text
φR = |image_real - image_generated|
φD = |discriminator_feature_real - discriminator_feature_generated|
φF = discriminator_feature_real
```

These are reduced into compact features:

```text
φR → Linear → 10D
φD → Linear → 10D
φF → Linear → 10D
```

The three 10D vectors are concatenated:

```text
temporal feature per frame = 30D
```

A single-layer LSTM then predicts a traversability score for each frame in the sequence:

```text
image sequence
    ↓
frozen Generator + InvG + Discriminator
    ↓
per-frame 30D GONet feature
    ↓
LSTM, hidden_dim = 64
    ↓
linear output
    ↓
sigmoid
    ↓
temporally smoothed traversability probability
```

Important design decision:

```text
GONet+T is trained as a temporal stabilizer of vanilla GONet outputs.
It is not trained as a new classifier with new human labels.
```

In this reproduction, GONet+T is trained on pseudo-labels produced by the trained vanilla GONet model over unlabelled GO Stanford sequences.

---

## 2. Repository structure

Recommended repository layout:

```text
gonet_pytorch/
├── datasets/
│   ├── __init__.py
│   ├── go_stanford.py
│   └── go_stanford_temporal.py
├── models/
│   ├── __init__.py
│   ├── gonet.py
│   └── gonet_temporal.py
├── tools/
│   ├── inspect_preprocessing.py
│   ├── sweep_threshold.py
│   ├── infer_unlablelled_gs.py
│   ├── build_unlabelled_sequence_manifest.py
│   ├── pseudo_label_unlabelled_manifest.py
│   ├── visualize_pseudo_labels.py
│   ├── compare_gonet_vs_gonet_t.py
│   ├── run_gonet_t_full_inference.py
│   └── summarize_gonet_t_comparisons.py
├── train_gan.py
├── train_invg.py
├── train_fl.py
├── train_gonet_t.py
├── evaluate_gonet.py
├── .gitignore
└── README.md
```

Expected generated folders after training:

```text
checkpoints/
├── gonet_gan/
├── gonet_invg/
├── gonet_fl/
└── gonet_t_logit_lpred05_lsmooth05/

outputs/
├── gonet_eval_test/
├── gonet_eval_test_thr085/
├── unlabelled_inference_videos/
├── gonet_t_manifests/
├── gonet_t_pseudo_labels/
├── gonet_t_compare_best_logit/
└── gonet_t_full_inference/
```


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

Set the GO Stanford dataset path once before running the commands below:

```bash
export DATA_ROOT=/path/to/go_stanford_dataset
```

All commands below use `$DATA_ROOT`. Replace `/path/to/go_stanford_dataset` with the actual location of your extracted GO Stanford dataset.

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

Do **not** apply the circular fisheye crop/mask parameters from the original ROS code to the released GO Stanford images.

Those original parameters:

```python
xc = 310
yc = 321
radius = 275
```

were meant for the raw live camera stream, not for the already-preprocessed dataset images.

---

## 6. Step 2 — Inspect preprocessing

Before training, verify that the dataset loader produces normal-looking images.

```bash
PYTHONPATH=. python tools/inspect_preprocessing.py \
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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
  --data-root $DATA_ROOT \
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


## 14. Step 10 — Build temporal manifests for GONet+T

GONet+T requires temporal sequences rather than randomly shuffled images.

The GO Stanford whole dataset filenames follow:

```text
img_buildX_Y_Z.jpg
```

where:

```text
X = building number
Y = time order / frame index
Z = camera side, L or R
```

For temporal modelling, frames are grouped by:

```text
building_id + side
```

and then split into contiguous temporal segments whenever the frame index has a gap.

This is important. A building is not treated as one continuous sequence. For example:

```text
img_build10_1000_L.jpg
img_build10_1001_L.jpg
img_build10_1002_L.jpg

gap

img_build10_1027_L.jpg
img_build10_1028_L.jpg
```

These become two different temporal segments.

No frames are discarded during manifest creation.

Build manifests for training unlabelled left/right frames:

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split train \
  --side L \
  --output-dir outputs/gonet_t_manifests
```

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split train \
  --side R \
  --output-dir outputs/gonet_t_manifests
```

Outputs:

```text
outputs/gonet_t_manifests/
├── train_unlabel_L_manifest.csv
├── train_unlabel_L_segments.csv
├── train_unlabel_R_manifest.csv
└── train_unlabel_R_segments.csv
```

Each manifest row contains:

```text
global_index
segment_id
segment_local_index
building_id
frame_idx
side
filename
path
```

Each segment summary contains:

```text
segment_id
building_id
side
start_frame_idx
end_frame_idx
num_frames
```

In our train split setup:

```text
train unlabel L + R:
4104 temporal segments
102,712 frames
minimum segment length: 2
maximum segment length: 239
mean segment length: ~25 frames
```

When using `--max-length 64` during training, long segments are split into chunks, but no frames are discarded:

```text
4298 chunks
102,712 frames preserved
maximum chunk length: 64
```

---

## 15. Step 11 — Generate pseudo-labels for unlabelled sequences

GONet+T is trained using pseudo-labels from vanilla GONet.

The trained vanilla GONet checkpoint is applied to every unlabelled frame in the temporal manifests. The output probability becomes the pseudo-label:

```text
unlabelled frame
    ↓
vanilla GONet
    ↓
prob_traversable
```

Generate pseudo-labels for train unlabelled L/R:

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/train_unlabel_L_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/train_unlabel_R_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

The resulting CSVs contain all manifest columns plus:

```text
prob_traversable
```

These pseudo-labels are used as temporal training targets for GONet+T.

---

## 16. Step 12 — Visualize pseudo-labels before temporal training

Before training the LSTM, inspect vanilla GONet probabilities over time:

```bash
PYTHONPATH=. python tools/visualize_pseudo_labels.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
  --output-dir outputs/gonet_t_visual_checks/train_unlabel_L \
  --top-k 10 \
  --threshold 0.85 \
  --fps 3 \
  --scale 4 \
  --make-videos
```

```bash
PYTHONPATH=. python tools/visualize_pseudo_labels.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --output-dir outputs/gonet_t_visual_checks/train_unlabel_R \
  --top-k 10 \
  --threshold 0.85 \
  --fps 3 \
  --scale 4 \
  --make-videos
```

Outputs include:

```text
segment probability plots
annotated segment videos
segment_probability_summary.csv
```

This helps check whether vanilla GONet is temporally noisy and whether GONet+T is likely to help.

---

## 17. Step 13 — Verify temporal dataset and model

The temporal dataset returns variable-length segments:

```text
images: [T_i, 3, 128, 128]
labels: [T_i, 1]
mask:   [T_i]
```

The custom collate function pads a batch to the longest segment in the batch:

```text
images: [B, T_max, 3, 128, 128]
labels: [B, T_max, 1]
mask:   [B, T_max]
```

No frames are discarded.

Inspect the temporal dataset:

```bash
PYTHONPATH=. python tools/inspect_temporal_dataset.py \
  --pseudo-csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --batch-size 4 \
  --min-length 1 \
  --max-length 64 \
  --num-workers 0
```

Expected output:

```text
Dataset summary:
  num_segments: 4298
  num_frames: 102712
  min_length: 1
  max_length: 64
  mean_length: ~23.9

One padded batch:
  images: torch.Size([4, T_max, 3, 128, 128])
  labels: torch.Size([4, T_max, 1])
  mask: torch.Size([4, T_max])
```

Verify the temporal model definitions:

```bash
PYTHONPATH=. python models/gonet_temporal.py
```

Expected output:

```text
Device: cuda
Input: torch.Size([2, 11, 3, 128, 128])
Output prob: torch.Size([2, 11, 1])
Output logit: torch.Size([2, 11, 1])
```

---

## 18. Step 14 — Train GONet+T

GONet+T uses:

```text
frozen Generator
frozen InvG
frozen Discriminator
trainable feature reducer
trainable LSTM classifier
```

The architecture is:

```text
φR = |image_real - image_generated|               → Linear → 10D
φD = |feature_real - feature_generated|           → Linear → 10D
φF = discriminator feature from real image        → Linear → 10D

concat(φR, φD, φF) → 30D feature per frame
30D feature sequence → single-layer LSTM
LSTM hidden_dim = 64
linear output → sigmoid probability
```

The loss is:

```text
total_loss = λ_pred * prediction_loss + λ_smooth * temporal_smoothness_loss
```

Design decisions used in the best run:

```text
sequence handling: variable-length segments with padding/masks
max chunk length: 64 frames
feature reducer: 10D per feature group, 30D total
temporal model: single-layer LSTM
hidden dimension: 64
target mode: logit
λ_pred: 0.5
λ_smooth: 0.5
```

### Why train in logit space?

Directly training GONet+T on probabilities caused amplitude compression:

```text
high probabilities became slightly lower
low probabilities became slightly higher
```

This made the temporal output smooth but reduced confidence.

The better design is to train on logits:

```text
logit(p) = log(p / (1 - p))
```

The temporal model learns an unbounded confidence score and converts it back to probability with sigmoid at inference time. This preserved high/low confidence while still smoothing frame-to-frame noise.

Train the best logit-space GONet+T model:

```bash
PYTHONPATH=. python train_gonet_t.py \
  --pseudo-csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output-dir checkpoints/gonet_t_logit_lpred05_lsmooth05 \
  --epochs 10 \
  --batch-size 4 \
  --max-length 64 \
  --min-length 1 \
  --lr 1e-4 \
  --lambda-pred 0.5 \
  --lambda-smooth 0.5 \
  --target-mode logit \
  --device cuda
```

If interrupted, resume with:

```bash
PYTHONPATH=. python train_gonet_t.py \
  --pseudo-csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
    outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output-dir checkpoints/gonet_t_logit_lpred05_lsmooth05 \
  --epochs 10 \
  --batch-size 4 \
  --max-length 64 \
  --min-length 1 \
  --lr 1e-4 \
  --lambda-pred 0.5 \
  --lambda-smooth 0.5 \
  --target-mode logit \
  --device cuda \
  --resume checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt
```

The main checkpoint is:

```text
checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt
```

---

## 19. Step 15 — Compare vanilla GONet and GONet+T on selected segments

Use the comparison script to generate:

```text
comparison plot
annotated video
comparison CSV
```

Example for left-camera train segments:

```bash
PYTHONPATH=. python tools/compare_gonet_vs_gonet_t.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-dir outputs/gonet_t_compare_best_logit/train_unlabel_L \
  --segment-id 58 1918 \
  --threshold 0.85 \
  --fps 3 \
  --scale 4 \
  --feature-chunk-size 64 \
  --device cuda
```

Example for right-camera train segments:

```bash
PYTHONPATH=. python tools/compare_gonet_vs_gonet_t.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/train_unlabel_R_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-dir outputs/gonet_t_compare_best_logit/train_unlabel_R \
  --segment-id 1906 1680 \
  --threshold 0.85 \
  --fps 3 \
  --scale 4 \
  --feature-chunk-size 64 \
  --device cuda
```

Or generate top-k segment comparisons automatically:

```bash
PYTHONPATH=. python tools/compare_gonet_vs_gonet_t.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/train_unlabel_L_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-dir outputs/gonet_t_compare_best_logit/top_segments_L \
  --top-k 10 \
  --min-length 40 \
  --threshold 0.85 \
  --fps 3 \
  --scale 4 \
  --feature-chunk-size 64 \
  --device cuda
```

The generated video overlays:

```text
Vanilla GONet probability
GONet+T probability
GO / NO-GO decisions
threshold
building / segment / frame information
```

---

## 20. Step 16 — Run GONet+T on test and validation unlabelled data

The GONet+T model was trained on train unlabelled sequences only. For held-out unlabelled evaluation, build manifests and pseudo-labels for:

```text
whole_dataset/data_test/unlabel_L
whole_dataset/data_test/unlabel_R
whole_dataset/data_vali/unlabel_L
whole_dataset/data_vali/unlabel_R
```

### Build test manifests

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split test \
  --side L \
  --output-dir outputs/gonet_t_manifests
```

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split test \
  --side R \
  --output-dir outputs/gonet_t_manifests
```

### Build validation manifests

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split val \
  --side L \
  --output-dir outputs/gonet_t_manifests
```

```bash
PYTHONPATH=. python tools/build_unlabelled_sequence_manifest.py \
  --data-root $DATA_ROOT \
  --split val \
  --side R \
  --output-dir outputs/gonet_t_manifests
```

### Generate pseudo-labels for test and validation

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/test_unlabel_L_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/test_unlabel_L_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/test_unlabel_R_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/test_unlabel_R_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/val_unlabel_L_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/val_unlabel_L_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/pseudo_label_unlabelled_manifest.py \
  --manifest outputs/gonet_t_manifests/val_unlabel_R_manifest.csv \
  --checkpoint checkpoints/gonet_fl/fl_best.pt \
  --output outputs/gonet_t_pseudo_labels/val_unlabel_R_pseudo.csv \
  --batch-size 128 \
  --device cuda
```

### Run full-split GONet+T inference

This computes GONet+T probabilities for every unlabelled frame and writes one full comparison CSV per split/side.

```bash
PYTHONPATH=. python tools/run_gonet_t_full_inference.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/test_unlabel_L_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-csv outputs/gonet_t_full_inference/test_unlabel_L_full_comparison.csv \
  --threshold 0.85 \
  --feature-chunk-size 64 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/run_gonet_t_full_inference.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/test_unlabel_R_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-csv outputs/gonet_t_full_inference/test_unlabel_R_full_comparison.csv \
  --threshold 0.85 \
  --feature-chunk-size 64 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/run_gonet_t_full_inference.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/val_unlabel_L_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-csv outputs/gonet_t_full_inference/val_unlabel_L_full_comparison.csv \
  --threshold 0.85 \
  --feature-chunk-size 64 \
  --device cuda
```

```bash
PYTHONPATH=. python tools/run_gonet_t_full_inference.py \
  --pseudo-csv outputs/gonet_t_pseudo_labels/val_unlabel_R_pseudo.csv \
  --gonet-checkpoint checkpoints/gonet_fl/fl_best.pt \
  --gonet-t-checkpoint checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt \
  --output-csv outputs/gonet_t_full_inference/val_unlabel_R_full_comparison.csv \
  --threshold 0.85 \
  --feature-chunk-size 64 \
  --device cuda
```

Each full comparison CSV contains:

```text
segment_id
segment_local_index
building_id
frame_idx
side
filename
path
vanilla_prob
gonet_t_prob
threshold
vanilla_decision
gonet_t_decision
abs_difference
```

Each run also creates a summary CSV.

---

## 21. GONet+T results on GO Stanford unlabelled splits

These results are from the **GO Stanford dataset only**. No warehouse or external dataset was used.

The vanilla GONet model was first used to pseudo-label unlabelled GO Stanford frames. GONet+T was then evaluated as a temporal stabilizer against those vanilla GONet probabilities.

| Split | Frames | Segments | Mean vanilla | Mean GONet+T | Mean abs diff | Vanilla GO | GONet+T GO | Flip rate | Jitter reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test L | 9023 | 352 | 0.4724 | 0.4669 | 0.0203 | 2945 | 2847 | 2.26% | 8.91% |
| Test R | 9023 | 352 | 0.4830 | 0.4755 | 0.0209 | 3048 | 2901 | 2.49% | 9.36% |
| Val L | 8023 | 294 | 0.4181 | 0.4083 | 0.0207 | 2184 | 2043 | 2.53% | 9.15% |
| Val R | 8023 | 294 | 0.4299 | 0.4210 | 0.0208 | 2284 | 2113 | 3.05% | 9.26% |

Combined over all four unlabelled split/side combinations:

```text
Total frames evaluated: 34,092
Total temporal segments: 1,292
Mean absolute probability difference: ~0.021
Decision flip rate: ~2.6%
Jitter reduction: ~9.2%
```

Interpretation:

```text
GONet+T preserves the vanilla GONet probability scale while reducing frame-to-frame prediction jitter by about 9%.
```

GONet+T is slightly more conservative than vanilla GONet, because the number of GO frames is consistently lower after temporal smoothing. This is acceptable for a robotics traversability prototype because it reduces unstable GO decisions rather than making the system more aggressive.

Important limitation:

```text
These are not human-labelled accuracy numbers.
They measure temporal stabilization relative to vanilla GONet pseudo-labels on GO Stanford unlabelled sequences.
```

---

## 22. Recommended final checkpoints

For vanilla GONet:

```text
checkpoints/gonet_fl/fl_best.pt
```

For GONet+T:

```text
checkpoints/gonet_t_logit_lpred05_lsmooth05/gonet_t_latest.pt
```

For the GAN used by this run:

```text
checkpoints/gonet_gan/gan_epoch_0020.pt
```

For InvG:

```text
checkpoints/gonet_invg/invg_latest.pt
```

---

## 23. Optional — Remove `PYTHONPATH=.`

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
python train_gonet_t.py ...
python tools/infer_unlablelled_gs.py ...
```

without `PYTHONPATH=.`.

---

## 24. Troubleshooting

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

## 25. Data used by each training stage

| Stage | Dataset folders | Labels? | Purpose |
|---|---|---:|---|
| DCGAN | `whole_dataset/data_train/positive_L`, `positive_R` | No binary labels | Learn traversable image manifold |
| InvG | `whole_dataset/data_train/positive_L`, `positive_R` | No binary labels | Learn image → latent mapping |
| FL classifier | `hand_labelled_dataset/data_train_annotation/positive_*`, `negative_*` | Yes | Learn GO/NO-GO decision |
| Vanilla GONet evaluation | `hand_labelled_dataset/data_test_annotation/positive_*`, `negative_*` | Yes | Test image-level classifier performance |
| Vanilla GONet inference video | `whole_dataset/data_test/unlabel_L/R` | No | Visualize frame-wise predictions on unlabelled sequences |
| GONet+T training | `whole_dataset/data_train/unlabel_L/R` | Pseudo-labels from vanilla GONet | Learn temporal stabilization |
| GONet+T held-out analysis | `whole_dataset/data_test/unlabel_L/R`, `whole_dataset/data_vali/unlabel_L/R` | Pseudo-labels from vanilla GONet | Measure temporal smoothing on unlabelled sequences |

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

## 26. Limitations

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

The GONet+T extension adds temporal smoothing, but it is still based on pseudo-labels from vanilla GONet and is not a replacement for human-labelled temporal ground truth.

For warehouse AMRs, future extensions should include:

```text
path-aware inference
debris segmentation
uncertainty estimation
fusion with depth or projected planned path
evaluation on warehouse robot data
```

---

## 27. Citation / acknowledgement

This project is based on the GONet traversability-estimation idea and the GO Stanford dataset.

Please cite and acknowledge the original GONet authors and dataset source when using this repository publicly.