#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from datasets.go_stanford import GOStanfordLabelledDataset
from models.gonet import Generator, InvG, Discriminator, GONetClassifier


def load_full_checkpoint(checkpoint_path, device, nz=100, use_tanh=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator = Generator(nz=nz, use_tanh=use_tanh).to(device)
    invg = InvG(nz=nz).to(device)
    discriminator = Discriminator().to(device)
    classifier = GONetClassifier().to(device)

    generator.load_state_dict(checkpoint["generator"])
    invg.load_state_dict(checkpoint["invg"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    classifier.load_state_dict(checkpoint["classifier"])

    generator.eval()
    invg.eval()
    discriminator.eval()
    classifier.eval()

    return generator, invg, discriminator, classifier, checkpoint


def run_inference(generator, invg, discriminator, classifier, dataloader, device):
    all_probs = []
    all_labels = []
    all_paths = []
    all_recon_l1 = []
    all_feature_l1 = []

    criterion_l1 = nn.L1Loss(reduction="none")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            paths = batch["path"]

            z_hat = invg(images)
            img_gen = generator(z_hat)

            dis_real = discriminator(images)
            dis_gen = discriminator(img_gen)

            probs = classifier(
                img_error=images - img_gen,
                dis_error=dis_real - dis_gen,
                dis_real=dis_real,
            )

            # Per-image reconstruction error
            recon_l1 = criterion_l1(img_gen, images)
            recon_l1 = recon_l1.view(recon_l1.size(0), -1).mean(dim=1)

            # Per-image discriminator feature error
            feature_l1 = criterion_l1(dis_gen, dis_real)
            feature_l1 = feature_l1.view(feature_l1.size(0), -1).mean(dim=1)

            all_probs.extend(probs.detach().cpu().view(-1).tolist())
            all_labels.extend(labels.detach().cpu().view(-1).tolist())
            all_paths.extend(list(paths))
            all_recon_l1.extend(recon_l1.detach().cpu().view(-1).tolist())
            all_feature_l1.extend(feature_l1.detach().cpu().view(-1).tolist())

    return {
        "probs": all_probs,
        "labels": all_labels,
        "paths": all_paths,
        "recon_l1": all_recon_l1,
        "feature_l1": all_feature_l1,
    }


def compute_metrics(labels, probs, threshold):
    y_true = [int(x) for x in labels]
    y_prob = probs
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision_positive_traversable": precision,
        "recall_positive_traversable": recall,
        "f1_positive_traversable": f1,
        "roc_auc": auc,
        "average_precision": ap,
        "tn_negative_correct": int(tn),
        "fp_negative_predicted_positive": int(fp),
        "fn_positive_predicted_negative": int(fn),
        "tp_positive_correct": int(tp),
        "confusion_matrix_labels_0_negative_1_positive": cm.tolist(),
    }


def save_predictions_csv(results, threshold, output_dir):
    csv_path = output_dir / "predictions.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "path",
            "label",
            "prob_traversable",
            "prediction",
            "recon_l1",
            "feature_l1",
        ])

        for path, label, prob, recon_l1, feature_l1 in zip(
            results["paths"],
            results["labels"],
            results["probs"],
            results["recon_l1"],
            results["feature_l1"],
        ):
            pred = 1 if prob >= threshold else 0

            writer.writerow([
                path,
                int(label),
                float(prob),
                int(pred),
                float(recon_l1),
                float(feature_l1),
            ])

    return csv_path


def plot_roc(labels, probs, output_dir):
    y_true = [int(x) for x in labels]

    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
    except ValueError:
        return

    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()


def plot_pr(labels, probs, output_dir):
    y_true = [int(x) for x in labels]

    try:
        precision, recall, _ = precision_recall_curve(y_true, probs)
    except ValueError:
        return

    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()


def plot_score_histogram(labels, probs, output_dir):
    y_true = [int(x) for x in labels]

    pos_probs = [p for p, y in zip(probs, y_true) if y == 1]
    neg_probs = [p for p, y in zip(probs, y_true) if y == 0]

    plt.figure()
    plt.hist(neg_probs, bins=30, alpha=0.6, label="Negative / no-go")
    plt.hist(pos_probs, bins=30, alpha=0.6, label="Positive / go")
    plt.xlabel("Predicted traversability probability")
    plt.ylabel("Count")
    plt.title("Prediction Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "score_histogram.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to fl_best.pt or fl_latest.pt",
    )
    parser.add_argument("--output-dir", default="outputs/gonet_eval")

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "vali", "validation", "test"],
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    dataset = GOStanfordLabelledDataset(
        root=args.data_root,
        split=args.split,
        side="both",
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print(f"Loaded {len(dataset)} labelled images from split: {args.split}")

    generator, invg, discriminator, classifier, checkpoint = load_full_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    results = run_inference(
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        classifier=classifier,
        dataloader=dataloader,
        device=device,
    )

    metrics = compute_metrics(
        labels=results["labels"],
        probs=results["probs"],
        threshold=args.threshold,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = save_predictions_csv(
        results=results,
        threshold=args.threshold,
        output_dir=output_dir,
    )

    plot_roc(results["labels"], results["probs"], output_dir)
    plot_pr(results["labels"], results["probs"], output_dir)
    plot_score_histogram(results["labels"], results["probs"], output_dir)

    print()
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    print()
    print(f"Saved predictions to: {csv_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()