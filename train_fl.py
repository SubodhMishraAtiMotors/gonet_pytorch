#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from datasets.go_stanford import GOStanfordLabelledDataset
from models.gonet import Generator, InvG, Discriminator, GONetClassifier, init_weights_normal


def plot_losses(history, output_dir):
    if len(history["epoch"]) == 0:
        return

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train BCE")
    plt.plot(history["epoch"], history["val_loss"], label="Val BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GONet Classifier Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "fl_loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["val_accuracy"], label="Val Accuracy")
    plt.plot(history["epoch"], history["val_f1"], label="Val F1")
    plt.plot(history["epoch"], history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("GONet Classifier Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "fl_metrics_curve.png", dpi=150)
    plt.close()


def load_generator(generator, gan_checkpoint_path, device):
    ckpt = torch.load(gan_checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    return generator


def load_invg(invg, invg_checkpoint_path, device):
    ckpt = torch.load(invg_checkpoint_path, map_location=device)
    invg.load_state_dict(ckpt["invg"])
    return invg


def load_discriminator(discriminator, gan_checkpoint_path, device):
    ckpt = torch.load(gan_checkpoint_path, map_location=device)
    discriminator.load_state_dict(ckpt["discriminator"])
    return discriminator


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def forward_gonet_features(generator, invg, discriminator, classifier, img_real):
    with torch.no_grad():
        z_hat = invg(img_real)
        img_gen = generator(z_hat)

        dis_real = discriminator(img_real)
        dis_gen = discriminator(img_gen)

        img_error = img_real - img_gen
        dis_error = dis_real - dis_gen

    prob = classifier(
        img_error=img_error,
        dis_error=dis_error,
        dis_real=dis_real,
    )

    return prob


def evaluate(generator, invg, discriminator, classifier, dataloader, criterion, device, threshold=0.5):
    generator.eval()
    invg.eval()
    discriminator.eval()
    classifier.eval()

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            z_hat = invg(images)
            img_gen = generator(z_hat)

            dis_real = discriminator(images)
            dis_gen = discriminator(img_gen)

            probs = classifier(
                img_error=images - img_gen,
                dis_error=dis_real - dis_gen,
                dis_real=dis_real,
            )

            loss = criterion(probs, labels)

            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz

            all_probs.extend(probs.detach().cpu().view(-1).tolist())
            all_labels.extend(labels.detach().cpu().view(-1).tolist())

    avg_loss = total_loss / max(1, total_count)

    y_true = [int(x) for x in all_labels]
    y_prob = all_probs
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def save_checkpoint(output_dir, epoch, generator, invg, discriminator, classifier, optimizer, args, history, name):
    ckpt = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "invg": invg.state_dict(),
        "discriminator": discriminator.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
    }

    torch.save(ckpt, output_dir / name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gan-checkpoint", required=True)
    parser.add_argument("--invg-checkpoint", required=True)
    parser.add_argument("--output-dir", default="checkpoints/gonet_fl")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--use-tanh",
        action="store_true",
        help="Use only if GAN was trained with --use-tanh.",
    )

    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_dataset = GOStanfordLabelledDataset(
        root=args.data_root,
        split="train",
        side="both",
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    val_dataset = GOStanfordLabelledDataset(
        root=args.data_root,
        split="val",
        side="both",
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print(f"Train labelled images: {len(train_dataset)}")
    print(f"Val labelled images:   {len(val_dataset)}")

    generator = Generator(nz=args.nz, use_tanh=args.use_tanh).to(device)
    invg = InvG(nz=args.nz).to(device)
    discriminator = Discriminator().to(device)
    classifier = GONetClassifier().to(device)

    classifier.apply(init_weights_normal)

    print(f"Loading GAN checkpoint:  {args.gan_checkpoint}")
    generator = load_generator(generator, args.gan_checkpoint, device)
    discriminator = load_discriminator(discriminator, args.gan_checkpoint, device)

    print(f"Loading InvG checkpoint: {args.invg_checkpoint}")
    invg = load_invg(invg, args.invg_checkpoint, device)

    freeze_module(generator)
    freeze_module(invg)
    freeze_module(discriminator)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
    }

    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        generator.eval()
        invg.eval()
        discriminator.eval()

        running_loss = 0.0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            probs = forward_gonet_features(
                generator=generator,
                invg=invg,
                discriminator=discriminator,
                classifier=classifier,
                img_real=images,
            )

            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            bsz = images.size(0)
            running_loss += loss.item() * bsz
            running_count += bsz

            pbar.set_postfix({"BCE": f"{loss.item():.5f}"})

        train_loss = running_loss / max(1, running_count)

        val_metrics = evaluate(
            generator=generator,
            invg=invg,
            discriminator=discriminator,
            classifier=classifier,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        print("Confusion matrix labels=[0 negative, 1 positive]:")
        print(val_metrics["confusion_matrix"])

        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            generator=generator,
            invg=invg,
            discriminator=discriminator,
            classifier=classifier,
            optimizer=optimizer,
            args=args,
            history=history,
            name="fl_latest.pt",
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                generator=generator,
                invg=invg,
                discriminator=discriminator,
                classifier=classifier,
                optimizer=optimizer,
                args=args,
                history=history,
                name="fl_best.pt",
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                generator=generator,
                invg=invg,
                discriminator=discriminator,
                classifier=classifier,
                optimizer=optimizer,
                args=args,
                history=history,
                name=f"fl_epoch_{epoch:04d}.pt",
            )

        plot_losses(history, output_dir)

    print()
    print(f"Training complete. Outputs saved to: {output_dir}")
    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()