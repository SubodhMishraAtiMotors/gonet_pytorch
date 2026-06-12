#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.go_stanford_temporal import (
    GOStanfordTemporalPseudoDataset,
    temporal_pseudo_collate_fn,
    summarize_temporal_dataset,
)

from models.gonet import Generator, InvG, Discriminator
from models.gonet_temporal import (
    GONetTemporalFeatureReducer,
    GONetTemporalClassifier,
    GONetTemporalFull,
    masked_prediction_loss,
    masked_smoothness_loss,
    init_temporal_weights,
    prob_to_logit,
)


def load_backbone_from_gonet_checkpoint(checkpoint_path, device, nz=100, use_tanh=False):
    ckpt = torch.load(checkpoint_path, map_location=device)

    generator = Generator(nz=nz, use_tanh=use_tanh).to(device)
    invg = InvG(nz=nz).to(device)
    discriminator = Discriminator().to(device)

    generator.load_state_dict(ckpt["generator"])
    invg.load_state_dict(ckpt["invg"])
    discriminator.load_state_dict(ckpt["discriminator"])

    generator.eval()
    invg.eval()
    discriminator.eval()

    return generator, invg, discriminator


def plot_history(history, output_dir):
    if len(history["epoch"]) == 0:
        return

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train total")
    plt.plot(history["epoch"], history["train_pred_loss"], label="Train prediction")
    plt.plot(history["epoch"], history["train_smooth_loss"], label="Train smoothness")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GONet+T Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "gonet_t_train_loss.png", dpi=150)
    plt.close()


def save_checkpoint(
    output_dir,
    name,
    epoch,
    model,
    optimizer,
    args,
    history,
):
    ckpt = {
        "epoch": epoch,
        "feature_reducer": model.feature_reducer.state_dict(),
        "temporal_classifier": model.temporal_classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
    }

    torch.save(ckpt, output_dir / name)


def load_resume_checkpoint(resume_path, model, optimizer, device):
    resume_path = Path(resume_path)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    print(f"Resuming GONet+T from: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)

    model.feature_reducer.load_state_dict(ckpt["feature_reducer"])
    model.temporal_classifier.load_state_dict(ckpt["temporal_classifier"])

    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    start_epoch = int(ckpt["epoch"]) + 1

    history = ckpt.get(
        "history",
        {
            "epoch": [],
            "train_loss": [],
            "train_pred_loss": [],
            "train_smooth_loss": [],
        },
    )

    print(f"Resume checkpoint epoch: {ckpt['epoch']}")
    print(f"Starting from epoch: {start_epoch}")

    return start_epoch, history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        nargs="+",
        required=True,
        help="Pseudo-label CSVs for temporal training.",
    )

    parser.add_argument(
        "--gonet-checkpoint",
        required=True,
        help="Vanilla GONet checkpoint, e.g. checkpoints/gonet_fl/fl_best.pt",
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/gonet_t",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")

    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=1)

    parser.add_argument("--reduced-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--lambda-pred",
        type=float,
        default=0.8,
        help="Weight for prediction loss.",
    )

    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.2,
        help="Weight for temporal smoothness loss.",
    )

    parser.add_argument(
        "--prediction-loss",
        default="mse",
        choices=["mse", "l1"],
    )

    parser.add_argument(
        "--smoothness-loss",
        default="mse",
        choices=["mse", "l1"],
    )

    parser.add_argument(
        "--target-mode",
        default="prob",
        choices=["prob", "logit"],
        help="Train GONet+T to match probabilities or logits of vanilla GONet.",
    )

    parser.add_argument(
        "--logit-eps",
        type=float,
        default=1e-4,
        help="Clamp epsilon used for probability-to-logit conversion.",
    )

    parser.add_argument("--save-every", type=int, default=5)

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to GONet+T checkpoint to resume from.",
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Target mode: {args.target_mode}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    dataset = GOStanfordTemporalPseudoDataset(
        pseudo_csvs=args.pseudo_csv,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    summary = summarize_temporal_dataset(dataset)

    print("Temporal dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=temporal_pseudo_collate_fn,
        pin_memory=(device == "cuda"),
    )

    print(f"Batches per epoch: {len(loader)}")

    print(f"Loading vanilla GONet backbone from: {args.gonet_checkpoint}")

    generator, invg, discriminator = load_backbone_from_gonet_checkpoint(
        checkpoint_path=args.gonet_checkpoint,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    feature_reducer = GONetTemporalFeatureReducer(
        reduced_dim=args.reduced_dim,
    ).to(device)

    temporal_classifier = GONetTemporalClassifier(
        input_dim=3 * args.reduced_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    feature_reducer.apply(init_temporal_weights)
    temporal_classifier.apply(init_temporal_weights)

    model = GONetTemporalFull(
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        feature_reducer=feature_reducer,
        temporal_classifier=temporal_classifier,
    ).to(device)

    model.freeze_gonet_backbone()

    trainable_params = list(model.feature_reducer.parameters()) + list(
        model.temporal_classifier.parameters()
    )

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_pred_loss": [],
        "train_smooth_loss": [],
    }

    start_epoch = 1

    if args.resume is not None:
        start_epoch, history = load_resume_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    if start_epoch > args.epochs:
        print()
        print(
            f"Nothing to train: resume starts at epoch {start_epoch}, "
            f"but --epochs is {args.epochs}."
        )
        print("Increase --epochs if you want to continue training.")
        return

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            model.freeze_gonet_backbone()

            running_total = 0.0
            running_pred = 0.0
            running_smooth = 0.0
            running_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

            for batch in pbar:
                images = batch["images"].to(device, non_blocking=True)
                labels_prob = batch["labels"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if args.target_mode == "prob":
                    preds = model(
                        images,
                        lengths=lengths,
                        return_logits=False,
                    )
                    targets = labels_prob

                elif args.target_mode == "logit":
                    preds = model(
                        images,
                        lengths=lengths,
                        return_logits=True,
                    )
                    targets = prob_to_logit(
                        labels_prob,
                        eps=args.logit_eps,
                    )

                else:
                    raise ValueError(f"Unsupported target mode: {args.target_mode}")

                pred_loss = masked_prediction_loss(
                    preds=preds,
                    targets=targets,
                    mask=mask,
                    loss_type=args.prediction_loss,
                )

                smooth_loss = masked_smoothness_loss(
                    preds=preds,
                    mask=mask,
                    loss_type=args.smoothness_loss,
                )

                total_loss = (
                    args.lambda_pred * pred_loss
                    + args.lambda_smooth * smooth_loss
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)

                optimizer.step()

                running_total += total_loss.item()
                running_pred += pred_loss.item()
                running_smooth += smooth_loss.item()
                running_batches += 1

                pbar.set_postfix(
                    {
                        "total": f"{total_loss.item():.5f}",
                        "pred": f"{pred_loss.item():.5f}",
                        "smooth": f"{smooth_loss.item():.5f}",
                    }
                )

            avg_total = running_total / max(1, running_batches)
            avg_pred = running_pred / max(1, running_batches)
            avg_smooth = running_smooth / max(1, running_batches)

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_total)
            history["train_pred_loss"].append(avg_pred)
            history["train_smooth_loss"].append(avg_smooth)

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"loss={avg_total:.6f} "
                f"pred={avg_pred:.6f} "
                f"smooth={avg_smooth:.6f}"
            )

            save_checkpoint(
                output_dir=output_dir,
                name="gonet_t_latest.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                history=history,
            )

            if epoch % args.save_every == 0:
                save_checkpoint(
                    output_dir=output_dir,
                    name=f"gonet_t_epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    history=history,
                )

            plot_history(history, output_dir)

    except KeyboardInterrupt:
        print()
        print("Training interrupted by user. Saving interrupt checkpoint...")

        interrupted_epoch = history["epoch"][-1] if len(history["epoch"]) > 0 else start_epoch - 1

        save_checkpoint(
            output_dir=output_dir,
            name="gonet_t_interrupted.pt",
            epoch=interrupted_epoch,
            model=model,
            optimizer=optimizer,
            args=args,
            history=history,
        )

        print(f"Saved: {output_dir / 'gonet_t_interrupted.pt'}")
        return

    print()
    print(f"GONet+T training complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()