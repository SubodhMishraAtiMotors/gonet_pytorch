#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.go_stanford import GOStanfordPositiveDataset
from models.gonet import Generator, Discriminator, InvG, init_weights_normal
from torch.utils.tensorboard import SummaryWriter

def denorm_for_save(x: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor from approximately [-1, 1] to [0, 1].
    """
    return torch.clamp((x * 128.0 + 128.0) / 255.0, 0.0, 1.0)


def save_reconstruction_samples(
    generator,
    invg,
    dataloader,
    epoch,
    output_dir,
    device,
    max_images=8,
):
    generator.eval()
    invg.eval()

    batch = next(iter(dataloader))
    real = batch["image"][:max_images].to(device)

    with torch.no_grad():
        z_hat = invg(real)
        recon = generator(z_hat)

    real_vis = denorm_for_save(real)
    recon_vis = denorm_for_save(recon)

    # Stack as:
    # row 1: real images
    # row 2: reconstructed images
    comparison = torch.cat([real_vis, recon_vis], dim=0)

    save_path = output_dir / f"recon_epoch_{epoch:04d}.png"
    save_image(comparison, save_path, nrow=max_images)

    generator.train()
    invg.train()


def plot_losses(loss_history, output_dir):
    if len(loss_history["epoch"]) == 0:
        return

    plt.figure()
    plt.plot(loss_history["epoch"], loss_history["train_loss"], label="Train Loss")
    plt.plot(loss_history["epoch"], loss_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Joint Image and Feature reconstruction loss")
    plt.title("InvG Reconstruction Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "invg_loss_curve.png", dpi=150)
    plt.close()


def save_checkpoint(
    output_dir,
    epoch,
    generator,
    discriminator,
    invg,
    optimizer,
    args,
    loss_history,
    is_latest=True,
):
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "invg": invg.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "loss_history": loss_history,
    }

    if is_latest:
        path = output_dir / "invg_latest.pt"
    else:
        path = output_dir / f"invg_epoch_{epoch:04d}.pt"

    torch.save(checkpoint, path)


def evaluate(generator, discriminator, invg, dataloader, criterion_images, criterion_features, device, lambda_recon):
    # Ensure all the models are in eval mode
    generator.eval()
    invg.eval()
    discriminator.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            real = batch["image"].to(device, non_blocking=True)
            z_hat = invg(real)
            recon = generator(z_hat)
            real_feature = discriminator(real)
            recon_feature = discriminator(recon)
            loss = (1.0-lambda_recon)*criterion_images(recon, real) + lambda_recon*criterion_features(recon_feature, real_feature)

            bsz = real.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz
    # Only the inverse generator is being trained here.
    invg.train()

    return total_loss / max(1, total_count)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to go_stanford_dataset",
    )

    parser.add_argument(
        "--gan-checkpoint",
        required=True,
        help="Path to trained GAN checkpoint, e.g. checkpoints/gonet_gan/gan_epoch_0020.pt",
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/gonet_invg",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--lambda-recon",
        type=float,
        default=0.5,
        help="Weight balancing image reconstruction and feature reconstruction.",
    )

    parser.add_argument(
        "--use-tanh",
        action="store_true",
        help="Use only if GAN generator was trained with --use-tanh.",
    )

    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=1)

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to InvG checkpoint to resume from.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_dataset = GOStanfordPositiveDataset(
        root=args.data_root,
        split="train",
        side="both",
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    val_dataset = GOStanfordPositiveDataset(
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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")

    generator = Generator(nz=args.nz, use_tanh=args.use_tanh).to(device)
    discriminator = Discriminator().to(device)
    invg = InvG(nz=args.nz).to(device)

    invg.apply(init_weights_normal)

    gan_ckpt_path = Path(args.gan_checkpoint)
    if not gan_ckpt_path.exists():
        raise FileNotFoundError(f"GAN checkpoint not found: {gan_ckpt_path}")

    print(f"Loading generator and discriminator from: {gan_ckpt_path}")
    gan_checkpoint = torch.load(gan_ckpt_path, map_location=device)
    generator.load_state_dict(gan_checkpoint["generator"])
    discriminator.load_state_dict(gan_checkpoint["discriminator"])

    # Freeze generator
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False
    
    # Freeze discriminator
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad = False

    criterion_images = nn.L1Loss()
    criterion_features = nn.MSELoss()

    optimizer = torch.optim.Adam(
        invg.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    start_epoch = 1

    loss_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming InvG from: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device)

        invg.load_state_dict(resume_checkpoint["invg"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

        start_epoch = int(resume_checkpoint["epoch"]) + 1

        if "loss_history" in resume_checkpoint:
            loss_history = resume_checkpoint["loss_history"]

        print(f"Resuming from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=args.output_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        invg.train()
        generator.eval()
        discriminator.eval()
        
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            real = batch["image"].to(device, non_blocking=True)
            bsz = real.size(0)

            optimizer.zero_grad(set_to_none=True)

            z_hat = invg(real)
            recon = generator(z_hat)
            recon_loss = criterion_images(recon, real)
            real_features = discriminator(real)
            recon_features = discriminator(recon)
            feature_recon_loss = criterion_features(recon_features, real_features)
            loss = (1.0-args.lambda_recon)*recon_loss + args.lambda_recon*feature_recon_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bsz
            running_count += bsz

            pbar.set_postfix(
                {
                    "L1": f"{recon_loss.item():.5f}",
                }
            )

        train_loss = running_loss / max(1, running_count)
        val_loss= evaluate(
            generator=generator,
            discriminator=discriminator,
            invg=invg,
            dataloader=val_loader,
            criterion_images=criterion_images,
            criterion_features=criterion_features,
            device=device,
            lambda_recon = args.lambda_recon
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_L1={train_loss:.6f} "
            f"val_L1={val_loss:.6f}"
        )

        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            invg=invg,
            optimizer=optimizer,
            args=args,
            loss_history=loss_history,
            is_latest=True,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                invg=invg,
                optimizer=optimizer,
                args=args,
                loss_history=loss_history,
                is_latest=False,
            )

        if epoch % args.sample_every == 0:
            save_reconstruction_samples(
                generator=generator,
                invg=invg,
                dataloader=val_loader,
                epoch=epoch,
                output_dir=samples_dir,
                device=device,
                max_images=8,
            )

        plot_losses(loss_history, output_dir)
    writer.close()
    print()
    print(f"InvG training complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()