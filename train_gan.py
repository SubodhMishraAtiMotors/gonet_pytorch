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
from models.gonet import Generator, Discriminator, init_weights_normal


def denorm_for_save(x: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor from approximately [-1, 1] to [0, 1].
    """
    return torch.clamp((x * 128.0 + 128.0) / 255.0, 0.0, 1.0)


def save_samples(generator, fixed_z, epoch, output_dir, device):
    generator.eval()

    with torch.no_grad():
        fake = generator(fixed_z.to(device))
        fake = denorm_for_save(fake)

    save_path = output_dir / f"samples_epoch_{epoch:04d}.png"
    save_image(fake, save_path, nrow=8)

    generator.train()


def plot_losses(loss_history, output_dir):
    if len(loss_history["epoch"]) == 0:
        return

    plt.figure()
    plt.plot(loss_history["epoch"], loss_history["g_loss"], label="Generator loss")
    plt.plot(loss_history["epoch"], loss_history["d_loss"], label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "gan_loss_curve.png", dpi=150)
    plt.close()


def save_checkpoint(
    output_dir,
    epoch,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    args,
    loss_history,
    is_latest=True,
):
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "args": vars(args),
        "loss_history": loss_history,
    }

    if is_latest:
        path = output_dir / "gan_latest.pt"
    else:
        path = output_dir / f"gan_epoch_{epoch:04d}.pt"

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to go_stanford_dataset",
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/gonet_gan",
        help="Directory to save checkpoints and samples",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    # beta1 : remembers gradient direction
    # beta2 : remembers gradient size / variance
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.9,
        help="Real label value. 1.0 disables smoothing. 0.9 is common for GANs.",
    )

    parser.add_argument(
        "--use-tanh",
        action="store_true",
        help="Add tanh to generator output. Original GONet code does not use this explicitly.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save numbered checkpoint every N epochs.",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Save generated image samples every N epochs.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from, e.g. checkpoints/gonet_gan/gan_latest.pt",
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

    dataset = GOStanfordPositiveDataset(
        root=args.data_root,
        split="train",
        side="both",
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    print(f"Training images: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    generator = Generator(nz=args.nz, use_tanh=args.use_tanh).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(init_weights_normal)
    discriminator.apply(init_weights_normal)

    criterion = nn.CrossEntropyLoss()
    # g_t = ∇θ L_t
    # m_t = β1 m_{t-1} + (1 - β1) g_t
    # v_t = β2 v_{t-1} + (1 - β2) g_t²
    # m̂_t = m_t / (1 - β1ᵗ)
    # v̂_t = v_t / (1 - β2ᵗ)
    # θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    start_epoch = 1

    loss_history = {
        "epoch": [],
        "g_loss": [],
        "d_loss": [],
    }

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)

        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])

        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        start_epoch = int(checkpoint["epoch"]) + 1

        if "loss_history" in checkpoint:
            loss_history = checkpoint["loss_history"]

        print(f"Resuming from epoch {start_epoch}")

    fixed_z = torch.randn(64, args.nz, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            real_images = batch["image"].to(device, non_blocking=True)
            bsz = real_images.size(0)

            # ============================================================
            # Train discriminator
            # ============================================================
            optimizer_d.zero_grad(set_to_none=True)

            real_targets = torch.ones(bsz, dtype=torch.long, device=device)
            fake_targets = torch.zeros(bsz, dtype=torch.long, device=device)

            _, real_logits = discriminator(real_images, return_logits=True)

            z = torch.randn(bsz, args.nz, device=device)
            fake_images = generator(z)

            _, fake_logits = discriminator(fake_images.detach(), return_logits=True)

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            optimizer_d.step()

            # ============================================================
            # Train generator
            # ============================================================
            optimizer_g.zero_grad(set_to_none=True)

            z = torch.randn(bsz, args.nz, device=device)
            fake_images = generator(z)

            _, fake_logits_for_g = discriminator(fake_images, return_logits=True)

            # Generator wants fake images to be classified as real.
            g_targets = torch.ones(bsz, dtype=torch.long, device=device)
            g_loss = criterion(fake_logits_for_g, g_targets)

            g_loss.backward()
            optimizer_g.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "D": f"{d_loss.item():.4f}",
                    "G": f"{g_loss.item():.4f}",
                }
            )

        avg_d_loss = running_d_loss / max(1, num_batches)
        avg_g_loss = running_g_loss / max(1, num_batches)

        loss_history["epoch"].append(epoch)
        loss_history["d_loss"].append(avg_d_loss)
        loss_history["g_loss"].append(avg_g_loss)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"D_loss={avg_d_loss:.6f} "
            f"G_loss={avg_g_loss:.6f}"
        )

        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
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
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                args=args,
                loss_history=loss_history,
                is_latest=False,
            )

        if epoch % args.sample_every == 0:
            save_samples(
                generator=generator,
                fixed_z=fixed_z,
                epoch=epoch,
                output_dir=samples_dir,
                device=device,
            )

        plot_losses(loss_history, output_dir)

    print()
    print(f"Training complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()