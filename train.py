"""
Fine-tune LiteRIFE for satellite frame interpolation on a single Colab T4.

Usage (from a Colab cell, after mounting Drive):
    !python train.py \
        --frames-dir /content/data/frames \
        --checkpoint-dir /content/drive/MyDrive/sat_interp_ckpts \
        --epochs 40 --batch-size 8 --patch-size 256

Design choices for T4 (16 GB) feasibility:
    - Mixed precision (torch.cuda.amp) roughly doubles usable batch size.
    - Gradient accumulation lets you simulate a larger effective batch
      without more memory.
    - Checkpoints save to Google Drive every `--save-every` epochs so a
      Colab disconnect doesn't lose progress -- resume with --resume.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from model import LiteRIFE
from data_loader import make_loaders


def ssim_loss(pred, target, window=7):
    """Lightweight single-scale SSIM-based loss (1 - SSIM), avoids pulling
    in an extra dependency for a metric this simple."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.avg_pool2d(pred, window, 1, window // 2)
    mu_t = F.avg_pool2d(target, window, 1, window // 2)
    sigma_p = F.avg_pool2d(pred * pred, window, 1, window // 2) - mu_p ** 2
    sigma_t = F.avg_pool2d(target * target, window, 1, window // 2) - mu_t ** 2
    sigma_pt = F.avg_pool2d(pred * target, window, 1, window // 2) - mu_p * mu_t
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
    )
    return 1 - ssim_map.mean()


def compute_loss(pred, target, l1_weight=1.0, ssim_weight=0.5):
    l1 = F.l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return l1_weight * l1 + ssim_weight * ssim, l1.item(), ssim.item()


def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    train_loader, val_loader = make_loaders(
        args.frames_dir, patch_size=args.patch_size,
        batch_size=args.batch_size, val_frac=args.val_frac,
        num_workers=args.num_workers,
    )
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}", flush=True)

    model = LiteRIFE(in_channels=1).to(device)
    start_epoch = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler(device, enabled=(device == "cuda"))

    # Prime optimizer step count so the LR scheduler doesn't fire a
    # false-positive warning. GradScaler skips the real optimizer.step()
    # on early AMP iterations (inf gradients), so without this the scheduler
    # sees step_count==0 and warns incorrectly.
    optimizer.step()
    optimizer.zero_grad()

    accum_steps = max(1, args.effective_batch // args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        stepped_this_epoch = False
        for step, batch in enumerate(train_loader):
            I0 = batch["I0"].to(device, non_blocking=True)
            I1 = batch["I1"].to(device, non_blocking=True)
            It = batch["It"].to(device, non_blocking=True)

            with autocast(device, enabled=(device == "cuda")):
                pred, flow, mask = model(I0, I1, t=0.5)
                loss, l1, ssim = compute_loss(pred, It)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                stepped_this_epoch = True

            running_loss += loss.item() * accum_steps

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item()*accum_steps:.4f} l1={l1:.4f} ssim_loss={ssim:.4f}", flush=True)

        # Flush any leftover accumulated gradients (when batches % accum_steps != 0)
        leftover = len(train_loader) % accum_steps
        if leftover != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            stepped_this_epoch = True

        if stepped_this_epoch:
            scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"== epoch {epoch} done in {time.time()-t0:.1f}s  avg_train_loss={avg_loss:.4f} ==", flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_psnr = evaluate(model, val_loader, device)
            print(f"== epoch {epoch} val_psnr={val_psnr:.2f} dB ==", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"liteRIFE_epoch{epoch}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}", flush=True)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_psnr, n = 0.0, 0
    for batch in val_loader:
        I0 = batch["I0"].to(device)
        I1 = batch["I1"].to(device)
        It = batch["It"].to(device)
        pred, _, _ = model(I0, I1, t=0.5)
        total_psnr += psnr(pred, It)
        n += 1
    model.train()
    return total_psnr / max(1, n)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--resume", default="")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--effective-batch", type=int, default=16,
                    help="simulated batch size via gradient accumulation")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()
    train(args)
