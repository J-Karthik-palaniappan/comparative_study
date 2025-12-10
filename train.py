import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import util
from dataloader import DatasetSR

def train_one_epoch(loader, model, criterion, optimizer, args):
    model.train()
    total_loss = 0
    for lq, gt, _, _ in tqdm(loader, desc="Training"):
        lq = lq.to(args.device)
        gt = gt.to(args.device)

        out = model(lq)
        loss = criterion(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def validate(dataset, model, args):
    model.eval()
    psnr_list, ssim_list, psnr_y_list, ssim_y_list = [], [], [], []
    with torch.no_grad():
        for lq, gt, _, _ in dataset:
            inp = lq.unsqueeze(0).to(args.device)
            out = model(inp)
            H, W = gt.shape[-2:]
            out = out[..., :H, :W]

            out_np = (out.squeeze().float().cpu()
                      .clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round()
            gt_np = (gt.numpy().transpose(1, 2, 0) * 255).round()

            psnr = util.calculate_psnr(out_np, gt_np, crop_border=0)
            ssim = util.calculate_ssim(out_np, gt_np, crop_border=0)
            psnr_y = util.calculate_psnr(out_np, gt_np, crop_border=0, test_y_channel=True)
            ssim_y = util.calculate_ssim(out_np, gt_np, crop_border=0, test_y_channel=True)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            psnr_y_list.append(psnr_y)
            ssim_y_list.append(ssim_y)

    return (
        sum(psnr_list)/len(psnr_list),
        sum(ssim_list)/len(ssim_list),
        sum(psnr_y_list)/len(psnr_y_list),
        sum(ssim_y_list)/len(ssim_y_list),
    )

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "loss", "psnr", "ssim", "psnr_y", "ssim_y"])
        writer.writerow(row)

if __name__ == "__main__":
    params = {
        "scale": 4,
        "train_gt": "datasets/DF2K/HR",
        "train_lq": "datasets/DF2K/LR_bicubic/X4",
        "val_gt": "datasets/SwinIR/Set5/HR",
        "val_lq": "datasets/SwinIR/Set5/LR_bicubic/X4",
        "patch_size": 64,
        "batch": 8,
        "epochs": 100,
        "lr": 2e-4,
        "window_size": 8,
        "device": "cuda",
        "save_dir": "train_logs"
    }

    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    train_set = DatasetSR(
        scale=args.scale,
        dataroot_L=args.train_lq,
        dataroot_H=args.train_gt,
        n_channels=3,
        patch_size=args.patch_size,
        phase='train',
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    val_set = DatasetSR(
        scale=args.scale,
        dataroot_L=args.val_lq,
        dataroot_H=args.val_gt,
        n_channels=3,
        patch_size=args.patch_size,
        phase='test',
    )

    lq, _, _, _ = val_set[0]
    model = util.define_model(args, img_size=lq.shape[-1])
    model = model.to(args.device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, "history.csv")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(train_loader, model, criterion, optimizer, args)
        psnr, ssim, psnr_y, ssim_y = validate(val_set, model, args)

        print(f"Epoch {epoch} | Loss {loss:.4f} | PSNR {psnr:.2f} | SSIM {ssim:.4f}")

        append_csv(history_path, [epoch, loss, psnr, ssim, psnr_y, ssim_y])

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
        save_checkpoint(model, ckpt_path)
