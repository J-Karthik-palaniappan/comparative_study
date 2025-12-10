import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torchvision.transforms as T

import os
import torch
import torch.nn.functional as F
import random
random.seed(42)

from utils import swinir_utils as util

def save_all_attention(model, save_dir="attn_mats"):
    os.makedirs(save_dir, exist_ok=True)
    
    for L, layer in enumerate(tqdm(model.layers)):
        # Some models might not have residual_group directly inside "layer"
        groups = (
            layer.residual_group.blocks
            if hasattr(layer.residual_group, "blocks")
            else layer.blocks
        )
        
        for G, block in enumerate(groups):
            # attn_dict = block.attn.capture   # {pre:..., mid:..., post:...}
            attn_dict = block.attn.attention.capture # debug

            for stage_name, attn_matrix in attn_dict.items():
                if attn_matrix is None:
                    continue

                # make sure it's on CPU
                attn_matrix = attn_matrix.detach().cpu()

                # split per head: [1, heads, tokens, tokens]
                B, H, T1, T2 = attn_matrix.shape
                
                for h in range(H):
                    head_mat = attn_matrix[0, h]    # [tokens, tokens]

                    # filename example: L0_G3_B0_pre_H1.npy
                    fname = f"L{L}_G{G}_{stage_name}_H{h}.npy"
                    fpath = os.path.join(save_dir, fname)

                    np.save(fpath, head_mat.numpy())

def pad(img, window_size):
    _, h_old, w_old = img.size()

    pad_h = (window_size - h_old % window_size) % window_size
    pad_w = (window_size - w_old % window_size) % window_size

    img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img

def tile_test(img_lq, model, args):
    sf = args.scale
    b, c, h, w = img_lq.size()
    tile = min(args.tile, h, w)
    assert tile % args.window_size == 0, "tile size should be a multiple of window_size"
    overlap = args.tile_overlap

    stride = tile - overlap
    h_starts = list(range(0, max(h - tile, 1), stride))
    w_starts = list(range(0, max(w - tile, 1), stride))

    # Ensure last tile reaches the boundary
    if h_starts[-1] != h - tile:
        h_starts.append(h - tile)
    if w_starts[-1] != w - tile:
        w_starts.append(w - tile)
    # Prepare accumulation & weight tensors
    E = torch.zeros(b, c, h * sf, w * sf, device=img_lq.device)
    W = torch.zeros_like(E)

    for hs in h_starts:
        for ws in w_starts:
            in_patch = img_lq[..., hs:hs + tile, ws:ws + tile]
            out_patch = model(in_patch)
            # Place output patch
            E[..., hs * sf:(hs + tile) * sf, ws * sf:(ws + tile) * sf].add_(out_patch)
            W[..., hs * sf:(hs + tile) * sf, ws * sf:(ws + tile) * sf].add_(1.0)
    output = E / W
    return output

def batch_test(dataset, model, save_dir):
    metrics = []
    for lq, gt, lq_pth, gt_pth in tqdm(dataset, desc="Processing"):
        inp = pad(lq, args.window_size)
        with torch.no_grad():
            if args.tile:
                out = tile_test(inp.unsqueeze(0).to(args.device), model, args)
            else:
                out = model(inp.unsqueeze(0).to(args.device))
        #unpad
        H, W = gt.shape[-2:]
        out = out[..., :H, :W]
        #save
        out_np = (out.squeeze().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        name = os.path.splitext(os.path.basename(gt_pth))[0]
        save_path = os.path.join(save_dir, f"{name}_{args.mech}.png")
        cv2.imwrite(save_path, out_np[:, :, ::-1])
        #compare
        gt_np = (gt.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        psnr = util.calculate_psnr(out_np, gt_np, crop_border=0)
        ssim = util.calculate_ssim(out_np, gt_np, crop_border=0)
        psnr_y = util.calculate_psnr(out_np, gt_np, crop_border=0, test_y_channel=True)
        ssim_y = util.calculate_ssim(out_np, gt_np, crop_border=0, test_y_channel=True)
        metrics.append([psnr, ssim, psnr_y, ssim_y])
    #report
    metrics = np.array(metrics)
    avg = metrics.mean(axis=0)
    print(avg)
    
if __name__ == "__main__":
    params = {
        "scale": 4,
        "model_path":"pretrained_models/SwinIR/x4.pth",
        "folder_gt": "datasets/SwinIR/Set5/HR",
        "folder_lq": "datasets/SwinIR/Set5/LR_bicubic/X4",
        "mech":"nystrom",
        "device":"cuda",
        "num_landmarks":16,
        "iters":2,
        "window_size":8,
        "tile":None,
        "debug":False
    }
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    dataset = util.DatasetSR(
        scale = args.scale,
        dataroot_L=args.folder_lq,
        dataroot_H=args.folder_gt,
        n_channels=3,
        patch_size=64,
        phase='test',
    )
    lq, gt, lq_pth, gt_pth = dataset[0]
    inp = pad(lq, args.window_size)

    print("Mech:", args.mech)
    model = util.define_model(args, img_size=inp.shape[-1])
    model = model.to(args.device)
    model.eval()

    save_dir = f"results/swinir_x{args.scale}"
    os.makedirs(save_dir, exist_ok=True)

    print("testing")
    if args.debug:
        out = model(inp.unsqueeze(0).to(args.device))
        # save_all_attention(model, save_dir=f"attn_mats/{args.mech}")

        out_np = (out.detach().squeeze().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        gt_np = (gt.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        psnr = util.calculate_psnr(out_np, gt_np, crop_border=0)
        ssim = util.calculate_ssim(out_np, gt_np, crop_border=0)
        print(psnr, ssim)

    else:
        batch_test(dataset, model, save_dir)