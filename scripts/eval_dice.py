import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import argparse
import csv

def binarize(path):
    m = cv2.imread(path, 0)
    if m is None:
        raise FileNotFoundError(f"Mask file not found: {path}")
    return (m > 127).astype(np.uint8)

def dice_iou(pred, gt):
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
    iou  = inter / (union + 1e-8)
    return float(dice), float(iou)

def main(gt_people_dir, gt_veh_dir, pred_dir, save_csv=None, verbose=False):
    pred_people_files  = glob(os.path.join(pred_dir, "*_people.png"))
    pred_vehicles_files = glob(os.path.join(pred_dir, "*_vehicles.png"))

    pred_people = {Path(p).stem.replace("_people",""): p for p in pred_people_files}
    pred_veh = {Path(p).stem.replace("_vehicles",""): p for p in pred_vehicles_files}

    gt_people = {Path(p).stem.replace("_L",""): p for p in glob(os.path.join(gt_people_dir, "*.png"))}
    gt_veh = {Path(p).stem.replace("_L",""): p for p in glob(os.path.join(gt_veh_dir, "*.png"))}

    stems = sorted(set(pred_people.keys()) & set(pred_veh.keys()) & set(gt_people.keys()) & set(gt_veh.keys()))

    if not stems:
        print("No matching prediction and ground truth masks found. Check paths and naming conventions.")
        print(f"Pred people: {len(pred_people_files)} files")
        print(f"Pred vehicles: {len(pred_vehicles_files)} files")
        print(f"GT people: {len(gt_people)} files")
        print(f"GT vehicles: {len(gt_veh)} files")
        return

    dice_p, iou_p, dice_v, iou_v = [], [], [], []
    per_image_results = []

    for idx, s in enumerate(stems, 1):
        pp = binarize(pred_people[s])
        pv = binarize(pred_veh[s])
        gp = binarize(gt_people[s])
        gv = binarize(gt_veh[s])

        d_p, i_p = dice_iou(pp, gp)
        d_v, i_v = dice_iou(pv, gv)

        dice_p.append(d_p)
        iou_p.append(i_p)
        dice_v.append(d_v)
        iou_v.append(i_v)

        per_image_results.append({
            "image": s,
            "dice_people": d_p,
            "iou_people": i_p,
            "dice_vehicles": d_v,
            "iou_vehicles": i_v,
        })

        if verbose:
            print(f"[{idx}/{len(stems)}] {s}: DiceP={d_p:.4f}, IoUP={i_p:.4f}, DiceV={d_v:.4f}, IoUV={i_v:.4f}")

    print(f"\nEvaluated {len(stems)} images")
    print(f"People   - Mean Dice: {np.mean(dice_p):.4f} | Mean IoU: {np.mean(iou_p):.4f}")
    print(f"Vehicles - Mean Dice: {np.mean(dice_v):.4f} | Mean IoU: {np.mean(iou_v):.4f}")
    print(f"Overall  - Mean Dice: {np.mean(dice_p + dice_v):.4f} | Mean IoU: {np.mean(iou_p + iou_v):.4f}")

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "dice_people", "iou_people", "dice_vehicles", "iou_vehicles"])
            writer.writeheader()
            writer.writerows(per_image_results)
        print(f"Saved per-image results to {save_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions with Dice and IoU")
    parser.add_argument("--gt_people_dir", required=True, help="Ground truth people masks directory")
    parser.add_argument("--gt_veh_dir", required=True, help="Ground truth vehicle masks directory")
    parser.add_argument("--pred_dir", required=True, help="Predicted masks directory")
    parser.add_argument("--save_csv", default=None, help="CSV path to save per-image results")
    parser.add_argument("--verbose", action="store_true", help="Print per-image scores")
    args = parser.parse_args()

    main(args.gt_people_dir, args.gt_veh_dir, args.pred_dir, args.save_csv, args.verbose)

