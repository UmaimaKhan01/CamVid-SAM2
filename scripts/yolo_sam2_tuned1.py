import os, cv2, numpy as np, torch
from glob import glob
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Paths (do NOT overwrite baseline)
# -----------------------------
IMG_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val"
OUT_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\outputs\masks_yolo_sam2_tuned1"
CKPT    = r"C:\Users\umaim\Downloads\Camvid_sam2\weights\sam2_hiera_small.pt"

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load YOLOv8 detector (COCO classes)
# -----------------------------
# COCO class ids: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
PEOPLE_CLS   = {0}
VEHICLE_CLS  = {1, 2, 3, 5, 7}
DET_CONF_THR = 0.35  # adjust later if needed
yolo = YOLO("yolov8n.pt")  # small & fast; you can try 'yolov8s.pt' later

# -----------------------------
# Load SAM2
# -----------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(
    config_file=r"C:\Users\umaim\Downloads\Camvid_sam2\sam2\sam2\configs\sam2\sam2_hiera_s.yaml",
    device=DEVICE,
    checkpoint=CKPT,
)
predictor = SAM2ImagePredictor(sam2_model)

def masks_from_boxes(image_bgr, boxes_xyxy):
    """Runs SAM2 with box prompts and returns a list of binary uint8 masks."""
    # BGR -> RGB, ensure contiguous
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    predictor.set_image(image_rgb)

    masks = []
    for box in boxes_xyxy:
        try:
            box = np.array(box, dtype=np.float32)
            pred_masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )
            mask = pred_masks[0]
            if mask.max() > 1.0:
                # logits => convert to probs
                mask = torch.sigmoid(torch.from_numpy(mask)).numpy()
            binary_mask = (mask > 0.5).astype(np.uint8)
            masks.append(np.ascontiguousarray(binary_mask))
        except Exception as e:
            h, w = image_bgr.shape[:2]
            masks.append(np.zeros((h, w), np.uint8))
    return masks

def combine_people_vehicles(image_bgr, det):
    """
    det: Ultralytics result object.
    Returns (people_mask, vehicles_mask) uint8 in {0,1}.
    """
    H, W = image_bgr.shape[:2]
    people = np.zeros((H, W), np.uint8)
    vehicles = np.zeros((H, W), np.uint8)

    if det.boxes is None or len(det.boxes) == 0:
        return people, vehicles

    # Gather boxes by category
    boxes = det.boxes.xyxy.cpu().numpy()
    confs = det.boxes.conf.cpu().numpy()
    clss  = det.boxes.cls.cpu().numpy().astype(int)

    people_boxes   = [b for b, c, cl in zip(boxes, confs, clss) if c >= DET_CONF_THR and cl in PEOPLE_CLS]
    vehicles_boxes = [b for b, c, cl in zip(boxes, confs, clss) if c >= DET_CONF_THR and cl in VEHICLE_CLS]

    # Run SAM2 per group
    if people_boxes:
        pmasks = masks_from_boxes(image_bgr, people_boxes)
        for m in pmasks:
            people |= m

    if vehicles_boxes:
        vmasks = masks_from_boxes(image_bgr, vehicles_boxes)
        for m in vmasks:
            vehicles |= m

    return people, vehicles

def main():
    images = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Found images: {len(images)} | device: {DEVICE}")

    for i, ip in enumerate(images, 1):
        img = cv2.imread(ip)
        if img is None:
            print("skip unreadable:", ip)
            continue

        # YOLO inference
        det_results = yolo.predict(source=img, verbose=False, device=0 if DEVICE=="cuda" else None)
        det = det_results[0]

        ppl, veh = combine_people_vehicles(img, det)

        stem = Path(ip).stem
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_people.png"), ppl * 255)
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_vehicles.png"), veh * 255)

        if i % 10 == 0:
            print(f"[{i}/{len(images)}] {stem} ✓")

    print("Done. YOLO→SAM2 masks at:", OUT_DIR)

if __name__ == "__main__":
    main()
