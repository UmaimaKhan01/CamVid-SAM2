import os, cv2, numpy as np, torch
from glob import glob
from pathlib import Path
from torchvision.ops import nms
from ultralytics import YOLO

# --- YOUR PATHS ---
IMG_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val"
OUT_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\outputs\masks_yolo_sam2"
CKPT    = r"C:\Users\umaim\Downloads\Camvid_sam2\weights\sam2_hiera_small.pt"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COCO class ids we care about
COCO_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_KEYWORDS = {"car", "bus", "truck", "bicycle", "motorcycle"}

# --- SAM2 predictor ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(
    config_file=r"C:\Users\umaim\Downloads\Camvid_sam2\sam2\sam2\configs\sam2\sam2_hiera_s.yaml",
    checkpoint=CKPT,
    device=DEVICE,
)
predictor = SAM2ImagePredictor(sam2_model)

# --- YOLO detector (pretrained on COCO) ---
detector = YOLO("yolov8n.pt")  # use 'yolov8m.pt' if you want stronger/slower

def detect_boxes_yolo(img_bgr, conf_thr=0.25, nms_iou=0.5, min_area=600, max_rel_area=0.4):
    """
    Returns: boxes_xyxy [N,4], labels [N], scores [N]
    - keeps only target COCO classes
    - score filter, area filter, giant-box guard, NMS
    """
    H, W = img_bgr.shape[:2]
    res = detector.predict(source=img_bgr[..., ::-1], imgsz=max(H, W), conf=conf_thr, verbose=False, device=0 if DEVICE=="cuda" else None)
    if len(res) == 0:
        return np.zeros((0,4), float), [], np.zeros((0,), float)

    # Collect all boxes
    boxes_list, scores_list, labels_list = [], [], []
    r = res[0]
    if r.boxes is None or r.boxes.data is None:
        return np.zeros((0,4), float), [], np.zeros((0,), float)

    xyxy = r.boxes.xyxy.cpu().numpy()          # (N,4)
    conf = r.boxes.conf.cpu().numpy()          # (N,)
    cls  = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

    for b, s, c in zip(xyxy, conf, cls):
        if c in COCO_TO_NAME:
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(COCO_TO_NAME[c])

    if not boxes_list:
        return np.zeros((0,4), float), [], np.zeros((0,), float)

    boxes  = torch.tensor(np.stack(boxes_list, axis=0), dtype=torch.float32, device=DEVICE)
    scores = torch.tensor(np.array(scores_list), dtype=torch.float32, device=DEVICE)

    # clip
    boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, W-1)
    boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, H-1)

    # area filters
    wh = (boxes[:,2]-boxes[:,0]).clamp(min=0) * (boxes[:,3]-boxes[:,1]).clamp(min=0)
    keep = wh >= float(min_area)
    boxes, scores = boxes[keep], scores[keep]
    labels_list = [l for l, k in zip(labels_list, keep.tolist()) if k]
    if boxes.numel() == 0:
        return np.zeros((0,4), float), [], np.zeros((0,), float)

    big_keep = (wh[keep] / float(H*W)) <= float(max_rel_area)
    boxes, scores = boxes[big_keep], scores[big_keep]
    labels_list = [l for l, k in zip(labels_list, big_keep.tolist()) if k]
    if boxes.numel() == 0:
        return np.zeros((0,4), float), [], np.zeros((0,), float)

    # NMS
    keep_idx = nms(boxes, scores, float(nms_iou))
    boxes, scores = boxes[keep_idx], scores[keep_idx]
    labels = [labels_list[i] for i in keep_idx.tolist()]

    return boxes.detach().cpu().numpy().astype(float), labels, scores.detach().cpu().numpy().astype(float)

def masks_from_boxes(image_bgr, boxes_xyxy, thr=0.7):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    predictor.set_image(image_rgb)

    masks = []
    kernel = np.ones((3,3), np.uint8)
    for box in boxes_xyxy:
        box = [float(v) for v in box]
        pred_masks, _, _ = predictor.predict(box=box, multimask_output=False)
        m = pred_masks[0]
        if m.max() > 1.0:  # logits → probs
            m = torch.sigmoid(torch.from_numpy(m)).numpy()
        m = (m > float(thr)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        masks.append(m)
    return masks

def merge_people_vehicles(labels, masks):
    if not masks:
        return None, None
    H, W = masks[0].shape
    people   = np.zeros((H, W), np.uint8)
    vehicles = np.zeros((H, W), np.uint8)
    for lab, m in zip(labels, masks):
        lab = lab.lower().strip().strip(".")
        if lab == "person" or "pedestrian" in lab:
            people |= m
        elif lab in VEHICLE_KEYWORDS:
            vehicles |= m
    return people, vehicles

def save_debug(img_bgr, boxes, labels, path):
    vis = img_bgr.copy()
    for b, t in zip(boxes, labels):
        x1,y1,x2,y2 = [int(round(v)) for v in b]
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, t, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.imwrite(path, vis)

# --- main ---
images = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"Found images: {len(images)} | device: {DEVICE}")

DBG_DIR = os.path.join(Path(OUT_DIR).parent, "debug_yolo")
os.makedirs(DBG_DIR, exist_ok=True)

for i, ip in enumerate(images, 1):
    img = cv2.imread(ip)
    if img is None:
        print("skip unreadable:", ip); continue

    boxes, labels, scores = detect_boxes_yolo(img, conf_thr=0.35, nms_iou=0.5, min_area=600, max_rel_area=0.35)
    if i <= 6:
        save_debug(img, boxes, labels, os.path.join(DBG_DIR, f"{Path(ip).stem}.png"))

    if len(boxes) == 0:
        h, w = img.shape[:2]
        ppl = np.zeros((h, w), np.uint8)
        veh = np.zeros((h, w), np.uint8)
    else:
        det_masks = masks_from_boxes(img, boxes, thr=0.7)
        ppl, veh  = merge_people_vehicles(labels, det_masks)
        if ppl is None: ppl = np.zeros_like(det_masks[0])
        if veh is None: veh = np.zeros_like(det_masks[0])

    stem = Path(ip).stem
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_people.png"),   (ppl*255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_vehicles.png"), (veh*255).astype(np.uint8))

    if i % 10 == 0:
        print(f"[{i}/{len(images)}] {stem} ✓")

print("Done. Masks at:", OUT_DIR)
print("Debug overlays (first few frames) at:", DBG_DIR)
