# import os, cv2, numpy as np, torch
# from glob import glob
# from pathlib import Path

# IMG_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val"
# OUT_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\outputs\masks_baseline"
# CKPT    = r"C:\Users\umaim\Downloads\Camvid_sam2\weights\sam2_hiera_small.pt"
# os.makedirs(OUT_DIR, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # person + vehicle prompts (matches how you built GT)
# TEXT_QUERIES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]

# # --- GroundingDINO (HuggingFace) for text->boxes ---
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# gd_id = "IDEA-Research/grounding-dino-base"
# processor = AutoProcessor.from_pretrained(gd_id)
# gd_model  = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(DEVICE).eval()

# # --- SAM2 image predictor (Hydra config key, not a path) ---
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# sam2_model = build_sam2(
#     config_file=r"C:\Users\umaim\Downloads\Camvid_sam2\sam2\sam2\configs\sam2\sam2_hiera_s.yaml",  # absolute YAML path
#     device=DEVICE,
#     checkpoint=CKPT,
# )
# predictor = SAM2ImagePredictor(sam2_model)

# def grounding_dino_boxes(image_bgr, texts, box_thr=0.35, text_thr=0.25):
#     """
#     Returns (boxes_xyxy, labels).
#     Uses HF GroundingDINO post_process WITHOUT threshold args, then filters by score >= box_thr.
#     'text_thr' kept for API symmetry (if your processor returns per-phrase probs, you could use it),
#     but most builds expose only per-box 'scores' here.
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     h, w = image_rgb.shape[:2]

#     # GroundingDINO likes period-separated phrases
#     prompt = ". ".join(texts) + "."
#     inputs = processor(images=image_rgb, text=prompt, return_tensors="pt")
#     inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = gd_model(**inputs)

#     target_sizes = torch.tensor([[h, w]], device=DEVICE)

#     # No threshold parameters here; your version only accepts outputs, input_ids, target_sizes.
#     results = processor.post_process_grounded_object_detection(
#         outputs=outputs,
#         input_ids=inputs["input_ids"],
#         target_sizes=target_sizes,
#     )

#     if not results or len(results[0].get("boxes", [])) == 0:
#         return np.zeros((0, 4), dtype=float), []

#     res0 = results[0]
#     boxes_all = res0["boxes"].detach().cpu().numpy().astype(float)   # xyxy in pixels
#     labels_all = [str(s) for s in res0["labels"]]
#     # Some processor versions expose 'scores'; if missing, accept all.
#     if "scores" in res0:
#         scores_all = res0["scores"].detach().cpu().numpy()
#         keep = scores_all >= float(box_thr)
#     else:
#         keep = np.ones(len(boxes_all), dtype=bool)

#     boxes = boxes_all[keep]
#     labels = [l for l, k in zip(labels_all, keep) if k]

#     return boxes, labels
#     # res0 = results[0]
#     # boxes_all = res0["boxes"].detach().cpu().numpy().astype(float)
#     # # Prefer string labels if present; fall back to ids otherwise
#     # if "text_labels" in res0:
#     #     labels_all = [str(s) for s in res0["text_labels"]]
#     # else:
#     #     labels_all = [str(s) for s in res0["labels"]]
#     # # Optional score filtering (kept from your code)
#     # if "scores" in res0:
#     #     scores_all = res0["scores"].detach().cpu().numpy()
#     #     keep = scores_all >= float(box_thr)
#     # else:
#     #     keep = np.ones(len(boxes_all), dtype=bool)
    
#     # boxes = boxes_all[keep]
#     # labels = [l for l, k in zip(labels_all, keep) if k]
#     # return boxes, labels

# def masks_from_boxes(image_bgr, boxes_xyxy):
#     predictor.set_image(image_bgr[:, :, ::-1].copy())  # BGR->RGB with contiguous copy
#     masks = []
#     for box in boxes_xyxy:
#         pred_masks, _, _ = predictor.predict(box=box, multimask_output=False)
#         m = (pred_masks[0] > 0.5).astype(np.uint8)  # threshold at 0.5
#         masks.append(m)
#     return masks


# def to_people_vs_vehicles(labels, masks):
#     if not masks:
#         return None, None
#     H, W = masks[0].shape
#     people   = np.zeros((H, W), np.uint8)
#     vehicles = np.zeros((H, W), np.uint8)
#     for lab, m in zip(labels, masks):
#         lab = lab.lower()
#         if "person" in lab:
#             people |= m
#         elif any(k in lab for k in ["car", "bus", "truck", "bicycle", "motorcycle", "motorbike", "bike"]):
#             vehicles |= m
#     return people, vehicles


# images = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
# print("Found images:", len(images), "| device:", DEVICE)

# for i, ip in enumerate(images, 1):
#     img = cv2.imread(ip)
#     if img is None:
#         print("skip unreadable:", ip); continue

#     boxes, labels = grounding_dino_boxes(img, TEXT_QUERIES, box_thr=0.35, text_thr=0.25)
#     if len(boxes) == 0:
#         h, w = img.shape[:2]
#         ppl = np.zeros((h, w), np.uint8)
#         veh = np.zeros((h, w), np.uint8)
#     else:
#         det_masks = masks_from_boxes(img, boxes)
#         ppl, veh = to_people_vs_vehicles(labels, det_masks)
#         if ppl is None: ppl = np.zeros_like(det_masks[0])
#         if veh is None: veh = np.zeros_like(det_masks[0])

#     stem = Path(ip).stem
#     cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_people.png"), ppl * 255)
#     cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_vehicles.png"), veh * 255)

#     if i % 10 == 0:
#         print(f"[{i}/{len(images)}] {stem} ✓")

# print("Done. Baseline masks at:", OUT_DIR)


import os, cv2, numpy as np, torch
from glob import glob
from pathlib import Path

IMG_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val"
OUT_DIR = r"C:\Users\umaim\Downloads\Camvid_sam2\outputs\masks_baseline"
CKPT    = r"C:\Users\umaim\Downloads\Camvid_sam2\weights\sam2_hiera_small.pt"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# person + vehicle prompts (matches how you built GT)
TEXT_QUERIES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]

# --- GroundingDINO (HuggingFace) for text->boxes ---
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
gd_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(gd_id)
gd_model  = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(DEVICE).eval()

# --- SAM2 image predictor (Hydra config key, not a path) ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(
    config_file=r"C:\Users\umaim\Downloads\Camvid_sam2\sam2\sam2\configs\sam2\sam2_hiera_s.yaml",
    device=DEVICE,
    checkpoint=CKPT,
)
predictor = SAM2ImagePredictor(sam2_model)

def grounding_dino_boxes(image_bgr, texts, box_thr=0.35, text_thr=0.25):
    """
    Returns (boxes_xyxy, labels).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # GroundingDINO likes period-separated phrases
    prompt = ". ".join(texts) + "."
    inputs = processor(images=image_rgb, text=prompt, return_tensors="pt")
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gd_model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=DEVICE)

    # No threshold parameters here
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        target_sizes=target_sizes,
    )

    if not results or len(results[0].get("boxes", [])) == 0:
        return np.zeros((0, 4), dtype=float), []

    res0 = results[0]
    boxes_all = res0["boxes"].detach().cpu().numpy().astype(float)
    labels_all = [str(s) for s in res0["labels"]]
    
    if "scores" in res0:
        scores_all = res0["scores"].detach().cpu().numpy()
        keep = scores_all >= float(box_thr)
    else:
        keep = np.ones(len(boxes_all), dtype=bool)

    boxes = boxes_all[keep]
    labels = [l for l, k in zip(labels_all, keep) if k]

    return boxes, labels

def masks_from_boxes(image_bgr, boxes_xyxy):
    """FIXED VERSION - addresses memory and thresholding issues"""
    # Ensure image is RGB and contiguous
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    
    predictor.set_image(image_rgb)
    masks = []
    
    for box in boxes_xyxy:
        try:
            # Ensure box is float32 and proper format
            box = np.array(box, dtype=np.float32)
            pred_masks, scores, _ = predictor.predict(
                box=box, 
                multimask_output=False
            )
            
            # Debug: print mask statistics
            print(f"Raw mask shape: {pred_masks.shape}, dtype: {pred_masks.dtype}")
            print(f"Raw mask range: {pred_masks.min():.4f} to {pred_masks.max():.4f}")
            print(f"Mask scores: {scores}")
            
            # Proper thresholding and type conversion
            mask = pred_masks[0]  # Take first (and only) mask
            
            # Apply threshold based on mask values (SAM2 typically outputs logits or probabilities)
            if mask.max() > 1.0:
                # Likely logits, apply sigmoid first
                mask = torch.sigmoid(torch.from_numpy(mask)).numpy()
            
            # Binary threshold
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Ensure contiguous array
            binary_mask = np.ascontiguousarray(binary_mask)
            
            print(f"Final mask unique values: {np.unique(binary_mask)}")
            print(f"Final mask shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")
            
            masks.append(binary_mask)
            
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            # Create empty mask as fallback
            h, w = image_rgb.shape[:2]
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            masks.append(empty_mask)
    
    return masks

def to_people_vs_vehicles(labels, masks):
    """FIXED VERSION - ensures proper mask combining"""
    if not masks:
        return None, None
    
    H, W = masks[0].shape
    people = np.zeros((H, W), dtype=np.uint8)
    vehicles = np.zeros((H, W), dtype=np.uint8)
    
    for lab, m in zip(labels, masks):
        lab = lab.lower()
        # Ensure mask is binary uint8
        m = m.astype(np.uint8)
        
        if "person" in lab:
            people = np.logical_or(people, m).astype(np.uint8)
        elif any(k in lab for k in ["car", "bus", "truck", "bicycle", "motorcycle", "motorbike", "bike"]):
            vehicles = np.logical_or(vehicles, m).astype(np.uint8)
    
    return people, vehicles

# Main processing loop
images = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
print("Found images:", len(images), "| device:", DEVICE)

for i, ip in enumerate(images, 1):
    print(f"\nProcessing image {i}/{len(images)}: {Path(ip).name}")
    
    img = cv2.imread(ip)
    if img is None:
        print("skip unreadable:", ip)
        continue

    # Get detections
    boxes, labels = grounding_dino_boxes(img, TEXT_QUERIES, box_thr=0.35, text_thr=0.25)
    print(f"Found {len(boxes)} detections: {labels}")
    
    if len(boxes) == 0:
        h, w = img.shape[:2]
        ppl = np.zeros((h, w), dtype=np.uint8)
        veh = np.zeros((h, w), dtype=np.uint8)
    else:
        det_masks = masks_from_boxes(img, boxes)
        ppl, veh = to_people_vs_vehicles(labels, det_masks)
        if ppl is None: 
            ppl = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if veh is None: 
            veh = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Debug before saving
    print(f"People mask - shape: {ppl.shape}, dtype: {ppl.dtype}, unique: {np.unique(ppl)}")
    print(f"Vehicle mask - shape: {veh.shape}, dtype: {veh.dtype}, unique: {np.unique(veh)}")
    
    stem = Path(ip).stem
    
    # Ensure masks are contiguous and properly scaled before saving
    ppl_save = np.ascontiguousarray(ppl * 255, dtype=np.uint8)
    veh_save = np.ascontiguousarray(veh * 255, dtype=np.uint8)
    
    success1 = cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_people.png"), ppl_save)
    success2 = cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_vehicles.png"), veh_save)
    
    print(f"Save success - people: {success1}, vehicles: {success2}")

    if i % 10 == 0:
        print(f"[{i}/{len(images)}] {stem} ✓")

print("Done. Baseline masks at:", OUT_DIR)