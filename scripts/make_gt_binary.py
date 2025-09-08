import os, cv2, numpy as np
from glob import glob

IMG_DIR  = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val"
LAB_DIR  = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val_labels"
OUT_PEOP = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val_gt_people"
OUT_VEHS = r"C:\Users\umaim\Downloads\Camvid_sam2\data\val_gt_vehicles"
os.makedirs(OUT_PEOP, exist_ok=True)
os.makedirs(OUT_VEHS, exist_ok=True)

# CamVid (SegNet 11-class) common palette (RGB)
# Adjust if your set uses slightly different colors.
PALETTE = {
    "Sky":        (128,128,128),
    "Building":   (128,0,0),
    "Pole":       (192,192,128),
    "Road":       (128,64,128),
    "Pavement":   (60,40,222),
    "Tree":       (128,128,0),
    "SignSymbol": (192,128,128),
    "Fence":      (64,64,128),
    "Car":        (64,0,128),
    "Pedestrian": (64,64,0),
    "Bicyclist":  (0,128,192),
    "Unlabelled": (0,0,0),
}

# define target color sets (people & vehicles)
PEOPLE_COLORS   = {PALETTE["Pedestrian"]}                     # add rider here if present
VEHICLE_COLORS  = {PALETTE["Car"], PALETTE["Bicyclist"]}      # add Bus/Truck/Motorcycle if present

def rgb_to_mask(label_img, colors):
    # label_img is BGR from cv2.imread -> convert to RGB first
    rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for c in colors:
        r,g,b = c
        hits = (rgb[:,:,0]==r) & (rgb[:,:,1]==g) & (rgb[:,:,2]==b)
        mask[hits] = 1
    return mask

# list pairs by stripping optional "_L"
def stem(p): return os.path.splitext(os.path.basename(p))[0]
def norm_name(name): return name[:-2] if name.endswith("_L") else name

imgs = sorted(glob(os.path.join(IMG_DIR, "*.png")) + glob(os.path.join(IMG_DIR, "*.jpg")))
labs = sorted(glob(os.path.join(LAB_DIR, "*.png")) + glob(os.path.join(LAB_DIR, "*.jpg")))
img_map  = {stem(p): p for p in imgs}
lab_map  = {norm_name(stem(p)): p for p in labs}
keys = sorted(set(img_map.keys()) & set(lab_map.keys()))

unknown_colors = set()
for k in keys:
    lab_p = lab_map[k]
    lab = cv2.imread(lab_p, cv2.IMREAD_COLOR)
    if lab is None:
        print("WARN: could not read", lab_p); continue

    people_mask  = rgb_to_mask(lab, PEOPLE_COLORS)
    vehicle_mask = rgb_to_mask(lab, VEHICLE_COLORS)

    # Optional: track palette coverage to verify
    rgb = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB).reshape(-1,3)
    uniq = np.unique(rgb, axis=0)
    for tup in map(tuple, uniq):
        if tup not in PALETTE.values():
            unknown_colors.add(tup)

    # save masks as 0/255 PNGs
    cv2.imwrite(os.path.join(OUT_PEOP, f"{k}.png"), (people_mask*255))
    cv2.imwrite(os.path.join(OUT_VEHS, f"{k}.png"), (vehicle_mask*255))

print(f"wrote people masks to: {OUT_PEOP}")
print(f"wrote vehicle masks to: {OUT_VEHS}")
if unknown_colors:
    print("NOTE: found colors not in current PALETTE (likely extra classes):", list(unknown_colors)[:10], "... total:", len(unknown_colors))
