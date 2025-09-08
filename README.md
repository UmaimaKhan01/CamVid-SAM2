
````markdown
# CamVid-SAM2

This project applies **Segment Anything Model 2 (SAM2)** for automatic segmentation of **people** and **vehicles** in the CamVid dataset.  
It follows the assignment requirements: baseline with text prompts and an improved algorithm that beats the baseline using YOLO + SAM2 fusion.

---

## 📂 Dataset
- **CamVid dataset**: [Kaggle Link](https://www.kaggle.com/datasets/carlolepelaars/camvid)  
- Validation set (`val/`) used for evaluation.  
- Ground truth masks for people and vehicles are in:
  - `data/val_gt_people/`
  - `data/val_gt_vehicles/`

---

## ⚙️ Methods

### Baseline
- Uses **text prompts only** with GroundingDINO + SAM2.
- Prompts: `["person", "car", "bus", "truck", "bicycle", "motorcycle"]`
- Generates masks without manual clicks/boxes.

### Improved: YOLO + SAM2
- Runs **YOLOv8** for bounding box detection.
- Boxes are passed as prompts to **SAM2** to refine segmentation.
- Sweeps confidence + mask thresholds for best Dice score.

---

## 📊 Results (Dice & IoU)

| Method      | People Dice | Vehicles Dice | Overall Dice |
|-------------|-------------|---------------|--------------|
| **Baseline (Text prompt)** | 0.012 | 0.070 | 0.041 |
| **YOLO+SAM2 (best tuned)** | 0.023 | 0.101 | 0.062 |

---

## 🚀 Usage

### 1. Install dependencies
```bash
conda create -n sam2_env python=3.10 -y
conda activate sam2_env
pip install -r requirements.txt
````

### 2. Baseline (Text Prompt)

```bash
python scripts/baseline_text2sam2.py
```

### 3. YOLO + SAM2 (Improved)

```bash
python scripts/yolosam2_tuned.py
```

### 4. Evaluation

```bash
python scripts/eval_dice.py \
  --gt_people_dir data/val_gt_people \
  --gt_veh_dir data/val_gt_vehicles \
  --pred_dir outputs/masks_yolo_sam2 \
  --verbose
```

---

## 📁 Repo Structure

```
CamVid-SAM2/
├── data/                # CamVid val images + GT masks
├── outputs/             # Generated masks & results
├── scripts/             # Baseline + YOLO-SAM2 + evaluation scripts
├── sam2/                # Local SAM2 package
├── requirements.txt     # Dependencies
└── README.md            # Project report (this file)
```

---

## ✨ Notes

* No manual clicks/boxes needed → everything is automatic.
* Evaluated using **Dice score & IoU**.
* YOLO+SAM2 clearly outperforms the text-only baseline.

---



👉 Want me to also generate a **shorter polished project description** (like a paragraph) that you can paste in the GitHub repo’s **About section**?
```
