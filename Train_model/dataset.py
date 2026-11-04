# dataset.py
import re
import cv2
import numpy as np
import pyclipper
import torch
from pathlib import Path
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from config import IMG_SIZE, SHRINK_RATIO

class DBNetDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = cv2.imread(str(file_path))
        h_orig, w_orig = image.shape[:2]
        
        W, H = IMG_SIZE
        image = cv2.resize(image, (W, H))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Chuẩn hóa ImageNet cơ bản
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        gt_score = np.zeros((H, W), dtype=np.float32)
        gt_thresh = np.zeros((H, W), dtype=np.float32)

        # Ánh xạ tên file ảnh im0001.jpg -> gt_1.txt
        match = re.match(r"im(\d+)", file_path.name)
        if match:
            label_path = self.label_dir / f"gt_{int(match.group(1))}.txt"
            if label_path.exists():
                with open(label_path, encoding="utf-8-sig") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 8: continue
                        pts = np.array(parts[:8], dtype=np.float32).reshape(4, 2)
                        pts[:, 0] *= W / w_orig
                        pts[:, 1] *= H / h_orig
                        int_pts = pts.astype(np.int32)

                        cv2.fillPoly(gt_thresh, [int_pts], 1.0)
                        
                        # Tạo polygon thu nhỏ cho probability map
                        try:
                            poly = Polygon(int_pts)
                            distance = poly.area * (1 - SHRINK_RATIO ** 2) / (poly.length + 1e-5)
                            offset = pyclipper.PyclipperOffset()
                            offset.AddPath(int_pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                            shrinked = offset.Execute(-distance)
                            if shrinked:
                                cv2.fillPoly(gt_score, [np.array(shrinked[0], dtype=np.int32)], 1.0)
                        except:
                            pass

        gt_thresh = cv2.dilate(gt_thresh, np.ones((3, 3), np.uint8)) * 0.7

        return image_tensor, torch.from_numpy(gt_score).unsqueeze(0), torch.from_numpy(gt_thresh).unsqueeze(0)