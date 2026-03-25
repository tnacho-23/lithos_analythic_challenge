import os
import cv2
import numpy as np
import pandas as pd
import torch
import sys
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

# --- CONFIGURACIÓN DE RUTAS ---
TEST_BASE_PATH = Path("/home/lithos_analithics_challenge/images/full_dataset_processed/test")
IMAGES_PATH = TEST_BASE_PATH / "images"
LABELS_PATH = TEST_BASE_PATH / "labels"

try:
    from yolo.yolo_segmentor import YOLOSegmentor
    from segformer.segformer_segmentor import SegFormerSegmentor
    from sam.sam_segmentor import SAM2Segmentor
except ImportError as e:
    print(f"[!] Error de importación: {e}. Asegúrate de ejecutar desde la raíz del proyecto.")
    sys.exit(1)

class LithosEvaluator:
    """Evaluador de granulometría que genera reportes CSV independientes por método."""
    
    def __init__(self, pixel_to_mm: float = 1.0):
        self.pixel_to_mm = pixel_to_mm
        self.save_dir = Path(__file__).resolve().parent / "metrics"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _yolo_to_masks(self, label_file: Path, img_shape: Tuple[int, int]) -> List[np.ndarray]:
        h, w = img_shape[:2]
        masks = []
        if not label_file.exists(): return masks
        with open(label_file, 'r') as f:
            for line in f.readlines():
                data = line.split()
                if len(data) < 3: continue
                points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points.astype(np.int32)], 1)
                masks.append(mask)
        return masks

    def get_granulometry_points(self, masks: List[np.ndarray]) -> Tuple[float, float, float]:
        if not masks: return 0.0, 0.0, 0.0
        areas = [np.sum(m) * (self.pixel_to_mm**2) for m in masks]
        diameters = [2 * np.sqrt(a / np.pi) for a in areas]
        diameters.sort()
        volumes = np.array(diameters)**3
        total_vol = np.sum(volumes)
        if total_vol == 0: return 0.0, 0.0, 0.0
        cumulative_pass = np.cumsum(volumes) / total_vol
        d20 = float(np.interp(0.20, cumulative_pass, diameters))
        d50 = float(np.interp(0.50, cumulative_pass, diameters))
        d80 = float(np.interp(0.80, cumulative_pass, diameters))
        return d20, d50, d80

    def calculate_iou(self, gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], img_shape: Tuple[int, int]) -> float:
        h, w = img_shape[:2]
        full_gt = np.zeros((h, w), dtype=np.uint8)
        full_pred = np.zeros((h, w), dtype=np.uint8)
        for m in gt_masks: full_gt[m > 0] = 1
        for m in pred_masks: full_pred[m > 0] = 1
        intersection = np.logical_and(full_gt, full_pred).sum()
        union = np.logical_or(full_gt, full_pred).sum()
        return float(intersection / union) if union > 0 else 1.0

    def evaluate_and_save(self, model: Any, method_name: str):
        """Procesa el dataset y guarda un CSV exclusivo para este método."""
        results = []
        img_files = list(IMAGES_PATH.glob("*.jpg")) + list(IMAGES_PATH.glob("*.png"))
        print(f"\n>>> EVALUANDO MÉTODO: {method_name.upper()}")
        
        for img_path in tqdm(img_files, desc=f"Procesando {method_name}"):
            # Medir tiempo de inferencia (Requisito Módulo 2)
            start_time = time.time()
            _, pred_masks, _ = model.process_image(img_path)
            inference_time = time.time() - start_time
            
            img_cv = cv2.imread(str(img_path))
            if img_cv is None: continue
            h, w = img_cv.shape[:2]
            
            gt_masks = self._yolo_to_masks(LABELS_PATH / f"{img_path.stem}.txt", (h, w))
            gt_pts = self.get_granulometry_points(gt_masks)
            pr_pts = self.get_granulometry_points(pred_masks)
            iou = self.calculate_iou(gt_masks, pred_masks, (h, w))
            
            results.append({
                "imagen": img_path.name,
                "iou": round(iou, 4),
                "inference_time_sec": round(inference_time, 4),
                "gt_d50": round(gt_pts[1], 2),
                "pred_d50": round(pr_pts[1], 2),
                "gt_d80": round(gt_pts[2], 2),
                "pred_d80": round(pr_pts[2], 2),
                "gt_count": len(gt_masks),
                "pred_count": len(pred_masks),
                "error_abs_d50": round(abs(gt_pts[1] - pr_pts[1]), 2)
            })

        # Guardar CSV individual
        df = pd.DataFrame(results)
        csv_path = self.save_dir / f"detalle_{method_name.upper()}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Detalle guardado en: {csv_path}")

def main():
    evaluator = LithosEvaluator(pixel_to_mm=1.0)

    model_configs = [
        {"name": "yolo", "weights": "/home/lithos_analithics_challenge/weights/yolo/best.pt"},
        {"name": "segformer", "weights": "/home/lithos_analithics_challenge/weights/segformer/final"},
        {"name": "sam2", "cfg": "sam2_hiera_b+.yaml", "ckpt": "/home/checkpoints/sam2_hiera_base_plus.pt"}
    ]

    for config in model_configs:
        name = config["name"]
        try:
            print(f"[*] Cargando {name.upper()}...")
            if name == "yolo": model = YOLOSegmentor(config["weights"])
            elif name == "segformer": model = SegFormerSegmentor(config["weights"])
            elif name == "sam2": model = SAM2Segmentor(config["cfg"], config["ckpt"])
            
            evaluator.evaluate_and_save(model, name)
            
            # Limpieza de memoria para el siguiente modelo
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[!] Error crítico en {name}: {e}")

    print(f"\n[FIN] Todos los reportes CSV se encuentran en: {evaluator.save_dir}")

if __name__ == "__main__":
    main()