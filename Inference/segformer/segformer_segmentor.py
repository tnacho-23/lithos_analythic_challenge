import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormerSegmentor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_tile(self, tile_bgr, threshold=0.7):
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=tile_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        upsampled_probs = torch.nn.functional.interpolate(
            probs, size=tile_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        # Clase 1: 'Rock'
        rock_probs = upsampled_probs[0, 1].cpu().numpy()
        return np.where(rock_probs > threshold, 1, 0).astype(np.uint8)

    def process_image(self, image_path: Path):
        image = cv2.imread(str(image_path))
        if image is None: return None, None, 0
        
        h, w = image.shape[:2]
        total_area = h * w
        full_rock_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Configuración de tiles (3x3 con overlap)
        rows, cols, overlap = 3, 3, 100
        step_h, step_w = h // rows, w // cols
        threshold_initial = 0.65
        threshold_refine = 0.75
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Fase 1: Inferencia por Tiles
        for r in range(rows):
            for c in range(cols):
                y1, y2 = max(0, r * step_h - overlap), min(h, (r + 1) * step_h + overlap)
                x1, x2 = max(0, c * step_w - overlap), min(w, (c + 1) * step_w + overlap)
                tile_mask = self.predict_tile(image[y1:y2, x1:x2], threshold=threshold_initial)
                full_rock_mask[y1:y2, x1:x2] = np.maximum(full_rock_mask[y1:y2, x1:x2], tile_mask)

        # Fase 2: Refinamiento (Re-segmentación de áreas grandes)
        body_mask = cv2.morphologyEx(full_rock_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(body_mask)
        final_refined_mask = np.zeros_like(body_mask)

        for i in range(1, num_labels):
            area_pct = (stats[i, cv2.CC_STAT_AREA] / total_area) * 100
            if area_pct > 20.0: continue
            elif area_pct > 1.0:
                x, y, bw, bh = stats[i, :4]
                pad = 25
                y1_c, y2_c = max(0, y-pad), min(h, y+bh+pad)
                x1_c, x2_c = max(0, x-pad), min(w, x+bw+pad)
                roi_mask = self.predict_tile(image[y1_c:y2_c, x1_c:x2_c], threshold=threshold_refine)
                final_refined_mask[y1_c:y2_c, x1_c:x2_c] = np.maximum(final_refined_mask[y1_c:y2_c, x1_c:x2_c], roi_mask)
            else:
                final_refined_mask[labels_im == i] = 1

        final_refined_mask = cv2.morphologyEx(final_refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Separación en máscaras individuales para RockAnalytics
        num_labels_final, labels_final = cv2.connectedComponents(final_refined_mask)
        individual_masks = []
        color_mask = np.zeros_like(image)

        for label in range(1, num_labels_final):
            mask_single = np.where(labels_final == label, 255, 0).astype(np.uint8)
            individual_masks.append(mask_single)
            # Para visualización
            color = np.random.randint(60, 255, (3,)).tolist()
            color_mask[labels_final == label] = color

        blended = cv2.addWeighted(image, 0.7, color_mask, 0.35, 0)
        return blended, individual_masks, num_labels_final - 1