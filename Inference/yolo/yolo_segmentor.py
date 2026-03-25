import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from pathlib import Path

class YOLOSegmentor:
    def __init__(self, model_path: str):
        # Cargamos el modelo YOLOv8-seg
        self.model = YOLO(model_path, task='segment')

    def apply_custom_preprocessing(self, bgr_image):
        """Mejora el contraste para segmentación de rocas."""
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process_image(self, image_path: Path, overlap=300, iou_threshold=0.9):
        """Procesa una sola imagen usando tiles y retorna máscaras globales."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, 0
        
        h, w = image.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        
        # Coordenadas de los 4 cuadrantes (tiles) con traslape
        tiles_coords = [
            (0, mid_h + overlap, 0, mid_w + overlap),
            (0, mid_h + overlap, mid_w - overlap, w),
            (mid_h - overlap, h, 0, mid_w + overlap),
            (mid_h - overlap, h, mid_w - overlap, w)
        ]
        
        all_boxes = []
        all_confs = []
        all_masks_data = []

        for (y1, y2, x1, x2) in tiles_coords:
            tile = image[y1:y2, x1:x2]
            proc_tile = self.apply_custom_preprocessing(tile)
            
            # Inferencia
            results = self.model.predict(proc_tile, conf=0.30, imgsz=640, verbose=False)
            res = results[0]
            
            if res.boxes is not None and len(res.boxes) > 0:
                # 1. Ajustar cajas a coordenadas globales
                boxes = res.boxes.xyxy.clone()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                
                all_boxes.append(boxes)
                all_confs.append(res.boxes.conf)
                
                # 2. Redimensionar y guardar máscaras relitivas al canvas global
                for i, mask in enumerate(res.masks.data):
                    m = mask.cpu().numpy()
                    # Resize al tamaño del tile original (antes de meterlo a YOLO)
                    m_resized = cv2.resize(m, (x2 - x1, y2 - y1))
                    all_masks_data.append({"mask": m_resized, "coords": (y1, y2, x1, x2)})

        if not all_boxes:
            return image, [], 0

        # Concatenar resultados de todos los tiles
        all_boxes = torch.cat(all_boxes)
        all_confs = torch.cat(all_confs)

        # NMS Global para eliminar duplicados en las zonas de overlap
        keep_indices = torchvision.ops.nms(all_boxes, all_confs, iou_threshold)
        
        # Filtrar máscaras finales
        final_masks = []
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for idx in keep_indices:
            idx = idx.item()
            m_info = all_masks_data[idx]
            m, (y1, y2, x1, x2) = m_info["mask"], m_info["coords"]
            
            # Crear máscara booleana global para el canvas
            bool_mask = m > 0.5
            color = np.random.randint(60, 255, (3,)).tolist()
            
            # Pintar en el canvas para la visualización
            tile_area = mask_canvas[y1:y2, x1:x2]
            tile_area[bool_mask] = color
            
            # Guardamos la máscara reconstruida en su posición global para analítica
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = (bool_mask * 255).astype(np.uint8)
            final_masks.append(full_mask)

        # Crear imagen combinada (Blended)
        blended = cv2.addWeighted(image, 0.7, mask_canvas, 0.35, 0)
        
        return blended, final_masks, len(keep_indices)