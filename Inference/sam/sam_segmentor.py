import cv2
import torch
import numpy as np
import torchvision
import gc
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Segmentor:
    def __init__(self, model_cfg: str, checkpoint_path: str, max_area_ratio: float = 0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.max_area_ratio = max_area_ratio
        
        # --- CONFIGURACIÓN PARA EVITAR REFINAMIENTO ITERATIVO ---
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,            # Reducimos a 32 (1024 pts) para que la CPU no colapse
            points_per_batch=64,
            pred_iou_thresh=0.8,           # Umbral más alto para procesar menos máscaras mediocres
            stability_score_thresh=0.85,
            min_mask_region_area=100,
            
            # PARÁMETROS CLAVE PARA EVITAR EL ERROR DE _C:
            use_m2m=False,                 # DESACTIVA el refinamiento iterativo (Mask-to-Mask)
            multimask_output=False,        # Genera solo la mejor máscara por punto (ahorra CPU)
            output_mode="binary_mask"      # Asegura que devuelva máscaras simples
        )

    def apply_custom_preprocessing(self, bgr_image):
        # Reducimos d=5 para que el filtro bilateral no sume tanto tiempo de CPU
        smoothed = cv2.bilateralFilter(bgr_image, d=5, sigmaColor=50, sigmaSpace=50)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process_image(self, image_path: Path):
        image = cv2.imread(str(image_path))
        if image is None: return None, None, 0
        
        h, w = image.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        overlap = 150 # Menos solapamiento para procesar menos área repetida
        border_margin = 50
        
        tiles_coords = [
            (0, mid_h + overlap, 0, mid_w + overlap),
            (0, mid_h + overlap, mid_w - overlap, w),
            (mid_h - overlap, h, 0, mid_w + overlap),
            (mid_h - overlap, h, mid_w - overlap, w)
        ]
        
        all_boxes, all_confs, all_masks_data = [], [], []

        for (y1, y2, x1, x2) in tiles_coords:
            tile = image[y1:y2, x1:x2]
            proc_tile = self.apply_custom_preprocessing(tile)
            tile_rgb = cv2.cvtColor(proc_tile, cv2.COLOR_BGR2RGB)
            
            # Liberar memoria antes de cada tile
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # Aquí es donde SAM 2 generará el warning, pero no se quedará pegado
                    masks = self.mask_generator.generate(tile_rgb)
            
            for m_dict in masks:
                if m_dict['area'] > ((y2-y1)*(x2-x1) * self.max_area_ratio):
                    continue
                
                tx1, ty1, tw, th = m_dict['bbox']
                all_boxes.append([tx1 + x1, ty1 + y1, tx1 + tw + x1, ty1 + th + y1])
                all_confs.append(m_dict['predicted_iou'])
                all_masks_data.append({"segmentation": m_dict['segmentation'], "coords": (y1, y2, x1, x2)})

        if not all_boxes: return image, [], 0

        # NMS Global (necesario por el solapamiento de tiles)
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        confs_tensor = torch.tensor(all_confs, dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_tensor, confs_tensor, iou_threshold=0.5)

        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        allowed_region = np.zeros((h, w), dtype=bool)
        allowed_region[border_margin:h-border_margin, border_margin:w-border_margin] = True
        
        final_individual_masks = []
        for idx in keep_indices:
            idx = idx.item()
            m_data = all_masks_data[idx]
            m, (y1, y2, x1, x2) = m_data["segmentation"], m_data["coords"]
            
            tile_allowed = allowed_region[y1:y2, x1:x2]
            final_bool_mask = np.logical_and(m, tile_allowed)
            
            if np.any(final_bool_mask):
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2][final_bool_mask] = 255
                final_individual_masks.append(full_mask)
                
                color = np.random.randint(60, 255, (3,)).tolist()
                mask_canvas[y1:y2, x1:x2][final_bool_mask] = color

        blended = cv2.addWeighted(image, 0.7, mask_canvas, 0.35, 0)
        return blended, final_individual_masks, len(final_individual_masks)