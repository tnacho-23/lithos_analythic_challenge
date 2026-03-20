import cv2
import torch
import numpy as np
import torchvision
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2RobustSegmentor:
    def __init__(self, model_cfg: str, checkpoint_path: str, max_area_ratio: float = 0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.max_area_ratio = max_area_ratio
        
        # [MODIFICADO] Configuración de ALTA SENSIBILIDAD para rocas pequeñas
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,            # Más puntos = detecta cosas más pequeñas
            points_per_batch=64, 
            pred_iou_thresh=0.5,           # [BAJADO] Más permisivo con la calidad
            stability_score_thresh=0.75,    # [BAJADO] Más permisivo con bordes difusos
            min_mask_region_area=50,       # Eliminar solo ruido muy pequeño
        )

    def apply_custom_preprocessing(self, bgr_image):
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process_folder(self, input_folder: str):
        input_path = Path(input_folder)
        output_folder = input_path / "processed_sam2"
        output_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        overlap = 300 
        iou_threshold = 0.6 
        
        # [NUEVO] Definir el margen de exclusión de bordes
        border_margin = 50

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None: continue
            h, w = image.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            
            # Tiling coords
            tiles_coords = [
                (0, mid_h + overlap, 0, mid_w + overlap),
                (0, mid_h + overlap, mid_w - overlap, w),
                (mid_h - overlap, h, 0, mid_w + overlap),
                (mid_h - overlap, h, mid_w - overlap, w)
            ]
            
            all_boxes = []
            all_confs = []
            all_masks = []

            print(f"--- Procesando SAM2 Tiled & Cropped: {img_file.name} ---")

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                proc_tile = self.apply_custom_preprocessing(tile)
                tile_rgb = cv2.cvtColor(proc_tile, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        masks = self.mask_generator.generate(tile_rgb)
                
                for m_dict in masks:
                    # Filtro de área relativa
                    if m_dict['area'] > ((y2-y1)*(x2-x1) * self.max_area_ratio):
                        continue
                    
                    # bbox global
                    tx1, ty1, tw, th = m_dict['bbox']
                    global_box = [tx1 + x1, ty1 + y1, tx1 + tw + x1, ty1 + th + y1]
                    
                    all_boxes.append(global_box)
                    all_confs.append(m_dict['predicted_iou']) 
                    all_masks.append({
                        "segmentation": m_dict['segmentation'], 
                        "coords": (y1, y2, x1, x2)
                    })

            if not all_boxes:
                print(f"No se detectó nada en {img_file.name}")
                continue

            # NMS Global
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            confs_tensor = torch.tensor(all_confs, dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_tensor, confs_tensor, iou_threshold)

            mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # [NUEVO] Crear una máscara booleana global de la zona permitida
            # Todo True excepto los bordes de 50px
            allowed_region_mask = np.zeros((h, w), dtype=bool)
            allowed_region_mask[border_margin:h-border_margin, border_margin:w-border_margin] = True

            for idx in keep_indices:
                idx = idx.item()
                m_data = all_masks[idx]
                m, (y1, y2, x1, x2) = m_data["segmentation"], m_data["coords"]
                
                # [NUEVO] Intersección: Solo pintamos si está en la máscara de SAM
                # Y ADEMÁS está dentro de la región permitida global.
                
                # Primero extraemos la región permitida correspondiente a este tile
                tile_allowed_region = allowed_region_mask[y1:y2, x1:x2]
                
                # Combinamos con la máscara de SAM usando AND lógico
                final_bool_mask = np.logical_and(m, tile_allowed_region)
                
                # Solo pintamos si queda algo de máscara después del recorte
                if np.any(final_bool_mask):
                    color = np.random.randint(60, 255, (3,)).tolist()
                    mask_canvas[y1:y2, x1:x2][final_bool_mask] = color

            blended = cv2.addWeighted(image, 0.7, mask_canvas, 0.35, 0)
            
            # [OPCIONAL] Dibujar un rectángulo rojo para visualizar el borde excluido
            # cv2.rectangle(blended, (border_margin, border_margin), (w-border_margin, h-border_margin), (0,0,255), 2)
            
            cv2.imwrite(str(output_folder / img_file.name), blended)
            print(f"Detecciones únicas SAM2 (sin bordes): {len(keep_indices)}\n")

if __name__ == "__main__":
    # Actualiza estas rutas
    CHECKPOINT = "/home/checkpoints/sam2_hiera_base_plus.pt"
    CONFIG = "sam2_hiera_b+.yaml" 
    FOLDER_PATH = "/home/lithos_analithics_challenge/images/given_dataset/valid"

    segmentor = SAM2RobustSegmentor(CONFIG, CHECKPOINT)
    segmentor.process_folder(FOLDER_PATH)