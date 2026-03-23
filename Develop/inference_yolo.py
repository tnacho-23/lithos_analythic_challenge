import cv2
import numpy as np
import torch
import torchvision  # <--- Añadimos esta importación
from ultralytics import YOLO
from pathlib import Path

class YOLOSegmentor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task='segment')

    def apply_custom_preprocessing(self, bgr_image):
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process_folder(self, input_folder: str):
        input_path = Path(input_folder)
        output_folder = input_path / "processed_yolo"
        output_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        overlap = 300 
        iou_threshold = 0.9 

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None: continue
            h, w = image.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            
            tiles_coords = [
                (0, mid_h + overlap, 0, mid_w + overlap),
                (0, mid_h + overlap, mid_w - overlap, w),
                (mid_h - overlap, h, 0, mid_w + overlap),
                (mid_h - overlap, h, mid_w - overlap, w)
            ]
            
            all_boxes = []
            all_confs = []
            all_masks = []

            print(f"--- Procesando con NMS Global: {img_file.name} ---")

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                proc_tile = self.apply_custom_preprocessing(tile)
                results = self.model.predict(proc_tile, conf=0.10, imgsz=640, verbose=False)
                
                res = results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.clone()
                    boxes[:, [0, 2]] += x1
                    boxes[:, [1, 3]] += y1
                    
                    all_boxes.append(boxes)
                    all_confs.append(res.boxes.conf)
                    
                    for i, mask in enumerate(res.masks.data):
                        # Redimensionamos la máscara al tamaño real del tile antes de guardarla
                        m = mask.cpu().numpy()
                        m = cv2.resize(m, (x2 - x1, y2 - y1))
                        all_masks.append({"mask": m, "coords": (y1, y2, x1, x2)})

            if not all_boxes: 
                print(f"No se detectaron rocas en {img_file.name}")
                continue

            all_boxes = torch.cat(all_boxes)
            all_confs = torch.cat(all_confs)

            # CORRECCIÓN AQUÍ: Usamos torchvision.ops.nms
            # Pasamos las cajas y las confianzas (deben estar en la misma GPU/CPU)
            keep_indices = torchvision.ops.nms(all_boxes, all_confs, iou_threshold)
            
            mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            final_count = 0

            for idx in keep_indices:
                idx = idx.item()
                m_data = all_masks[idx]
                m, (y1, y2, x1, x2) = m_data["mask"], m_data["coords"]
                
                bool_mask = m > 0.5
                color = np.random.randint(60, 255, (3,)).tolist()
                
                # Pintamos en la región global correspondiente
                tile_area = mask_canvas[y1:y2, x1:x2]
                tile_area[bool_mask] = color
                final_count += 1

            blended = cv2.addWeighted(image, 0.7, mask_canvas, 0.35, 0)
            cv2.imwrite(str(output_folder / img_file.name), blended)
            print(f"Detecciones únicas finales: {final_count}\n")

if __name__ == "__main__":
    pt_path = "/home/lithos_analithics_challenge/weights/yolo_approach/train_v1_medium/weights/best.pt"
    valid_folder = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    try:
        segmentor = YOLOSegmentor(pt_path)
        segmentor.process_folder(valid_folder)
    except Exception as e:
        print(f"ERROR: {e}")