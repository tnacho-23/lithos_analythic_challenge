import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormer3ClassSegmentor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_tile(self, tile_bgr):
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=tile_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=tile_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        # Retorna 0, 1 o 2 por píxel
        return upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

    def process_folder(self, input_folder: str):
        input_path = Path(input_folder)
        output_folder = input_path / "processed_segformer"
        output_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        overlap = 300 

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None: continue
            h, w = image.shape[:2]
            
            # 1. Reconstrucción de la máscara global de 3 clases
            full_3class_mask = np.zeros((h, w), dtype=np.uint8)
            mid_h, mid_w = h // 2, w // 2
            tiles_coords = [
                (0, mid_h + overlap, 0, mid_w + overlap),
                (0, mid_h + overlap, mid_w - overlap, w),
                (mid_h - overlap, h, 0, mid_w + overlap),
                (mid_h - overlap, h, mid_w - overlap, w)
            ]

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                tile_mask = self.predict_tile(tile)
                # Guardamos la predicción completa (0, 1, 2)
                full_3class_mask[y1:y2, x1:x2] = np.maximum(full_3class_mask[y1:y2, x1:x2], tile_mask.astype(np.uint8))

            # --- LÓGICA DE SEPARACIÓN ---
            # 2. Creamos una máscara binaria solo con la clase 1 (Cuerpo)
            # Esto automáticamente deja fuera los bordes (clase 2)
            body_mask = np.zeros_like(full_3class_mask)
            body_mask[full_3class_mask == 1] = 1

            # 3. (Opcional) Dilatar un poco para recuperar el tamaño perdido por el borde
            # pero sin que lleguen a tocarse.
            kernel = np.ones((3,3), np.uint8)
            body_mask = cv2.dilate(body_mask, kernel, iterations=1)

            # 4. Identificar rocas individuales
            num_labels, labels_im = cv2.connectedComponents(body_mask)

            # --- VISUALIZACIÓN ---
            color_mask = np.zeros_like(image)
            for label in range(1, num_labels):
                color = np.random.randint(60, 255, (3,)).tolist()
                color_mask[labels_im == label] = color

            # Opcional: Pintar los bordes detectados en un color tenue (ej: blanco)
            # color_mask[full_3class_mask == 2] = [200, 200, 200]

            blended = cv2.addWeighted(image, 0.7, color_mask, 0.35, 0)
            cv2.imwrite(str(output_folder / img_file.name), blended)
            print(f"Imagen: {img_file.name} | Rocas separadas: {num_labels - 1}")

if __name__ == "__main__":
    model_path = "/home/lithos_analithics_challenge/weights/segformer_approach/final"
    valid_folder = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    segmentor = SegFormer3ClassSegmentor(model_path)
    segmentor.process_folder(valid_folder)