import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormerRockSegmentor:
    def __init__(self, model_path: str):
        # 1. Cargar el procesador y el modelo entrenado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def apply_custom_preprocessing(self, bgr_image):
        # Mantenemos tu pre-procesado que ayuda a resaltar bordes de rocas
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def predict_tile(self, tile_bgr):
        # Convertir a PIL para el processor de Hugging Face
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=tile_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_labels, height/4, width/4]

        # Redimensionar logits al tamaño original del tile
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=tile_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # Obtener la clase con mayor probabilidad (0 o 1)
        pred = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return pred

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
            mid_h, mid_w = h // 2, w // 2
            
            # Coordenadas de los 4 cuadrantes con overlap (igual que en tu YOLO approach)
            tiles_coords = [
                (0, mid_h + overlap, 0, mid_w + overlap),
                (0, mid_h + overlap, mid_w - overlap, w),
                (mid_h - overlap, h, 0, mid_w + overlap),
                (mid_h - overlap, h, mid_w - overlap, w)
            ]
            
            # Canvas global para reconstruir la máscara completa
            # Usamos float32 para promediar zonas de overlap si quisiéramos, 
            # pero aquí haremos una unión simple para velocidad.
            full_mask = np.zeros((h, w), dtype=np.uint8)

            print(f"--- Procesando SegFormer: {img_file.name} ---")

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                proc_tile = self.apply_custom_preprocessing(tile)
                
                tile_mask = self.predict_tile(proc_tile)
                
                # Pegar el resultado del tile en la máscara global
                # El valor 1 indica 'roca'
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], tile_mask.astype(np.uint8))

            # --- VISUALIZACIÓN ---
            # Crear un canvas de color para las rocas (puedes elegir un color fijo, ej: Verde)
            color_mask = np.zeros_like(image)
            color_mask[full_mask == 1] = [0, 255, 0] # Pintar rocas en verde

            # Mezclar con la imagen original
            blended = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
            
            cv2.imwrite(str(output_folder / img_file.name), blended)
            print(f"Imagen guardada en {output_folder / img_file.name}\n")

if __name__ == "__main__":
    # Ruta al modelo que guardaste al final del entrenamiento
    model_path = "/home/lithos_analithics_challenge/weights/segformer_approach/final_rock_model"
    valid_folder = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    try:
        segmentor = SegFormerRockSegmentor(model_path)
        segmentor.process_folder(valid_folder)
    except Exception as e:
        print(f"ERROR: {e}")