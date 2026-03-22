import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormer3ClassSegmentor:
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
        
        # Upsampling a la resolución original del tile
        upsampled_probs = torch.nn.functional.interpolate(
            probs, size=tile_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        
        # Clase 1: 'Rock' (ajusta el índice si tu modelo usa otro orden)
        rock_probs = upsampled_probs[0, 1].cpu().numpy()
        mask = np.where(rock_probs > threshold, 1, 0).astype(np.uint8)
        return mask

    def process_folder(self, input_folder: str):
        # 1. Validar ruta de entrada
        input_path = Path(input_folder).resolve()
        if not input_path.exists():
            print(f"ERROR: La carpeta de entrada no existe: {input_path}")
            return

        # 2. Crear carpeta de salida
        output_folder = input_path / "processed_segformer"
        output_folder.mkdir(parents=True, exist_ok=True)

        # 3. Buscar imágenes
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
        
        print(f"--- Iniciando Procesamiento ---")
        print(f"Carpeta origen: {input_path}")
        print(f"Carpeta destino: {output_folder}")
        print(f"Imágenes encontradas: {len(image_files)}")

        if not image_files:
            print("No se encontraron archivos procesables.")
            return

        # --- CONFIGURACIÓN DE UMBRALES (AJUSTAR AQUÍ) ---
        threshold_initial = 0.65       # Menos estricto para detectar presencia general
        threshold_refine = 0.75        # MUCHO más estricto para el refinamiento de objetos grandes
        
        # --- OTRAS CONFIGURACIONES ---
        overlap = 100 
        rows, cols = 3, 3 
        resegment_threshold_pct = 1.0 
        max_area_pct = 20.0 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Error al leer: {img_file.name}")
                continue
            
            h, w = image.shape[:2]
            total_area = h * w
            full_rock_mask = np.zeros((h, w), dtype=np.uint8)
            step_h, step_w = h // rows, w // cols

            # --- Fase 1: Segmentación por Tiles (Umbral Inicial) ---
            for r in range(rows):
                for c in range(cols):
                    y1, y2 = max(0, r * step_h - overlap), min(h, (r + 1) * step_h + overlap)
                    x1, x2 = max(0, c * step_w - overlap), min(w, (c + 1) * step_w + overlap)
                    
                    tile_mask = self.predict_tile(image[y1:y2, x1:x2], threshold=threshold_initial)
                    full_rock_mask[y1:y2, x1:x2] = np.maximum(full_rock_mask[y1:y2, x1:x2], tile_mask)

            # --- Fase 2: Morfología & Re-segmentación Estricta ---
            body_mask = cv2.morphologyEx(full_rock_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(body_mask)
            final_refined_mask = np.zeros_like(body_mask)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                area_pct = (area / total_area) * 100
                x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], 
                               stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])

                if area_pct > max_area_pct: 
                    continue # Ignorar áreas excesivamente grandes (posible ruido de fondo)
                
                elif area_pct > resegment_threshold_pct:
                    # RE-SEGMENTACIÓN CON UMBRAL ESTRICTO
                    pad = 25
                    y1_c, y2_c = max(0, y-pad), min(h, y+bh+pad)
                    x1_c, x2_c = max(0, x-pad), min(w, x+bw+pad)
                    roi = image[y1_c:y2_c, x1_c:x2_c]
                    
                    # Llamada con el segundo umbral (más restrictivo)
                    roi_mask = self.predict_tile(roi, threshold=threshold_refine)
                    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    # Fusionar el ROI refinado en la máscara final
                    final_refined_mask[y1_c:y2_c, x1_c:x2_c] = np.maximum(
                        final_refined_mask[y1_c:y2_c, x1_c:x2_c], roi_mask
                    )
                else:
                    # Mantener áreas pequeñas que pasaron el filtro inicial
                    final_refined_mask[labels_im == i] = 1

            # Limpieza final para suavizar bordes
            final_refined_mask = cv2.morphologyEx(final_refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # --- Visualización ---
            num_labels_final, labels_final = cv2.connectedComponents(final_refined_mask)
            color_mask = np.zeros_like(image)
            for label in range(1, num_labels_final):
                color = np.random.randint(60, 255, (3,)).tolist()
                color_mask[labels_final == label] = color

            blended = cv2.addWeighted(image, 0.7, color_mask, 0.35, 0)
            
            # --- Guardado ---
            out_path = str(output_folder / img_file.name)
            cv2.imwrite(out_path, blended)
            print(f"Procesada: {img_file.name} | Rocas detectadas: {num_labels_final - 1}")

if __name__ == "__main__":
    # Configura tus rutas aquí
    model_path = "/home/lithos_analithics_challenge/weights/segformer_approach/final"
    valid_folder = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    segmentor = SegFormer3ClassSegmentor(model_path)
    segmentor.process_folder(valid_folder)