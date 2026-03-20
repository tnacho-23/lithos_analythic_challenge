import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from typing import Dict
from pathlib import Path

class YOLOSegmentor:
    def __init__(self, model_path: str):
        # Cargamos el modelo (ONNX o PT)
        self.model = YOLO(model_path, task='segment')

    def apply_custom_preprocessing(self, bgr_image):
        """Pipeline: Bilateral Filter -> LAB -> CLAHE -> BGR"""
        # 1. Filtro Bilateral para suavizar ruido sin perder bordes
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        # 2. Conversión a LAB para procesar luminancia
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 3. Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # 4. Re-combinar y volver a BGR
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def process_folder(self, input_folder: str) -> Dict:
        """Procesa imágenes dividiéndolas en 4 tiles para superar límites y ahorrar VRAM."""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró la carpeta: {input_folder}")

        output_folder = input_path / "processed_yolo"
        output_folder.mkdir(exist_ok=True)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]

        total_detections_folder = 0
        start_total_time = time.time()

        for img_file in image_files:
            # 1. Cargar imagen original
            image = cv2.imread(str(img_file))
            if image is None: continue

            h, w = image.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            
            # Definir coordenadas de los 4 cuadrantes
            tiles_coords = [
                (0, mid_h, 0, mid_w),     # Superior Izquierda
                (0, mid_h, mid_w, w),     # Superior Derecha
                (mid_h, h, 0, mid_w),     # Inferior Izquierda
                (mid_h, h, mid_w, w)      # Inferior Derecha
            ]
            
            # Lienzo para reconstruir el resultado visual
            full_output = np.zeros((h, w, 3), dtype=np.uint8)
            detections_in_this_image = 0

            print(f"--- Procesando: {img_file.name} ---")

            for i, (y1, y2, x1, x2) in enumerate(tiles_coords):
                tile = image[y1:y2, x1:x2]
                
                # Preprocesamiento por tile
                processed_tile = self.apply_custom_preprocessing(tile)

                # Inferencia individual por tile
                # max_det=1000 asegura que CADA pedazo de imagen pueda darte muchas rocas
                results = self.model.predict(
                    source=processed_tile, 
                    conf=0.10, 
                    max_det=300, 
                    imgsz=640,
                    save=False, 
                    verbose=False
                )
                
                res = results[0]
                num_objects = len(res.boxes)
                detections_in_this_image += num_objects

                # Dibujar máscaras en el fragmentom
                tile_visual = res.plot(
                    boxes=False, 
                    labels=False, 
                    conf=False, 
                    masks=True,
                    line_width=2
                )
                
                # Insertar el fragmento en la imagen final
                full_output[y1:y2, x1:x2] = tile_visual
                print(f"  Tile {i+1}: {num_objects} objetos detectados.")

            # Guardar la imagen compuesta
            save_path = output_folder / img_file.name
            cv2.imwrite(str(save_path), full_output)
            
            total_detections_folder += detections_in_this_image
            print(f"Total en imagen: {detections_in_this_image}\n")

        return {
            "total_time": time.time() - start_total_time,
            "output_folder": str(output_folder),
            "n_images": len(image_files),
            "total_detections": total_detections_folder
        }

if __name__ == "__main__":
    # Rutas
    onnx_path = "/home/lithos_analithics_challenge/weights/yolo_approach/onnx/rocas_segmentacionv1.onnx"
    folder_path = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    try:
        segmentor = YOLOSegmentor(onnx_path)
        print("Iniciando procesamiento con Slicing 2x2...\n")
        
        summary = segmentor.process_folder(folder_path)
        
        print("=" * 30)
        print(f"PROCESO COMPLETADO")
        print(f"Imágenes totales: {summary['n_images']}")
        print(f"Detecciones totales: {summary['total_detections']}")
        print(f"Tiempo total: {summary['total_time']:.2f}s")
        print(f"Resultados en: {summary['output_folder']}")
        print("=" * 30)
        
    except Exception as e:
        print(f"ERROR CRÍTICO: {e}")