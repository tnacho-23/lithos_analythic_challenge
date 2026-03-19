import cv2
import os
import time
from ultralytics import YOLO
from typing import Dict, List
from pathlib import Path

class YOLOSegmentor:
    def __init__(self, model_path: str):
        # Cargamos el modelo (funciona con .pt o .onnx)
        self.model = YOLO(model_path, task='segment')

    def process_folder(self, input_folder: str) -> Dict:
        """Procesa todas las imágenes de una carpeta y las guarda en 'processed'."""
        
        # Validar carpeta de entrada
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró la carpeta: {input_folder}")

        # Crear carpeta 'processed' dentro de la carpeta de entrada (o donde prefieras)
        output_folder = input_path / "processed_yolo"
        output_folder.mkdir(exist_ok=True)

        # Extensiones válidas de imagen
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]

        total_detections = 0
        start_total_time = time.time()

        for img_file in image_files:
            # Inferencia
            results = self.model.predict(source=str(img_file), conf=0.25, save=False, verbose=False)
            res = results[0]

            # Plot de solo máscaras
            result_img = res.plot(
                boxes=False, 
                labels=False, 
                conf=False, 
                masks=True,
                line_width=2
            )

            # Guardar con el mismo nombre en la carpeta processed
            save_path = output_folder / img_file.name
            cv2.imwrite(str(save_path), result_img)
            
            total_detections += len(res.boxes)
            print(f"Procesada: {img_file.name} -> {len(res.boxes)} rocas")

        return {
            "total_time": time.time() - start_total_time,
            "output_folder": str(output_folder),
            "n_images": len(image_files),
            "total_detections": total_detections
        }

if __name__ == "__main__":
    onnx_path = "/home/lithos_analithics_challenge/weights/yolo_approach/onnx/rocas_segmentacionv2.onnx"
    folder_path = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    try:
        segmentor = YOLOSegmentor(onnx_path)
        print(f"Iniciando procesamiento de carpeta: {folder_path}\n")
        
        summary = segmentor.process_folder(folder_path)
        
        print("-" * 30)
        print(f"FINALIZADO")
        print(f"Imágenes procesadas: {summary['n_images']}")
        print(f"Total rocas detectadas: {summary['total_detections']}")
        print(f"Carpeta de salida: {summary['output_folder']}")
        print(f"Tiempo total: {summary['total_time']:.2f}s")
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR: {e}")