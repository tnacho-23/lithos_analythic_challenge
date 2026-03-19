import cv2
import os
import time
from ultralytics import YOLO
from typing import Dict

class YOLOSegmentor:
    def __init__(self, model_path: str):
        # Cargamos el modelo ONNX
        self.model = YOLO(model_path, task='segment')

    def segment(self, image_path: str) -> Dict:
        """Segmentación que guarda solo las máscaras en 'result.jpg'."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

        start_time = time.time()
        
        # Realizar la inferencia
        results = self.model.predict(source=image_path, conf=0.25, save=False)
        
        inference_time = time.time() - start_time
        res = results[0]

        # --- CONFIGURACIÓN DEL PLOT (SOLO MÁSCARAS) ---
        # boxes=False: Elimina el rectángulo
        # labels=False: Elimina el nombre de la clase (0, rock, etc) y el score
        result_img = res.plot(
            boxes=False, 
            labels=False, 
            conf=False, 
            masks=True,
            line_width=2 # Grosor del contorno de la máscara
        )
        
        output_path = os.path.join(os.getcwd(), "result.jpg")
        cv2.imwrite(output_path, result_img)
        
        return {
            "inference_time": inference_time,
            "output_path": output_path,
            "n_detections": len(res.boxes)
        }

if __name__ == "__main__":
    onnx_path = "/home/lithos_analithics_challenge/weights/yolo_approach/rocas_segmentacionv2.onnx"
    img_path = "/home/lithos_analithics_challenge/images/og_dataset/valid/1706615061028_jpg.rf.f96b1f3bd3e7a02f9f076e127ba5aa4f.jpg"
    
    try:
        segmentor = YOLOSegmentor(onnx_path)
        print("Iniciando inferencia (solo máscaras)...")
        results = segmentor.segment(img_path)
        
        print("-" * 30)
        print(f"ÉXITO: Imagen guardada en {results['output_path']}")
        print(f"Rocas detectadas: {results['n_detections']}")
        print(f"Tiempo: {results['inference_time']:.4f}s")
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR: {e}")