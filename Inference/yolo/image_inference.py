import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from typing import Dict

class YOLOSegmentor:
    def __init__(self, model_path: str):
        # Cargamos el modelo ONNX
        self.model = YOLO(model_path, task='segment')

    def apply_custom_preprocessing(self, bgr_image):
        """Pipeline idéntico al entrenamiento: Bilateral + CLAHE"""
        # 1. Filtro Bilateral
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        # 2. Conversión a LAB y CLAHE en canal L
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # 3. Regresar a BGR
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def segment(self, image_path: str) -> Dict:
        """Segmentación con preprocesamiento que guarda máscaras en 'result.jpg'."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")

        # 1. Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo decodificar la imagen en: {image_path}")

        # 2. Preprocesar (Bilateral + CLAHE)
        processed_img = self.apply_custom_preprocessing(image)

        # 3. Realizar la inferencia sobre la imagen procesada
        start_time = time.time()
        results = self.model.predict(source=processed_img, conf=0.25, save=False, verbose=False)
        inference_time = time.time() - start_time
        
        res = results[0]

        # 4. Generar Plot (Solo máscaras sobre la imagen procesada)
        result_img = res.plot(
            boxes=False, 
            labels=False, 
            conf=False, 
            masks=True,
            line_width=2
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
        print("Iniciando inferencia con preprocesamiento (solo máscaras)...")
        results = segmentor.segment(img_path)
        
        print("-" * 30)
        print(f"ÉXITO: Imagen guardada en {results['output_path']}")
        print(f"Rocas detectadas: {results['n_detections']}")
        print(f"Tiempo de inferencia: {results['inference_time']:.4f}s")
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR: {e}")