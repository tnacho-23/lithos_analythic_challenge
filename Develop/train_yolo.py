import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def apply_custom_preprocessing(image):
    """Aplica el pipeline: Bilateral Filter -> LAB -> CLAHE -> BGR"""
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced

def preprocess_dataset(original_path, processed_path):
    original_path = Path(original_path)
    processed_path = Path(processed_path)
    for img_file in original_path.rglob("*"):
        if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            relative_path = img_file.relative_to(original_path)
            target_file = processed_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(img_file))
            if img is not None:
                processed_img = apply_custom_preprocessing(img)
                cv2.imwrite(str(target_file), processed_img)
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                shutil.copy(label_file, target_file.with_suffix('.txt'))

def train_lithos_yolo():
    # --- CONFIGURACIÓN DE RUTAS ---
    original_data_root = "/home/lithos_analithics_challenge/images/full_dataset"
    processed_data_root = "/home/lithos_analithics_challenge/images/full_dataset_processed"
    
    # Definimos la carpeta base de pesos como el Proyecto
    # Esto hará que YOLO cree las carpetas de entrenamiento dentro de /weights/yolo_approach/
    weights_base_dir = "/home/lithos_analithics_challenge/weights/yolo_approach"
    experiment_name = "train_v3_medium"

    # 1. Preprocesar el dataset
    if not os.path.exists(processed_data_root):
        print("Aplicando preprocesamiento Bilateral + CLAHE...")
        preprocess_dataset(original_data_root, processed_data_root)
        shutil.copy(f"{original_data_root}/data.yaml", f"{processed_data_root}/data.yaml")

    # 2. Cargar Modelo (Medium - más pesado)
    model = YOLO(f"{weights_base_dir}/yolov8m-seg.pt") 

    # 3. Entrenar
    # El output quedará en: /home/lithos_analithics_challenge/weights/yolo_approach/train_v3_medium/
    results = model.train(
        data=f"{processed_data_root}/data.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16, # Reducido de 32 a 16 por ser modelo Medium (evitar OOM)
        device=0,
        project=weights_base_dir, 
        name=experiment_name,
        workers=0,
        exist_ok=True 
    )

    # 4. Exportar a ONNX
    print("Exportando modelo a ONNX...")
    path_temp = model.export(format="onnx") 
    
    # Ruta destino junto a los pesos generados
    onnx_dest_folder = os.path.join(weights_base_dir, "onnx")
    os.makedirs(onnx_dest_folder, exist_ok=True)
    
    ruta_final_onnx = os.path.join(onnx_dest_folder, "rocas_segmentacionv1.onnx")
    
    if os.path.exists(path_temp):
        shutil.move(path_temp, ruta_final_onnx)

    print("-" * 30)
    print(f"ENTRENAMIENTO Y EXPORTACIÓN COMPLETADOS")
    print(f"Pesos PyTorch (.pt): {weights_base_dir}/{experiment_name}/weights/best.pt")
    print(f"Modelo ONNX final: {ruta_final_onnx}")
    print("-" * 30)

if __name__ == "__main__":
    train_lithos_yolo()