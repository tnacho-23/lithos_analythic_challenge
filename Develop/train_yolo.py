import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN DE RUTAS ---
base_path = "/home/lithos_analithics_challenge"
original_data_root = Path(base_path) / "images/full_dataset"
processed_data_root = Path(base_path) / "images/full_dataset_processed"
weights_base_dir = Path(base_path) / "weights/yolo_approach"

# --- 2. FUNCIÓN PREPROCESS (Idéntica a DINOv2) ---
def apply_custom_preprocessing(image):
    """Bilateral Filter -> LAB -> CLAHE -> BGR"""
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_dataset(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    # Buscamos imágenes en la estructura de Roboflow (train/images, etc.)
    img_files = list(src_root.rglob("images/*/*.jpg"))
    
    if not img_files:
        print(f"ERROR: No se encontraron imágenes en {src_root}")
        return

    print(f"Iniciando preprocesamiento de {len(img_files)} imágenes...")
    
    for img_file in tqdm(img_files, desc="Procesando Dataset"):
        # split_name será 'train', 'valid' o 'test'
        split_name = img_file.parent.name 
        set_type = img_file.parent.parent.name # Carpeta del split
        
        # 1. Crear rutas espejo
        target_img_dir = dst_root / set_type / "images"
        target_lbl_dir = dst_root / set_type / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 2. Procesar Imagen
        img = cv2.imread(str(img_file))
        if img is not None:
            proc_img = apply_custom_preprocessing(img)
            cv2.imwrite(str(target_img_dir / img_file.name), proc_img)
            
        # 3. Copiar Label (Buscando en la carpeta hermana 'labels')
        label_src = img_file.parent.parent / "labels" / img_file.with_suffix('.txt').name
        if label_src.exists():
            shutil.copy(str(label_src), str(target_lbl_dir / label_src.name))

def train_lithos_yolo():
    # 1. Ejecutar preproceso si no existe la carpeta
    if not processed_data_root.exists():
        preprocess_dataset(original_data_root, processed_data_root)
        # Copiamos el data.yaml necesario para YOLO
        shutil.copy(original_data_root / "data.yaml", processed_data_root / "data.yaml")
    else:
        print(f"Usando dataset ya procesado en: {processed_data_root}")

    # 2. Cargar Modelo (Asegúrate de tener el .pt inicial o déjalo que lo descargue)
    model = YOLO("yolov8m-seg.pt") 

    # 3. Entrenar
    # NOTA: En el data.yaml de Roboflow las rutas suelen ser relativas. 
    # Asegúrate de que el data.yaml en 'processed' apunte a las carpetas correctas.
    results = model.train(
        data=str(processed_data_root / "data.yaml"), 
        epochs=100, 
        imgsz=640, 
        batch=16,
        device=0,
        project=str(weights_base_dir), 
        name="train_v3_medium_benchmark",
        workers=4,
        exist_ok=True 
    )

    # 4. Exportar
    print("Exportando a ONNX...")
    model.export(format="onnx")

if __name__ == "__main__":
    train_lithos_yolo()