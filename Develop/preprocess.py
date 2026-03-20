import os
import cv2
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
base_path = "/home/lithos_analithics_challenge"
original_data_root = Path(base_path) / "images/full_dataset"
processed_data_root = Path(base_path) / "images/full_dataset_processed"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def apply_custom_preprocessing(image):
    # Suavizado preservando bordes (ideal para rocas)
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Mejora de contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def main():
    logger.info(f"Iniciando preprocesamiento: {original_data_root} -> {processed_data_root}")
    
    for split in ['train', 'valid', 'test']:
        src_img_dir = original_data_root / split / "images"
        src_lbl_dir = original_data_root / split / "labels"
        
        dst_img_dir = processed_data_root / split / "images"
        dst_lbl_dir = processed_data_root / split / "labels"
        
        if not src_img_dir.exists():
            logger.warning(f"Split {split} no encontrado. Saltando...")
            continue
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img_files = list(src_img_dir.glob("*.jpg"))
        for img_file in tqdm(img_files, desc=f"Procesando {split}"):
            # 1. Procesar Imagen
            img = cv2.imread(str(img_file))
            if img is not None:
                proc = apply_custom_preprocessing(img)
                cv2.imwrite(str(dst_img_dir / img_file.name), proc)
            
            # 2. Copiar Label (sin cambios, solo se mueven a la nueva ruta)
            lbl_file = src_lbl_dir / img_file.with_suffix('.txt').name
            if lbl_file.exists():
                shutil.copy(str(lbl_file), str(dst_lbl_dir / lbl_file.name))

    logger.info("¡Preprocesamiento completado!")

if __name__ == "__main__":
    main()