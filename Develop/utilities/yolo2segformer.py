import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def yolo_to_semantic_mask(img_dir, label_dir, output_mask_dir, img_size=(640, 640)):
    """
    Convierte etiquetas de segmentación YOLO (.txt) a máscaras semánticas (.png).
    Todas las rocas se pintan con valor de píxel 1 (clase 'roca').
    """
    img_path = Path(img_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_mask_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in img_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    for img_file in tqdm(image_files, desc="Convirtiendo a Máscaras"):
        # 1. Cargar imagen para obtener dimensiones reales
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]

        # 2. Crear máscara negra (fondo = 0)
        # Usamos uint8 para que los valores sean 0 y 1
        mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Buscar archivo de etiqueta correspondiente
        label_file = lbl_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # parts[0] es la clase (usualmente 0 para rocas)
                # parts[1:] son los puntos x, y, x, y... normalizados (0-1)
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                
                # Des-normalizar coordenadas al tamaño real de la imagen
                coords[:, 0] *= w
                coords[:, 1] *= h
                
                # Convertir a enteros para OpenCV
                poly = coords.astype(np.int32)
                
                # 4. Pintar el polígono en la máscara con valor 1 (Clase Roca)
                cv2.fillPoly(mask, [poly], color=1)

        # 5. Guardar la máscara como PNG (sin compresión con pérdida)
        # Nota: La imagen se verá negra al abrirla porque los valores son 0 y 1.
        # Si quieres verla, multiplica 'mask * 255' antes de guardar (solo para visualización).
        mask_filename = out_path / f"{img_file.stem}.png"
        cv2.imwrite(str(mask_filename), mask)

if __name__ == "__main__":
    # --- CONFIGURA TUS RUTAS AQUÍ ---
    base_data = "/home/lithos_analithics_challenge/images/full_dataset_processed"
    
    # Procesar split de entrenamiento
    yolo_to_semantic_mask(
        img_dir=f"{base_data}/train/images",
        label_dir=f"{base_data}/train/labels",
        output_mask_dir=f"{base_data}/train/masks_png"
    )
    
    # Procesar split de validación
    yolo_to_semantic_mask(
        img_dir=f"{base_data}/valid/images",
        label_dir=f"{base_data}/valid/labels",
        output_mask_dir=f"{base_data}/valid/masks_png"
    )
    
    print("\nPROCESO COMPLETADO. Las máscaras están en la carpeta 'masks_png'.")