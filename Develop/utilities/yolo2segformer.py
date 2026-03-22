import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def yolo_to_3class_mask(img_dir, label_dir, output_mask_dir, border_thickness=7):
    """
    Convierte etiquetas YOLO a máscaras de 3 clases:
    0: Fondo
    1: Cuerpo de la roca
    2: Borde de la roca (para evitar que se fusionen)
    """
    img_path = Path(img_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_mask_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in img_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    for img_file in tqdm(image_files, desc="Generando Máscaras 3-Clases"):
        # 1. Obtener dimensiones
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]

        # 2. Crear máscara base (0 = Fondo)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Leer etiquetas
        label_file = lbl_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            polygons = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # Des-normalizar coordenadas
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                poly = coords.astype(np.int32)
                polygons.append(poly)
            
            # --- CAPA 1: CUERPO (Relleno) ---
            for poly in polygons:
                cv2.fillPoly(mask, [poly], color=1)
            
            # --- CAPA 2: BORDES (Contorno) ---
            # Pintamos los bordes al final para que queden sobre los cuerpos
            # y aseguren la separación entre rocas adyacentes.
            for poly in polygons:
                # isClosed=True para cerrar el polígono, grosor ajustable
                cv2.polylines(mask, [poly], isClosed=True, color=2, thickness=border_thickness)

        # 4. Guardar
        mask_filename = out_path / f"{img_file.stem}.png"
        cv2.imwrite(str(mask_filename), mask)

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    base_data = "/home/lithos_analithics_challenge/images/full_dataset_processed"
    THICKNESS = 25
    
    # Procesar Train
    yolo_to_3class_mask(
        img_dir=f"{base_data}/train/images",
        label_dir=f"{base_data}/train/labels",
        output_mask_dir=f"{base_data}/train/masks_segformer",
        border_thickness=THICKNESS
    )
    
    # Procesar Validation
    yolo_to_3class_mask(
        img_dir=f"{base_data}/valid/images",
        label_dir=f"{base_data}/valid/labels",
        output_mask_dir=f"{base_data}/valid/masks_segformer",
        border_thickness=THICKNESS
    )
    
    print(f"\n¡LISTO! Máscaras generadas con bordes de {THICKNESS}px.")