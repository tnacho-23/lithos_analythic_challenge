import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def train_lithos_yolo():
    # --- CONFIGURACIÓN DE RUTAS ---
    # Asumimos que los datos ya están listos en esta ubicación
    data_root = "/home/lithos_analithics_challenge/images/full_dataset_processed"
    
    # Directorio base para pesos y experimentos
    weights_base_dir = "/home/lithos_analithics_challenge/weights/yolo_approach"
    experiment_name = "train_v2_medium"
    yaml_path = f"{data_root}/data.yaml"

    # 1. Cargar Modelo (Medium)
    # Nota: Asegúrate de que el .pt base exista en esa ruta o usa "yolov8m-seg.pt" para descarga automática
    model = YOLO(f"{weights_base_dir}/yolov8m-seg.pt") 

    # 2. Entrenar
    # El output quedará en: /home/lithos_analithics_challenge/weights/yolo_approach/train_v3_medium/
    model.train(
        data=yaml_path, 
        epochs=100, 
        imgsz=640, 
        batch=16, 
        device=0,
        project=weights_base_dir, 
        name=experiment_name,
        workers=0,
        exist_ok=True 
    )

    # 3. Exportar a ONNX
    print("Exportando modelo a ONNX...")
    path_temp = model.export(format="onnx") 
    
    # Organizar salida de ONNX
    onnx_dest_folder = os.path.join(weights_base_dir, "onnx")
    os.makedirs(onnx_dest_folder, exist_ok=True)
    ruta_final_onnx = os.path.join(onnx_dest_folder, "rocas_segmentacionv2.onnx")
    
    if os.path.exists(path_temp):
        shutil.move(path_temp, ruta_final_onnx)

    print("-" * 30)
    print("ENTRENAMIENTO Y EXPORTACIÓN COMPLETADOS")
    print(f"Pesos PyTorch (.pt): {weights_base_dir}/{experiment_name}/weights/best.pt")
    print(f"Modelo ONNX final: {ruta_final_onnx}")
    print("-" * 30)

if __name__ == "__main__":
    train_lithos_yolo()