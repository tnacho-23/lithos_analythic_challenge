import os
import shutil
from ultralytics import YOLO

def train_lithos_yolo():
    model = YOLO("/home/lithos_analithics_challenge/weights/yolo_approach/yolov8n-seg.pt") 

    project_name = "proyectos_lithos"
    experiment_name = "segmentacion_rocas_v2"

    # 3. Entrenar
    results = model.train(
        data="/home/lithos_analithics_challenge/images/full_dataset/data.yaml", 
        epochs=10, 
        imgsz=640, 
        batch=32,
        device=0,
        project=project_name, 
        name=experiment_name,
        workers = 0,
        exist_ok=True  # Permite sobreescribir si la carpeta ya existe
    )


    path_temp = model.export(format="onnx") 

    ruta_final_destino = "/home/lithos_analithics_challenge/weights/yolo_approach/onnx/rocas_segmentacionv2.onnx"
    
    # Crear la carpeta de destino si no existe
    os.makedirs(os.path.dirname(ruta_final_destino), exist_ok=True)
    
    # Mover y renombrar el archivo exportado
    shutil.move(path_temp, ruta_final_destino)

    print("-" * 30)
    print(f"Entrenamiento completado en: {project_name}/{experiment_name}")
    print(f"Modelo ONNX final guardado en: {ruta_final_destino}")
    print("-" * 30)

if __name__ == "__main__":
    train_lithos_yolo()