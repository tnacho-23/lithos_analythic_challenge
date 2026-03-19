import os
import shutil
from ultralytics import YOLO

def train_lithos_yolo():
    model = YOLO("yolov8n-seg.pt") 

    project_name = "proyectos_lithos"
    experiment_name = "segmentacion_rocas_v1"

    # 3. Entrenar
    results = model.train(
        data="/home/lithos_analithics_challenge/images/rocks_2/data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device=0,
        project=project_name, 
        name=experiment_name,
        workers = 0,
        exist_ok=True  # Permite sobreescribir si la carpeta ya existe
    )


    path_temp = model.export(format="onnx") 

    ruta_final_destino = "./weights/rocas_segmentacion.onnx"
    
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