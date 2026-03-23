import os
import time
import argparse
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

# python3 main.py --input /home/lithos_analithics_challenge/images/given_dataset/valid


# Configuración de rutas para encontrar los módulos en carpetas hermanas
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Importaciones de tus módulos
from yolo.yolo_segmentor import YOLOSegmentor
from segformer.segformer_segmentor import SegFormerSegmentor
from sam.sam_segmentor import SAM2Segmentor
from utils.rock_analytics import RockAnalytics

def run_pipeline(method_name, segmentor, image_files, output_path):
    """Función auxiliar para ejecutar un modelo específico"""
    print(f"\n>>> Iniciando Pipeline: {method_name.upper()}")
    output_path.mkdir(parents=True, exist_ok=True)
    analytics = RockAnalytics(str(output_path))
    
    for img_file in tqdm(image_files, desc=f"Procesando {method_name}"):
        try:
            start_time = time.time()
            blended, masks, count = segmentor.process_image(img_file)
            inference_time = time.time() - start_time

            if blended is not None:
                cv2.imwrite(str(output_path / f"{img_file.stem}_seg.png"), blended)
                analytics.process_image_metrics(img_file.stem, masks, inference_time)
        except Exception as e:
            print(f" Error en {img_file.name} ({method_name}): {e}")

def main():
    parser = argparse.ArgumentParser(description="Orquestador Lithos - Ejecución Triple Automática")
    parser.add_argument("--input", type=str, required=True, help="Carpeta de imágenes")
    parser.add_argument("--output", type=str, default=None, help="Carpeta base de salida")
    parser.add_argument("--method", type=str, default="all", choices=["yolo", "segformer", "sam2", "all"],
                        help="Método específico o 'all' para correr los tres")
    
    # Pesos por defecto
    parser.add_argument("--yolo_weights", default="/home/lithos_analithics_challenge/weights/yolo_approach/train_v1_medium/weights/best.pt")
    parser.add_argument("--segformer_weights", default="/home/lithos_analithics_challenge/weights/segformer_approach/final")
    parser.add_argument("--sam2_cfg", default="sam2_hiera_b+.yaml")
    parser.add_argument("--sam2_ckpt", default="/home/checkpoints/sam2_hiera_base_plus.pt")

    args = parser.parse_args()
    input_path = Path(args.input)
    base_output = Path(args.output) if args.output else input_path / "Lithos_Final_Results"

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    
    # Determinar qué modelos correr
    methods_to_run = ["yolo", "segformer", "sam2"] if args.method == "all" else [args.method]

    for m in methods_to_run:
        try:
            # 1. Instanciar el modelo actual
            if m == "yolo":
                model = YOLOSegmentor(args.yolo_weights)
            elif m == "segformer":
                model = SegFormerSegmentor(args.segformer_weights)
            elif m == "sam2":
                model = SAM2Segmentor(args.sam2_cfg, args.sam2_ckpt)
            
            # 2. Definir subcarpeta de salida (ej: /output/YOLO/)
            current_out = base_output / m.upper()
            
            # 3. Ejecutar
            run_pipeline(m, model, image_files, current_out)
            
            # Liberar memoria GPU entre modelos (Crucial para SAM2)
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[!] No se pudo ejecutar el módulo {m}: {e}")

    print(f"\n[+] PROCESO COMPLETO. Resultados en: {base_output.absolute()}")

if __name__ == "__main__":
    main()