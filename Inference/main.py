import os
import time
import argparse
import cv2
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# DETERMINAR RAÍZ DEL PROYECTO
BASE_DIR = Path(__file__).resolve().parent.parent 
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from yolo.yolo_segmentor import YOLOSegmentor
    from segformer.segformer_segmentor import SegFormerSegmentor
    from sam.sam_segmentor import SAM2Segmentor
    from utils.rock_analytics import RockAnalytics
except ImportError as e:
    print(f"[!] Error de importación: {e}")
    sys.exit(1)

def run_pipeline(method_name, segmentor, image_files, output_path):
    print(f"\n>>> EJECUTANDO: {method_name.upper()}")
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
            print(f" [!] Error en {img_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Orquestador Lithos - Desarrollo con Volúmenes")
    
    # IMPORTANTE: Sin 'required=True', argparse usa el 'default' si no se provee el flag
    parser.add_argument("--input", type=str, default="/data_input", help="Carpeta de imágenes")
    parser.add_argument("--output", type=str, default=None, help="Carpeta de salida")
    parser.add_argument("--method", type=str, default="all", choices=["yolo", "segformer", "sam2", "all"])
    
    parser.add_argument("--yolo_weights", default="/app/weights/yolo/best.pt")
    parser.add_argument("--segformer_weights", default="/app/weights/segformer/final")
    parser.add_argument("--sam2_cfg", default="sam2_hiera_b+.yaml")
    parser.add_argument("--sam2_ckpt", default="/app/checkpoints/sam2_hiera_base_plus.pt")

    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"[!] ERROR: No existe la carpeta {input_path}")
        return

    # Solo buscamos imágenes si la carpeta existe
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not image_files:
        print(f"[!] No se encontraron imágenes en {input_path}")
        return

    base_output = Path(args.output) if args.output else input_path / "Lithos_Final_Results"
    methods_to_run = ["yolo", "segformer", "sam2"] if args.method == "all" else [args.method]

    print(f"\n" + "="*50)
    print(f"{'LITHOS ANALYTICS PIPELINE':^50}")
    print(f"{'Target: ' + str(input_path):^50}")
    print("="*50)

    for m in methods_to_run:
        w_path = args.yolo_weights if m == "yolo" else args.segformer_weights if m == "segformer" else args.sam2_ckpt
        if not os.path.exists(w_path):
            print(f"[!] Saltando {m.upper()}: Pesos no encontrados en {w_path}")
            continue

        try:
            print(f"[*] Cargando {m.upper()}...")
            if m == "yolo": model = YOLOSegmentor(args.yolo_weights)
            elif m == "segformer": model = SegFormerSegmentor(str(args.segformer_weights))
            elif m == "sam2": model = SAM2Segmentor(args.sam2_cfg, args.sam2_ckpt)
            
            run_pipeline(m, model, image_files, base_output / m.upper())
            
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"[!] Error en módulo {m}: {e}")

    print(f"\n[+] Proceso Terminado. Resultados en: {base_output.absolute()}")

if __name__ == "__main__":
    main()