import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))


import time
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

from yolo_segmentor import YOLOSegmentor
from utils.rock_analytics import RockAnalytics

# python3 yolo.py --input /home/lithos_analithics_challenge/images/given_dataset/valid

def main():
    # --- CONFIGURACIÓN DE ARGUMENTOS ---
    parser = argparse.ArgumentParser(description="Pipeline de Analítica de Rocas - Lithos Challenge")
    parser.add_argument("--input", type=str, required=True, help="Ruta a la carpeta con imágenes de entrada")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida (opcional)")
    parser.add_argument("--model", type=str, 
                        default="/home/lithos_analithics_challenge/weights/yolo_approach/train_v1_medium/weights/best.pt",
                        help="Ruta al archivo .pt de YOLO")
    
    args = parser.parse_args()

    # 1. Lógica de Carpeta de Salida Dinámica
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] ERROR: La carpeta de entrada {args.input} no existe.")
        return

    # Si no se define output, creamos 'Output_Results' dentro del input
    if args.output is None:
        final_output_path = input_path / "Output" / "YOLO_results"
    else:
        final_output_path = Path(args.output)

    # Extensiones permitidas
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    # 2. Inicialización
    print(f"[*] Cargando modelo: {args.model}")
    try:
        segmentor = YOLOSegmentor(args.model)
        analytics = RockAnalytics(str(final_output_path))
    except Exception as e:
        print(f"[!] ERROR inicialización: {e}")
        return

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    print(f"[*] Procesando {len(image_files)} imágenes.")
    print(f"[*] Los resultados se guardarán en: {final_output_path}")

    # 3. Bucle de procesamiento
    for img_file in tqdm(image_files, desc="Analizando rocas"):
        try:
            start_time = time.time()

            # Inferencia con Tiling y NMS Global
            blended_img, masks, count = segmentor.process_image(img_file)
            inference_time = time.time() - start_time

            if blended_img is None: continue

            # Guardar Imagen Segmentada (Punto 1)
            # Aseguramos que el directorio exista antes de escribir
            final_output_path.mkdir(parents=True, exist_ok=True)
            save_path = final_output_path / f"{img_file.stem}_segmented.png"
            cv2.imwrite(str(save_path), blended_img)

            # Generar Reportes (Puntos 2 al 6)
            analytics.process_image_metrics(
                image_name=img_file.stem,
                masks=masks,
                inference_time=inference_time
            )

        except Exception as e:
            print(f"\n[!] Error en {img_file.name}: {e}")
            continue

    print(f"\n[+] PROCESO COMPLETADO.")
    print(f"[+] Carpeta de resultados: {final_output_path.absolute()}")

if __name__ == "__main__":
    main()