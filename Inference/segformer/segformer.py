import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import time
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

from segformer_segmentor import SegFormerSegmentor
from utils.rock_analytics import RockAnalytics

# python3 segformer.py --input /home/lithos_analithics_challenge/images/given_dataset/valid


def main():
    parser = argparse.ArgumentParser(description="Pipeline SegFormer - Lithos Challenge")
    parser.add_argument("--input", type=str, required=True, help="Carpeta de imágenes")
    parser.add_argument("--output", type=str, default=None, help="Carpeta de salida")
    parser.add_argument("--model", type=str, 
                        default="/home/lithos_analithics_challenge/weights/segformer_approach/final",
                        help="Ruta al modelo SegFormer")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path / "Output" / "SegFormer_Results"
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Inicialización
    print(f"[*] Cargando SegFormer desde: {args.model}")
    segmentor = SegFormerSegmentor(args.model)
    analytics = RockAnalytics(str(output_path))

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]

    # 2. Procesamiento
    for img_file in tqdm(image_files, desc="Procesando SegFormer"):
        start_time = time.time()
        
        blended, masks, count = segmentor.process_image(img_file)
        inference_time = time.time() - start_time
        
        if blended is not None:
            # Guardar visualización
            cv2.imwrite(str(output_path / f"{img_file.stem}_seg_blended.png"), blended)
            
            # Analítica (Genera CSV, JSON y los 3 Gráficos con labels)
            analytics.process_image_metrics(
                image_name=img_file.stem,
                masks=masks,
                inference_time=inference_time
            )

    print(f"\n[+] Analítica terminada. Revisa: {output_path}")

if __name__ == "__main__":
    main()