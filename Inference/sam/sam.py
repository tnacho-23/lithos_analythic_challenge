import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))


import argparse
import time
import cv2
from pathlib import Path
from tqdm import tqdm
from sam_segmentor import SAM2Segmentor
from utils.rock_analytics import RockAnalytics

# python3 sam.py --input /home/lithos_analithics_challenge/images/given_dataset/valid


def main():
    parser = argparse.ArgumentParser(description="Pipeline SAM2 - Lithos Challenge")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cfg", type=str, default="sam2_hiera_b+.yaml")
    parser.add_argument("--checkpoint", type=str, default="/home/checkpoints/sam2_hiera_base_plus.pt")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = args.output if args.output else input_path / "Output" / "SAM2_Results"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"[*] Inicializando SAM2...")
    segmentor = SAM2Segmentor(args.cfg, args.checkpoint)
    analytics = RockAnalytics(str(output_path))

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]

    for img_file in tqdm(image_files, desc="Procesando con SAM2"):
        start_time = time.time()
        blended, masks, count = segmentor.process_image(img_file)
        inference_time = time.time() - start_time
        
        if blended is not None:
            cv2.imwrite(str(Path(output_path) / f"{img_file.stem}_sam2_blended.png"), blended)
            analytics.process_image_metrics(img_file.stem, masks, inference_time)

    print(f"\n[+] Analítica SAM2 completada en: {output_path}")

if __name__ == "__main__":
    main()