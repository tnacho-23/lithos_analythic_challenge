import cv2
import torch
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Segmentor:
    def __init__(self, model_cfg: str, checkpoint_path: str, max_area_ratio: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.max_area_ratio = 0.3  # Ejemplo: 0.5 significa "máximo 50% de la imagen"
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=48, 
            points_per_batch=128, 
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )

        
        '''
        pred_iou_thresh (Umbral de Confianza)
            Este es el filtro principal de calidad. SAM 2 predice qué tan buena es la máscara que acaba de generar (Intersection over Union)

        stability_score_thresh (Umbral de Estabilidad)
            Mide qué tanto cambia la máscara si alteras ligeramente el umbral de corte binario. Es vital para rocas con bordes difusos o cubiertas de polvo.

        min_mask_region_area (Filtro de Tamaño)
            Este no es un umbral probabilístico, sino físico (en píxeles)
        
        '''

    def preprocess_image(self, bgr_image):
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def process_folder(self, input_folder: str):
        input_path = Path(input_folder)
        output_folder = input_path / "processed_sam2"
        output_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None: continue
            
            # Calcular área total para el filtro
            h, w = image.shape[:2]
            total_pixels = h * w

            processed_img = self.preprocess_image(image)
            image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks = self.mask_generator.generate(image_rgb)

            annotated_image = image.copy()
            count = 0
            
            for mask in masks:
                # --- FILTRO DE TAMAÑO MÁXIMO ---
                # mask['area'] ya viene calculado por el generador de SAM 2
                if mask['area'] > (total_pixels * self.max_area_ratio):
                    continue 
                
                count += 1
                m = mask['segmentation']
                color = np.random.randint(0, 255, (3,)).tolist()
                annotated_image[m] = (annotated_image[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

            cv2.imwrite(str(output_folder / img_file.name), annotated_image)
            print(f"OK: {img_file.name} (Mostrando {count} de {len(masks)} detectadas)")

if __name__ == "__main__":
    CHECKPOINT = "/home/checkpoints/sam2_hiera_base_plus.pt"
    CONFIG = "sam2_hiera_b+.yaml" 
    FOLDER_PATH = "/home/lithos_analithics_challenge/images/given_dataset/valid"

    # Se instancia con un filtro de máximo 30% del tamaño total de la imagen
    segmentor = SAM2Segmentor(CONFIG, CHECKPOINT)
    segmentor.process_folder(FOLDER_PATH)