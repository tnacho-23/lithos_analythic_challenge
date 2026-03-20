import cv2
import torch
import numpy as np
import torchvision
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2YOLOAnnotator:
    def __init__(self, model_cfg: str, checkpoint_path: str, max_area_ratio: float = 0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.max_area_ratio = max_area_ratio
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,
            points_per_batch=64, 
            pred_iou_thresh=0.5,
            stability_score_thresh=0.75,
            min_mask_region_area=50,
        )

    def apply_custom_preprocessing(self, bgr_image):
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def mask_to_yolo_polygons(self, mask, img_w, img_h):
        """Converts a binary mask to YOLO normalized polygon coordinates."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) < 3:  # Valid polygons need at least 3 points
                continue
            # Flatten and normalize
            poly = contour.reshape(-1, 2).astype(float)
            poly[:, 0] /= img_w
            poly[:, 1] /= img_h
            polygons.append(poly.reshape(-1).tolist())
        return polygons

    def process_folder(self, input_folder: str, class_id: int = 0):
        input_path = Path(input_folder)
        # Create 'labels' folder inside the input folder
        labels_folder = input_path / "labels"
        labels_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        overlap = 300 
        iou_threshold = 0.6 
        border_margin = 50

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None: continue
            h, w = image.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            
            tiles_coords = [
                (0, mid_h + overlap, 0, mid_w + overlap),
                (0, mid_h + overlap, mid_w - overlap, w),
                (mid_h - overlap, h, 0, mid_w + overlap),
                (mid_h - overlap, h, mid_w - overlap, w)
            ]
            
            all_boxes = []
            all_confs = []
            all_masks = []

            print(f"--- Annotating: {img_file.name} ---")

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                proc_tile = self.apply_custom_preprocessing(tile)
                tile_rgb = cv2.cvtColor(proc_tile, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        masks = self.mask_generator.generate(tile_rgb)
                
                for m_dict in masks:
                    if m_dict['area'] > ((y2-y1)*(x2-x1) * self.max_area_ratio):
                        continue
                    
                    tx1, ty1, tw, th = m_dict['bbox']
                    global_box = [tx1 + x1, ty1 + y1, tx1 + tw + x1, ty1 + th + y1]
                    
                    all_boxes.append(global_box)
                    all_confs.append(m_dict['predicted_iou']) 
                    all_masks.append({"segmentation": m_dict['segmentation'], "coords": (y1, y2, x1, x2)})

            if not all_boxes:
                continue

            keep_indices = torchvision.ops.nms(torch.tensor(all_boxes), torch.tensor(all_confs), iou_threshold)
            
            allowed_region_mask = np.zeros((h, w), dtype=bool)
            allowed_region_mask[border_margin:h-border_margin, border_margin:w-border_margin] = True

            # Prepare the label file
            label_path = labels_folder / f"{img_file.stem}.txt"
            
            with open(label_path, "w") as f:
                for idx in keep_indices:
                    idx = idx.item()
                    m_data = all_masks[idx]
                    m, (y1, y2, x1, x2) = m_data["segmentation"], m_data["coords"]
                    
                    # Apply margin exclusion
                    tile_allowed = allowed_region_mask[y1:y2, x1:x2]
                    final_mask = np.logical_and(m, tile_allowed)
                    
                    if np.any(final_mask):
                        # Reconstruct full-size binary mask for this specific instance
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        full_mask[y1:y2, x1:x2][final_mask] = 1
                        
                        polygons = self.mask_to_yolo_polygons(full_mask, w, h)
                        for poly in polygons:
                            poly_str = " ".join([f"{coord:.6f}" for coord in poly])
                            f.write(f"{class_id} {poly_str}\n")

            print(f"Saved {len(keep_indices)} annotations to {label_path.name}")

if __name__ == "__main__":
    CHECKPOINT = "/home/checkpoints/sam2_hiera_base_plus.pt"
    CONFIG = "sam2_hiera_b+.yaml" 
    FOLDER_PATH = "/home/lithos_analithics_challenge/images/given_dataset/train"

    segmentor = SAM2YOLOAnnotator(CONFIG, CHECKPOINT)
    segmentor.process_folder(FOLDER_PATH)