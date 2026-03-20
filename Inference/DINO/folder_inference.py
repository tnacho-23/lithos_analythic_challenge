import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from PIL import Image
from torchvision import transforms

# --- 1. ARQUITECTURA REFORZADA (Debe ser idéntica al entrenamiento) ---
class DINOv2Segmentation(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Esta es la estructura que generó los pesos del .pth
        self.head = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        
        # Tomamos los últimos 1024 tokens (32x32) descartando Class y Register tokens
        patch_features = features[:, -1024:, :] 
        patch_features = patch_features.reshape(batch_size, 32, 32, 384).permute(0, 3, 1, 2)
        return self.head(patch_features)

class DINOSegmentor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOv2Segmentation().to(self.device)
        
        # Cargar pesos (Ahora coincidirán las llaves y formas)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Modelo cargado exitosamente desde {model_path}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def apply_custom_preprocessing(self, bgr_image):
        smoothed = cv2.bilateralFilter(bgr_image, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process_folder(self, input_folder: str):
        input_path = Path(input_folder)
        output_folder = input_path / "processed_dino"
        output_folder.mkdir(exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        overlap = 300 
        iou_threshold = 0.4 # Ajustado para rocas solapadas

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

            print(f"--- Procesando DINOv2: {img_file.name} ---")

            for (y1, y2, x1, x2) in tiles_coords:
                tile = image[y1:y2, x1:x2]
                proc_tile = self.apply_custom_preprocessing(tile)
                
                input_tensor = self.transform(proc_tile).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output_mask = self.model(input_tensor).squeeze().cpu().numpy()
                
                output_mask = cv2.resize(output_mask, (x2 - x1, y2 - y1))
                
                # Umbral de detección (como ya entrenó con Dice, 0.5 debería ser sólido)
                binary_mask = (output_mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) < 150: continue # Filtrar ruido
                    
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    conf = np.mean(output_mask[by:by+bh, bx:bx+bw])
                    
                    global_box = [bx + x1, by + y1, bx + bw + x1, by + bh + y1]
                    
                    inst_mask = np.zeros(output_mask.shape, dtype=np.uint8)
                    cv2.drawContours(inst_mask, [cnt], -1, 1, -1)
                    
                    all_boxes.append(global_box)
                    all_confs.append(conf)
                    all_masks.append({"mask": inst_mask, "coords": (y1, y2, x1, x2)})

            if not all_boxes:
                print(f"No se detectaron rocas en {img_file.name}")
                continue

            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            confs_t = torch.tensor(all_confs, dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_t, confs_t, iou_threshold)
            
            mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            final_count = 0

            for idx in keep_indices:
                idx = idx.item()
                m_data = all_masks[idx]
                m, (y1, y2, x1, x2) = m_data["mask"], m_data["coords"]
                
                color = np.random.randint(60, 255, (3,)).tolist()
                tile_area = mask_canvas[y1:y2, x1:x2]
                tile_area[m > 0] = color
                final_count += 1

            blended = cv2.addWeighted(image, 0.7, mask_canvas, 0.35, 0)
            cv2.imwrite(str(output_folder / img_file.name), blended)
            print(f"Rocas detectadas: {final_count}")

if __name__ == "__main__":
    dino_weights = "/home/lithos_analithics_challenge/weights/dino_approach/dino_rock_head_final.pth"
    valid_folder = "/home/lithos_analithics_challenge/images/given_dataset/valid"
    
    try:
        segmentor = DINOSegmentor(dino_weights)
        segmentor.process_folder(valid_folder)
    except Exception as e:
        import traceback
        traceback.print_exc()