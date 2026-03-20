import os
import torch
import torch.nn as nn
import numpy as np
import logging
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
base_path = "/home/lithos_analithics_challenge"
processed_data_root = Path(base_path) / "images/full_dataset_processed"
weights_dir = Path(base_path) / "weights/dino_approach"

os.makedirs(weights_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(weights_dir / "training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- MODELO DINOv2 ---
class DINOv2Segmentation(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        # Usamos ViT-S/14 pre-entrenado (Patch size = 14)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Cabeza de segmentación (Decoder simple)
        self.head = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 32 * 14 = 448 (vuelve al tamaño original)
            nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Extraer features: tokens de parches
        # n=1 extrae la última capa
        features = self.backbone.get_intermediate_layers(x, n=1)[0] 
        
        # DINOv2 devuelve [Batch, Num_Patches, Embed_Dim]
        # Para 448x448 con parches de 14x14, hay 32x32 = 1024 parches
        patch_features = features.reshape(batch_size, 32, 32, 384).permute(0, 3, 1, 2)
        return self.head(patch_features)

# --- DATASET ---
class LithosDinoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir = Path(root_dir) / split / "images"
        self.lbl_dir = Path(root_dir) / split / "labels"
        self.img_paths = list(self.img_dir.glob("*.jpg"))
        self.transform = transform
        
        if len(self.img_paths) == 0:
            logger.error(f"¡Error! No se encontraron imágenes en {self.img_dir}.")

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generar máscara desde etiquetas YOLO (segmentación polígono)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        lbl_path = self.lbl_dir / img_path.with_suffix('.txt').name
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    data = line.split()
                    if len(data) < 3: continue
                    # YOLO format: class x1 y1 x2 y2 ... (normalizado 0-1)
                    coords = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= img.shape[1]
                    coords[:, 1] *= img.shape[0]
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        
        if self.transform:
            img = self.transform(Image.fromarray(img))
            
        # La máscara debe coincidir con la salida (448x448)
        mask_res = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_res).float().unsqueeze(0)
            
        return img, mask_tensor

# --- FUNCIONES DE PÉRDIDA ---
def dice_loss(out, gt):
    smooth = 1e-6
    intersection = (out * gt).sum()
    return 1 - (2. * intersection + smooth) / (out.sum() + gt.sum() + smooth)

# --- ENTRENAMIENTO ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalización estándar para DINOv2
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = LithosDinoDataset(processed_data_root, split='train', transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = DINOv2Segmentation().to(device)
    # Solo entrenamos la "head", el backbone queda congelado
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    criterion_bce = nn.BCELoss()

    num_epochs = 1000
    logger.info(f"Iniciando entrenamiento en {device}...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Combinación de pérdidas para robustez en segmentación
            loss = dice_loss(outputs, masks) + criterion_bce(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1} Finalizada - Loss Promedio: {avg_loss:.4f}")
        
        # Guardado de seguridad
        torch.save(model.state_dict(), weights_dir / "dino_rock_v2_lastest.pth")

if __name__ == "__main__":
    train()