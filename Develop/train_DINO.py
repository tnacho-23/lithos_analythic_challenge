import os
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# --- 1. CONFIGURACIÓN DE RUTAS ---
base_path = "/home/lithos_analithics_challenge"
original_data_root = Path(base_path) / "images/full_dataset"
processed_data_root = Path(base_path) / "images/full_dataset_processed"
weights_dir = Path(base_path) / "weights/dino_approach"

os.makedirs(weights_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(weights_dir / "training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 2. PREPROCESAMIENTO (Imágenes + Labels) ---
def apply_custom_preprocessing(image):
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_dataset(src, dst):
    logger.info(f"Iniciando preprocesamiento desde {src}...")
    # Buscamos en train, valid y test según tu yaml
    for split in ['train', 'valid', 'test']:
        src_img_dir = src / split / "images"
        src_lbl_dir = src / split / "labels"
        
        dst_img_dir = dst / split / "images"
        dst_lbl_dir = dst / split / "labels"
        
        if not src_img_dir.exists(): continue
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img_files = list(src_img_dir.glob("*.jpg"))
        for img_file in tqdm(img_files, desc=f"Procesando {split}"):
            # 1. Procesar Imagen
            img = cv2.imread(str(img_file))
            if img is not None:
                proc = apply_custom_preprocessing(img)
                cv2.imwrite(str(dst_img_dir / img_file.name), proc)
            
            # 2. Copiar Label
            lbl_file = src_lbl_dir / img_file.with_suffix('.txt').name
            if lbl_file.exists():
                shutil.copy(str(lbl_file), str(dst_lbl_dir / lbl_file.name))

# --- 3. MODELO DINOv2 (Fix de Tokens) ---
class DINOv2Segmentation(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
        
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
        # Tomamos los últimos 1024 tokens (32x32)
        patch_features = features[:, -1024:, :] 
        patch_features = patch_features.reshape(batch_size, 32, 32, 384).permute(0, 3, 1, 2)
        return self.head(patch_features)

# --- 4. DATASET ---
class LithosDinoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir = Path(root_dir) / split / "images"
        self.lbl_dir = Path(root_dir) / split / "labels"
        self.img_paths = list(self.img_dir.glob("*.jpg"))
        self.transform = transform
        
        if len(self.img_paths) == 0:
            logger.error(f"¡Error! No hay imágenes en {self.img_dir}")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generar máscara
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        lbl_path = self.lbl_dir / img_path.with_suffix('.txt').name
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    coords = np.array(line.split()[1:], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= img.shape[1]
                    coords[:, 1] *= img.shape[0]
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        
        if self.transform:
            img = self.transform(Image.fromarray(img))
            mask_res = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_res).float().unsqueeze(0)
            
        return img, mask_tensor

# --- 5. ENTRENAMIENTO ---
def train_lithos_dino():
    if not processed_data_root.exists():
        preprocess_dataset(original_data_root, processed_data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Entrenamos con el split 'train'
    dataset = LithosDinoDataset(processed_data_root, split='train', transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DINOv2Segmentation().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    
    # Loss híbrida para "despertar" al modelo
    criterion_dice = lambda out, gt: 1 - (2.*(out*gt).sum() + 1.) / (out.sum() + gt.sum() + 1.)
    criterion_bce = nn.BCELoss()

    num_epochs = 40
    logger.info(f"Entrenando en {device} durante {num_epochs} epocas...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, masks in pbar:
            if masks.max() == 0:
                logger.warning("¡Advertencia! Máscara vacía detectada, saltando batch...")
                continue
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion_dice(outputs, masks) + criterion_bce(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mask_val": f"{outputs.max().item():.2f}"})
        
        logger.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
        torch.save(model.state_dict(), weights_dir / "dino_rock_head_final.pth")

if __name__ == "__main__":
    train_lithos_dino()