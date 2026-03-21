import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    TrainingArguments, 
    Trainer
)
import numpy as np

# --- 1. CONFIGURACIÓN DE RUTAS ---
DATA_ROOT = "/home/lithos_analithics_challenge/images/full_dataset_processed"
OUTPUT_DIR = "/home/lithos_analithics_challenge/weights/segformer_approach"
CHECKPOINT = "nvidia/mit-b0"  # mit-b0 es el más rápido. Usa b2 o b5 para más precisión.

# --- 2. DEFINICIÓN DEL DATASET ---
class LithosDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_name = os.path.splitext(self.img_names[idx])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        segmentation_map = Image.open(mask_path) # Valores 0 y 1

        # El processor redimensiona a 512x512 por defecto y normaliza
        inputs = self.processor(image, segmentation_map, return_tensors="pt")
        
        # Eliminar dimensión de batch extra [1, C, H, W] -> [C, H, W]
        for k,v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return inputs

# --- 3. PREPARACIÓN ---
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)

train_dataset = LithosDataset(
    img_dir=f"{DATA_ROOT}/train/images",
    mask_dir=f"{DATA_ROOT}/train/masks_png",
    processor=processor
)

val_dataset = LithosDataset(
    img_dir=f"{DATA_ROOT}/valid/images",
    mask_dir=f"{DATA_ROOT}/valid/masks_png",
    processor=processor
)

# Definir etiquetas
id2label = {0: "background", 1: "rock"}
label2id = {"background": 0, "rock": 1}

model = SegformerForSemanticSegmentation.from_pretrained(
    CHECKPOINT,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    use_safetensors=True 
)

# --- 4. ARGUMENTOS DE ENTRENAMIENTO ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=6e-5,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    # CAMBIO AQUÍ: de 'evaluation_strategy' a 'eval_strategy'
    eval_strategy="epoch", 
    
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False, # Importante para que no borre las imágenes del dataset
)

# --- 5. INICIAR ENTRENAMIENTO ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Iniciando entrenamiento de SegFormer...")
trainer.train()

# Guardar modelo final y procesador
trainer.save_model(f"{OUTPUT_DIR}/final_rock_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_rock_model")
print(f"Modelo guardado en: {OUTPUT_DIR}/final_rock_model")