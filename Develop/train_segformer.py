import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import evaluate # pip install evaluate
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    TrainingArguments, 
    Trainer
)

# --- 1. CONFIGURACIÓN ---
DATA_ROOT = "/home/lithos_analithics_challenge/images/full_dataset_processed"
OUTPUT_DIR = "/home/lithos_analithics_challenge/weights/segformer_approach"
CHECKPOINT = "nvidia/mit-b1" # Mismo backbone, nueva cabeza.

# Definir las 3 etiquetas
id2label = {0: "background", 1: "rock_body", 2: "rock_border"}
label2id = {"background": 0, "rock_body": 1, "rock_border": 2}
NUM_LABELS = len(id2label)

# --- 2. DATASET (Ajustado para leer las nuevas máscaras) ---
class Lithos3ClassDataset(Dataset):
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
        # Asegurarse de leer la máscara como escala de grises para conservar valores 0,1,2
        segmentation_map = Image.open(mask_path) 

        inputs = self.processor(image, segmentation_map, return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return inputs

# --- 3. MÉTRICAS (mIoU para 3 clases) ---
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits = torch.from_numpy(logits)
        
        logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits.numpy()
        
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=NUM_LABELS,
            ignore_index=255,
            reduce_labels=False,
        )
        
        # --- CORRECCIÓN AQUÍ ---
        # Extraemos el mIoU y la exactitud media, convirtiéndolos a float de Python
        mean_iou = float(metrics["mean_iou"])
        mean_accuracy = float(metrics["mean_accuracy"])
        
        # Convertimos los arrays de IoU por categoría a una lista de floats estándar
        per_category_iou = metrics.pop("per_category_iou").tolist()

        # Creamos el diccionario final asegurando que TODO sea serializable
        results = {
            "mean_iou": mean_iou,
            "mean_accuracy": mean_accuracy,
        }
        
        # Añadimos el IoU de cada clase (Fondo, Cuerpo, Borde)
        for i, v in enumerate(per_category_iou):
            results[f"iou_{id2label[i]}"] = float(v)

        return results

# --- 4. PREPARACIÓN DEL MODELO ---
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)

train_dataset = Lithos3ClassDataset(
    img_dir=f"{DATA_ROOT}/train/images",
    mask_dir=f"{DATA_ROOT}/train/masks_segformer",
    processor=processor
)

val_dataset = Lithos3ClassDataset(
    img_dir=f"{DATA_ROOT}/valid/images",
    mask_dir=f"{DATA_ROOT}/valid/masks_segformer",
    processor=processor
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SegformerForSemanticSegmentation.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    use_safetensors=True  
)
class_weights = torch.tensor([1.0, 1.0, 5.0]).to(device)
model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

# --- 5. ARGUMENTOS DE ENTRENAMIENTO ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=6e-5,
    num_train_epochs=150,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch", 
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

# --- 6. INICIAR ENTRENAMIENTO ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Iniciando entrenamiento de SegFormer (3 Clases: Fondo, Cuerpo, Borde)...")
trainer.train()

# Guardar modelo final
trainer.save_model(f"{OUTPUT_DIR}/final")
processor.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Modelo guardado en: {OUTPUT_DIR}/final")