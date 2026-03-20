import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_yolo_seg(img_path, label_path):
    # 1. Load the image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    overlay = image.copy()

    # 2. Read the label file
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: continue # Skip empty lines
            
            class_id = int(parts[0])
            # Reshape the rest into (N, 2) array of (x, y)
            coords = np.array(parts[1:]).reshape(-1, 2)
            
            # 3. Scale normalized coordinates (0-1) to pixel values (w, h)
            pts = (coords * [w, h]).astype(np.int32)
            
            # 4. Draw the polygon
            color = (0, 255, 0) # Green
            cv2.fillPoly(overlay, [pts], color) 
            cv2.polylines(image, [pts], True, color, 2)

    # 5. Blend and Show
    final_img = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(final_img)
    plt.title(f"Visualizing: {os.path.basename(label_path)}")
    plt.axis('off')
    plt.show()

# --- EXECUTION ---
# Replace these with your actual local paths
my_image = "C:/Users/ignac/Escritorio/Delyrium/lithos_analytics_challenge/images/full_dataset/test/images/VID-20240802-WA0018_mp4-0009_jpg.rf.4d27af874b3559edc5a63fb6b87398c4.jpg"
my_label = "C:/Users/ignac/Escritorio/Delyrium/lithos_analytics_challenge/images/full_dataset/test/labels/VID-20240802-WA0018_mp4-0009_jpg.rf.4d27af874b3559edc5a63fb6b87398c4.txt"

visualize_yolo_seg(my_image, my_label)