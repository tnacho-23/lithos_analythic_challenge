import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from scipy.interpolate import interp1d

class RockAnalytics:
    def __init__(self, output_folder: str):
        self.output_base = Path(output_folder)
        self.output_base.mkdir(parents=True, exist_ok=True)

    def process_image_metrics(self, image_name: str, masks: list, inference_time: float):
        """
        Calcula métricas de cada roca y genera los reportes para una imagen.
        """
        if not masks:
            print(f"Sin detecciones para {image_name}")
            return

        # 1. Extraer datos geométricos de cada máscara
        rock_data = []
        for idx, mask in enumerate(masks):
            # Asegurar que la máscara sea binaria
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            if area < 10: continue # Filtro de ruido mínimo

            # Diámetro equivalente (Punto 2 y 3)
            equiv_diameter = np.sqrt(4 * area / np.pi)

            # Score de Deformidad (Punto 4: Excentricidad)
            # Usamos elipse ajustada para obtener ejes mayor y menor
            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                # Excentricidad: 0 es un círculo perfecto, tiende a 1 si es muy alargada
                major_axis = max(MA, ma)
                minor_axis = min(MA, ma)
                eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
            else:
                eccentricity, aspect_ratio = 0, 1

            rock_data.append({
                "rock_id": idx,
                "area_px": area,
                "diameter_px": equiv_diameter,
                "eccentricity": round(eccentricity, 4),
                "aspect_ratio": round(aspect_ratio, 4)
            })

        df = pd.DataFrame(rock_data)
        
        # 2. Generar Gráficos y CSVs
        d_stats = self._generate_granulometry(df, image_name)
        self._generate_distribution_plot(df, image_name)
        self._generate_deformity_plot(df, image_name)
        
        # 3. Guardar Metadatos (Punto 6)
        metadata = {
            "image_name": image_name,
            "inference_time_ms": round(inference_time * 1000, 2),
            "total_rocks_detected": len(df),
            "method": "YOLOv8-Segmentation + Tiling + NMS Global",
            "granulometry_px": d_stats
        }
        self._save_json(image_name, metadata)
        
        # Guardar datos crudos tabulares (Punto 5)
        df.to_csv(self.output_base / f"{image_name}_raw_data.csv", index=False)

    def _generate_granulometry(self, df, name):
            """Gráfico de Curva Granulométrica con Labels numéricos (Punto 2)"""
            sizes = np.sort(df['diameter_px'].values)
            percent_passing = np.arange(1, len(sizes) + 1) / len(sizes) * 100
            
            # Calcular D20, D50, D80 mediante interpolación
            f = interp1d(percent_passing, sizes, bounds_error=False, fill_value="extrapolate")
            d_points = {
                "D20": round(float(f(20)), 2),
                "D50": round(float(f(50)), 2),
                "D80": round(float(f(80)), 2)
            }

            plt.figure(figsize=(10, 7))
            plt.plot(sizes, percent_passing, marker='.', color='royalblue', label='Curva Granulométrica', alpha=0.6)
            
            # Colores para cada marcador
            colors = {'D20': 'red', 'D50': 'green', 'D80': 'darkorange'}
            
            for p_name, val in d_points.items():
                percent = int(p_name[1:]) # extrae 20, 50 o 80
                color = colors[p_name]
                
                # Dibujar líneas guía
                plt.axvline(val, color=color, linestyle='--', alpha=0.5)
                plt.axhline(percent, color=color, linestyle='--', alpha=0.3)
                
                # AGREGAR LABELS (Etiquetas de texto)
                # Colocamos el texto ligeramente desplazado para que no tape la curva
                plt.text(val, percent + 2, f"{p_name}: {val} px", 
                        color=color, fontweight='bold', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))
                
                # Punto de intersección
                plt.plot(val, percent, 'o', color=color)

            plt.title(f"Análisis Granulométrico - {name}\nCortes D20, D50, D80", fontsize=14)
            plt.xlabel("Tamaño Equivalente (píxeles)", fontsize=12)
            plt.ylabel("% Pasante Acumulado", fontsize=12)
            plt.ylim(0, 105)
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.legend(loc='lower right')
            
            plt.savefig(self.output_base / f"{name}_granulometria.png", dpi=150)
            plt.close()
            
            return d_points

    def _generate_distribution_plot(self, df, name):
        """Gráfico de Distribución / Histograma (Punto 3)"""
        plt.figure(figsize=(8, 6))
        df['diameter_px'].hist(bins=25, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Distribución de Tamaño de Rocas - {name}")
        plt.xlabel("Diámetro (px)")
        plt.ylabel("Frecuencia (Cantidad)")
        plt.savefig(self.output_base / f"{name}_distribucion.png")
        plt.close()

    def _generate_deformity_plot(self, df, name):
        """Gráfico de Deformidad / Excentricidad (Punto 4)"""
        plt.figure(figsize=(8, 6))
        plt.scatter(df['diameter_px'], df['eccentricity'], c=df['eccentricity'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Score Excentricidad')
        plt.title(f"Deformidad vs Tamaño - {name}\n(0=Circular, 1=Alargada)")
        plt.xlabel("Tamaño (px)")
        plt.ylabel("Excentricidad")
        plt.savefig(self.output_base / f"{name}_deformidad.png")
        plt.close()

    def _save_json(self, name, data):
        with open(self.output_base / f"{name}_metadatos.json", 'w') as f:
            json.dump(data, f, indent=4)