## Lithos Analytics Inference Docker

Este contenedor permite ejecutar inferencia de segmentación de rocas utilizando YOLOv8, SegFormer y SAM2 de manera automática.

### 1. Construir la imagen
Clonar este repositorio y desde la carpeta Inference:

```
docker build -t lithos-inference .
```

### 2. Estructura de Carpetas
Para que el contenedor valide los pesos correctamente, organiza tus archivos locales así:

- weights/yolo/best.pt

- weights/segformer/final/ (debe contener config.json, etc.)

- mis_imagenes/ (fotos .jpg o .png)

### 3. Ejecución del contenedor

El Docker está configurado para usar la GPU. Asegúrate de tener instalado nvidia-container-toolkit.

* A. Correr los 3 modelos (YOLO + SegFormer + SAM2)

Si no especificas --method, el script ejecutará los tres en secuencia y guardará los resultados en una subcarpeta dentro de tus imágenes. Es necesario que montes el volumen para los pesos de yolo y segformer (contactar a ignacio.romero.a@ug.uchile.cl), el volumen con dirección a la carpeta Inference de este repositorio y el volumen con dirección a la carpeta con las imágenes a las cuales quieres realizar la inferencia.

```
docker run --gpus all \
  -v /ruta/absoluta/pesos:/app/weights \
  -v /ruta/carpeta/Inference/:/app/Inference
  -v /ruta/absoluta/mis_imagenes:/data_input \
  lithos-inference
```

- Ejemplo:

```
docker run --rm --gpus all -v C:\Users\ignac\Escritorio\Delyrium\lithos_analytics_challenge\weights:/app/weights -v C:\Users\ignac\Escritorio\Delyrium\lithos_analytics_challenge\Inference:/app/Inference -v C:\Users\ignac\Escritorio\Delyrium\lithos_analytics_challenge\images\full_dataset_processed\test\images:/data_input lithos_inference
```


* B. Correr un modelo específico
Puedes seleccionar un método usando el flag --method [yolo | segformer | sam2].

```
docker run --gpus all \
  -v /ruta/absoluta/pesos:/app/weights \
  -v /ruta/absoluta/mis_imagenes:/data_input \
  lithos-inference --input /data_input --method sam2
```

### 5. Resultados Generados
Por cada imagen y modelo, el sistema genera:
1. Imagen Segmentada: Visualización con máscaras de colores.

2. Gráfico Granulométrico: Curva acumulada con labels D20, D50, D80.

3. Histograma de Distribución: Frecuencia de tamaños en px.

4. Gráfico de Deformidad: Score de excentricidad vs tamaño.

5. CSV de Datos Crudos: Métricas geométricas por cada roca detectada.

6. JSON de Metadatos: Tiempos de inferencia y método utilizado.
