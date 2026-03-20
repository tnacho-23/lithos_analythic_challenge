from ultralytics.data.converter import convert_coco

# This will convert your COCO JSON to YOLO segmentation format
convert_coco(labels_dir='C:/Users/ignac/Escritorio/Delyrium/Lithos_Analytics_Challenge/images/rock_video/test', 
             use_segments=True)