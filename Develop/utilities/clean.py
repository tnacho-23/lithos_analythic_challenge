import os

def eliminar_duplicados(carpeta_origen, carpeta_destino):
    # Verificamos que ambas rutas existan para evitar errores
    if not os.path.exists(carpeta_origen) or not os.path.exists(carpeta_destino):
        print("Error: Una o ambas carpetas no existen.")
        return

    # Obtenemos la lista de nombres de archivos en la carpeta de origen
    archivos_origen = set(os.listdir(carpeta_origen))
    
    contador = 0

    # Recorremos la carpeta de destino
    for nombre_archivo in os.listdir(carpeta_destino):
        # Si el archivo existe en la carpeta de origen...
        if nombre_archivo in archivos_origen:
            ruta_a_eliminar = os.path.join(carpeta_destino, nombre_archivo)
            
            # Verificamos que sea un archivo (y no una subcarpeta) antes de borrar
            if os.path.isfile(ruta_a_eliminar):
                try:
                    os.remove(ruta_a_eliminar)
                    print(f"Eliminado: {nombre_archivo}")
                    contador += 1
                except Exception as e:
                    print(f"No se pudo eliminar {nombre_archivo}: {e}")

    print(f"\nProceso terminado. Se eliminaron {contador} archivos de la carpeta destino.")

# --- Configuración ---
# Cambia estas rutas por las tuyas
ruta_referencia = "C:/Users/ignac/Escritorio/Delyrium/lithos_analytics_challenge/images/given_dataset/train/labels" 
ruta_a_limpiar = "C:/Users/ignac/Escritorio/Delyrium/lithos_analytics_challenge/images/full_dataset/train/labels" 

eliminar_duplicados(ruta_referencia, ruta_a_limpiar)