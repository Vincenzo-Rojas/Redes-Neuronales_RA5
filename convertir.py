import csv
import os

def csv_a_txt(ruta_csv, columna, ruta_txt=None, separador=',', codificacion='utf-8'):
    """
    Convierte una columna de un CSV en un archivo de texto plano.

    Args:
        ruta_csv (str): Ruta al archivo CSV de entrada.
        columna (str): Nombre de la columna que se desea extraer.
        ruta_txt (str, opcional): Ruta del archivo de texto de salida. 
                                  Si no se indica, se crea en el mismo directorio que el CSV.
        separador (str, opcional): Separador usado en el CSV (por defecto ',').
        codificacion (str, opcional): Codificación del archivo CSV (por defecto 'utf-8').

    Returns:
        str: Ruta del archivo de texto generado.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {ruta_csv}")

    if ruta_txt is None:
        ruta_txt = os.path.splitext(ruta_csv)[0] + ".txt"

    with open(ruta_csv, 'r', encoding=codificacion, newline='') as f_csv:
        lector = csv.DictReader(f_csv, delimiter=separador)
        
        if columna not in lector.fieldnames:
            raise ValueError(f"La columna '{columna}' no existe en el CSV. Columnas disponibles: {lector.fieldnames}")

        with open(ruta_txt, 'w', encoding=codificacion) as f_txt:
            for fila in lector:
                valor = fila[columna].strip()
                if valor:  # Evita líneas vacías
                    f_txt.write(valor + '\n')

    print(f"Archivo de texto generado en: {ruta_txt}")
    return ruta_txt

# Ejemplo de uso
if __name__ == "__main__":
    ruta_csv = "noticias_dataset.csv"       # Cambia esto por tu CSV
    columna = "cuerpo"            # Cambia esto por la columna que quieras extraer
    csv_a_txt(ruta_csv, columna, "noticias.txt")
