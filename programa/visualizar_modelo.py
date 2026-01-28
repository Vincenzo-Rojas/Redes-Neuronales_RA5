"""
VISUALIZADOR DE ARQUITECTURA LSTM
Script para analizar y mostrar detalles del modelo entrenado

Autor: Vincenzo Rojas Carrera
Asignatura: Ampliacion de Programacion - RA5
"""

import pickle
from tensorflow.keras.models import load_model
import os
import numpy as np

def mostrar_info_modelo():
    """
    Muestra informacion detallada del modelo LSTM entrenado
    """
    nombre_modelo = 'modelo_lstm_optimizado'
    
    # Verificar que exista el modelo
    if not os.path.exists(f'{nombre_modelo}.keras'):
        print("=" * 70)
        print("ERROR: Modelo no encontrado")
        print("=" * 70)
        print(f"\nNo se encontro el archivo: {nombre_modelo}.keras")
        print("\nPor favor, ejecuta primero el programa principal:")
        print("  python predictor_palabras_optimizado.py")
        print("\nEsto entrenara y guardara el modelo.")
        return
    
    print("=" * 70)
    print("ANALISIS DEL MODELO LSTM ENTRENADO")
    print("=" * 70)
    
    # Cargar modelo
    print("\n[1] Cargando modelo...")
    model = load_model(f'{nombre_modelo}.keras')
    
    # Cargar tokenizer
    with open(f'{nombre_modelo}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Cargar configuracion
    with open(f'{nombre_modelo}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print("    Modelo cargado correctamente\n")
    
    # =================================================================
    # INFORMACION DEL VOCABULARIO
    # =================================================================
    print("=" * 70)
    print("[2] INFORMACION DEL VOCABULARIO")
    print("=" * 70)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\nTamano del vocabulario: {vocab_size} palabras unicas")
    print(f"Longitud de secuencia: {config['longitud_secuencia']} palabras")
    print(f"Token OOV: <UNK>")
    
    print(f"\nPrimeras 30 palabras del vocabulario:")
    print("-" * 70)
    palabras = list(tokenizer.word_index.items())[:30]
    
    # Mostrar en columnas
    for i in range(0, len(palabras), 3):
        fila = []
        for j in range(3):
            if i+j < len(palabras):
                palabra, idx = palabras[i+j]
                fila.append(f"{palabra:15} -> {idx:3}")
        print("  ".join(fila))
    
    # Estadisticas del vocabulario
    print(f"\nEstadisticas del vocabulario:")
    print(f"  - Palabras totales indexadas: {len(tokenizer.word_index)}")
    print(f"  - Indice mas alto: {max(tokenizer.word_index.values())}")
    
    # Palabras mas frecuentes (top 10)
    print(f"\nTop 10 palabras mas frecuentes:")
    top_words = list(tokenizer.word_index.items())[:10]
    for palabra, idx in top_words:
        print(f"  {idx}. '{palabra}'")
    
    # =================================================================
    # ARQUITECTURA DEL MODELO
    # =================================================================
    print("\n" + "=" * 70)
    print("[3] ARQUITECTURA DE LA RED NEURONAL")
    print("=" * 70)
    print()
    
    # Mostrar resumen del modelo
    model.summary()
    
    # =================================================================
    # DETALLES DE CADA CAPA
    # =================================================================
    print("\n" + "=" * 70)
    print("[4] DETALLES DE CADA CAPA")
    print("=" * 70)
    
    for i, layer in enumerate(model.layers):
        print(f"\nCapa {i+1}: {layer.name.upper()}")
        print("-" * 70)
        print(f"   Tipo: {layer.__class__.__name__}")
        
        if hasattr(layer, 'output_shape'):
            print(f"   Forma de salida: {layer.output_shape}")
        
        if hasattr(layer, 'count_params'):
            num_params = layer.count_params()
            print(f"   Parametros entrenables: {num_params:,}")
        
        # Informacion especifica por tipo de capa
        if layer.__class__.__name__ == 'Embedding':
            print(f"   Funcion: Convierte palabras en vectores densos")
            print(f"   Vocabulario: {layer.input_dim} palabras")
            print(f"   Dimension de embedding: {layer.output_dim}")
            print(f"   Ejemplo: palabra_5 -> vector_de_{layer.output_dim}_dimensiones")
        
        elif layer.__class__.__name__ == 'LSTM':
            print(f"   Funcion: Procesa secuencias y aprende dependencias temporales")
            print(f"   Unidades LSTM: {layer.units}")
            print(f"   Return sequences: {layer.return_sequences}")
            if hasattr(layer, 'dropout'):
                print(f"   Dropout: {layer.dropout}")
            print(f"   Caracteristicas:")
            print(f"     - Puertas de olvido (forget gates)")
            print(f"     - Puertas de entrada (input gates)")
            print(f"     - Puertas de salida (output gates)")
            print(f"     - Estado de celda (cell state)")
        
        elif layer.__class__.__name__ == 'Dropout':
            print(f"   Funcion: Regularizacion - previene sobreajuste")
            print(f"   Tasa de dropout: {layer.rate}")
            print(f"   Durante entrenamiento: Desactiva {layer.rate*100}% de neuronas")
        
        elif layer.__class__.__name__ == 'Dense':
            print(f"   Funcion: Clasificacion - predice la siguiente palabra")
            print(f"   Neuronas de salida: {layer.units}")
            print(f"   Activacion: {layer.activation.__name__}")
            print(f"   Salida: Probabilidades para cada palabra del vocabulario")
        
        elif layer.__class__.__name__ == 'Bidirectional':
            print(f"   Funcion: Procesa secuencia en ambas direcciones")
            print(f"   Permite capturar contexto anterior y posterior")
    
    # =================================================================
    # ESTADISTICAS GENERALES
    # =================================================================
    print("\n" + "=" * 70)
    print("[5] ESTADISTICAS GENERALES DEL MODELO")
    print("=" * 70)
    
    total_params = model.count_params()
    trainable_params = sum([layer.count_params() for layer in model.layers])
    
    print(f"\nParametros del modelo:")
    print(f"  - Total de parametros: {total_params:,}")
    print(f"  - Parametros entrenables: {trainable_params:,}")
    print(f"  - Parametros no entrenables: {(total_params - trainable_params):,}")
    
    # Calcular tamano del modelo
    tamanio_mb = os.path.getsize(f'{nombre_modelo}.keras') / (1024 * 1024)
    tamanio_tokenizer_mb = os.path.getsize(f'{nombre_modelo}_tokenizer.pkl') / (1024 * 1024)
    
    print(f"\nTamanos de archivos:")
    print(f"  - Modelo (.keras): {tamanio_mb:.2f} MB")
    print(f"  - Tokenizer (.pkl): {tamanio_tokenizer_mb:.2f} MB")
    print(f"  - Total: {(tamanio_mb + tamanio_tokenizer_mb):.2f} MB")
    
    # =================================================================
    # CONFIGURACION DE ENTRENAMIENTO
    # =================================================================
    print("\n" + "=" * 70)
    print("[6] CONFIGURACION DE ENTRENAMIENTO")
    print("=" * 70)
    
    print(f"\nOptimizador: {model.optimizer.__class__.__name__}")
    if hasattr(model.optimizer, 'learning_rate'):
        lr = model.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            print(f"Learning rate: {lr.numpy()}")
        else:
            print(f"Learning rate: {lr}")
    
    print(f"Funcion de perdida: sparse_categorical_crossentropy")
    print(f"Metricas: accuracy")
    
    # Configuracion adicional guardada
    if 'embedding_dim' in config:
        print(f"\nConfiguracion de arquitectura:")
        print(f"  - Embedding dimension: {config['embedding_dim']}")
        print(f"  - LSTM units: {config['lstm_units']}")
        print(f"  - Numero de capas LSTM: {config['num_capas']}")
        print(f"  - Bidireccional: {config['bidireccional']}")
    
    # =================================================================
    # FLUJO DE DATOS
    # =================================================================
    print("\n" + "=" * 70)
    print("[7] FLUJO DE DATOS EN LA RED")
    print("=" * 70)
    
    print("""
EJEMPLO: Predecir siguiente palabra para "el modelo aprende"

Paso 1: ENTRADA
   Texto original: "el modelo aprende"
   Tokenizacion: [45, 127, 89] (indices del vocabulario)
   Secuencia de entrada: [45, 127, 89, 0, 0, ...] (padding hasta longitud_secuencia)

Paso 2: CAPA EMBEDDING
   Entrada: [45, 127, 89, ...]
   Salida: Matriz de vectores densos
     - Vector para palabra 45: [0.23, -0.45, 0.12, ..., 0.67] (embedding_dim valores)
     - Vector para palabra 127: [0.67, 0.34, -0.23, ..., -0.12]
     - Vector para palabra 89: [-0.12, 0.89, 0.45, ..., 0.34]

Paso 3: PRIMERA CAPA LSTM
   Entrada: Secuencia de vectores de embedding
   Procesamiento:
     - Puerta de olvido: Decide que informacion descartar
     - Puerta de entrada: Decide que nueva informacion almacenar
     - Actualizacion del estado de celda
     - Puerta de salida: Decide que informacion exponer
   Salida: Secuencia de estados ocultos [lstm_units dimensiones cada uno]

Paso 4: DROPOUT
   Entrada: Secuencia de estados LSTM
   Procesamiento: Desactiva aleatoriamente 20% de conexiones
   Salida: Secuencia regularizada

Paso 5: SEGUNDA CAPA LSTM
   Entrada: Secuencia de estados regularizados
   Procesamiento: Igual que primera capa LSTM
   Salida: Vector final de estado [lstm_units dimensiones]

Paso 6: CAPA DENSE + SOFTMAX
   Entrada: Vector de estado LSTM final [lstm_units dims]
   Procesamiento:
     - Transformacion lineal: W * x + b
     - Activacion softmax: convierte en probabilidades
   Salida: Vector de probabilidades [vocab_size dimensiones]
     [0.001, 0.003, 0.052, 0.015, ..., 0.234, ...]
            ^                              ^
            palabra_3: 5.2% prob        palabra_N: 23.4% prob (mas probable)

Paso 7: SELECCION DE PALABRA
   Se selecciona la palabra con mayor probabilidad (o usando sampling)
   Ejemplo: palabra con indice 1523 tiene prob 0.234
   Se convierte indice a palabra: "cuando"
   Resultado final: "el modelo aprende cuando"
""")
    
    # =================================================================
    # CAPACIDADES Y LIMITACIONES
    # =================================================================
    print("\n" + "=" * 70)
    print("[8] CAPACIDADES Y LIMITACIONES")
    print("=" * 70)
    
    print("\nCapacidades del modelo:")
    print("  - Procesa secuencias de hasta", config['longitud_secuencia'], "palabras")
    print("  - Vocabulario de", vocab_size, "palabras")
    print("  - Memoria a largo plazo mediante celdas LSTM")
    print("  - Regularizacion mediante dropout")
    print("  - Generacion de texto coherente")
    
    print("\nLimitaciones:")
    print("  - Solo conoce palabras del vocabulario de entrenamiento")
    print("  - Contexto limitado a", config['longitud_secuencia'], "palabras anteriores")
    print("  - No comprende significado real, solo patrones estadisticos")
    print("  - Requiere corpus grande para buenos resultados")
    print("  - Puede repetir patrones del texto de entrenamiento")
    
    # =================================================================
    # RECOMENDACIONES DE USO
    # =================================================================
    print("\n" + "=" * 70)
    print("[9] RECOMENDACIONES DE USO")
    print("=" * 70)
    
    print("\nPara obtener mejores resultados en la generacion:")
    print("  1. Usa semillas (frases iniciales) coherentes y en minusculas")
    print("  2. Ajusta la temperatura segun necesidad:")
    print("     - 0.5-0.7: Texto conservador y predecible")
    print("     - 0.8-1.0: Equilibrado (recomendado)")
    print("     - 1.2-1.5: Texto creativo con mas variedad")
    print("  3. Usa top-k entre 5-10 para balance entre coherencia y variedad")
    print("  4. Genera mas palabras (15-30) para ver patrones completos")
    print("  5. Si las predicciones son repetitivas, aumenta temperatura")
    print("  6. Si las predicciones son incoherentes, reduce temperatura")
    
    print("\nPara mejorar el modelo:")
    print("  1. Entrenar con mas datos (corpus mas grande)")
    print("  2. Aumentar el numero de epocas si no hay sobreajuste")
    print("  3. Ajustar hiperparametros (lstm_units, embedding_dim)")
    print("  4. Experimentar con arquitecturas bidireccionales")
    print("  5. Probar con mas capas LSTM si hay recursos suficientes")
    
    # =================================================================
    # FINALIZACION
    # =================================================================
    print("\n" + "=" * 70)
    print("ANALISIS COMPLETADO")
    print("=" * 70)
    print()
    print("El modelo esta listo para generar texto.")
    print("Ejecuta el programa principal para usar el modo interactivo:")
    print("  python predictor_palabras_optimizado.py")
    print()

if __name__ == "__main__":
    mostrar_info_modelo()
