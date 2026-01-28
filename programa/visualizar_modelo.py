"""
VISUALIZADOR DE ARQUITECTURA LSTM
Script auxiliar para mostrar detalles del modelo entrenado
"""

import pickle
from tensorflow.keras.models import load_model
import os

def mostrar_info_modelo():
    """
    Muestra información detallada del modelo entrenado
    """
    nombre_modelo = 'modelo_lstm_predictor'
    
    # Verificar que exista el modelo
    if not os.path.exists(f'{nombre_modelo}.keras'):
        print("No se encontró el modelo entrenado.")
        print("Por favor, ejecuta primero 'python predictor_palabras.py' para entrenar el modelo.")
        return
    
    print("="*70)
    print("ANÁLISIS DEL MODELO LSTM ENTRENADO")
    print("="*70)
    
    # Cargar modelo
    print("\n[1] Cargando modelo...")
    model = load_model(f'{nombre_modelo}.keras')
    
    # Cargar tokenizer
    with open(f'{nombre_modelo}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Cargar configuración
    with open(f'{nombre_modelo}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(" Modelo cargado correctamente\n")
    
    # Información del vocabulario
    print("="*70)
    print("[2] INFORMACIÓN DEL VOCABULARIO")
    print("="*70)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\n Tamaño del vocabulario: {vocab_size} palabras únicas")
    print(f" Longitud de secuencia: {config['longitud_secuencia']} palabras")
    
    print(f"\n Primeras 30 palabras del vocabulario:")
    print("-" * 70)
    palabras = list(tokenizer.word_index.items())[:30]
    for i in range(0, len(palabras), 3):
        fila = []
        for j in range(3):
            if i+j < len(palabras):
                palabra, idx = palabras[i+j]
                fila.append(f"{palabra:15} → {idx:3}")
        print("  ".join(fila))
    
    # Arquitectura del modelo
    print("\n" + "="*70)
    print("[3] ARQUITECTURA DE LA RED NEURONAL")
    print("="*70)
    print()
    model.summary()
    
    # Información de las capas
    print("\n" + "="*70)
    print("[4] DETALLES DE CADA CAPA")
    print("="*70)
    
    for i, layer in enumerate(model.layers):
        print(f"\n Capa {i+1}: {layer.name.upper()}")
        print("-" * 70)
        print(f"   Tipo: {layer.__class__.__name__}")
        
        if hasattr(layer, 'output_shape'):
            print(f"   Forma de salida: {layer.output_shape}")
        
        if hasattr(layer, 'count_params'):
            num_params = layer.count_params()
            print(f"   Parámetros entrenables: {num_params:,}")
        
        # Información específica por tipo de capa
        if layer.__class__.__name__ == 'Embedding':
            print(f"    Función: Convierte palabras (números) en vectores densos")
            print(f"    Vocabulario: {layer.input_dim} palabras")
            print(f"    Dimensión de embedding: {layer.output_dim}")
            print(f"    Ejemplo: palabra_5 → vector_50_dimensiones")
        
        elif layer.__class__.__name__ == 'LSTM':
            print(f"    Función: Procesa secuencias y aprende dependencias temporales")
            print(f"    Unidades LSTM: {layer.units}")
            print(f"    Puertas internas: Olvido, Entrada, Salida")
            print(f"    Soluciona: Vanishing gradient problem")
        
        elif layer.__class__.__name__ == 'Dense':
            print(f"    Función: Clasificación - predice la siguiente palabra")
            print(f"    Neuronas de salida: {layer.units}")
            print(f"    Activación: {layer.activation.__name__}")
            print(f"    Salida: Probabilidades para cada palabra del vocabulario")
    
    # Estadísticas del modelo
    print("\n" + "="*70)
    print("[5] ESTADÍSTICAS GENERALES")
    print("="*70)
    
    total_params = model.count_params()
    trainable_params = sum([layer.count_params() for layer in model.layers])
    
    print(f"\n Total de parámetros: {total_params:,}")
    print(f" Parámetros entrenables: {trainable_params:,}")
    
    # Calcular tamaño del modelo
    import sys
    tamanio_mb = os.path.getsize(f'{nombre_modelo}.keras') / (1024 * 1024)
    print(f" Tamaño del archivo del modelo: {tamanio_mb:.2f} MB")
    
    # Información de compilación
    print("\n" + "="*70)
    print("[6] CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*70)
    print(f"\n  Optimizador: {model.optimizer.__class__.__name__}")
    print(f"  Tasa de aprendizaje: {model.optimizer.learning_rate.numpy()}")
    print(f"  Función de pérdida: sparse_categorical_crossentropy")
    print(f"  Métricas: accuracy")
    
    # Ejemplos de uso
    print("\n" + "="*70)
    print("[7] FLUJO DE DATOS EN LA RED")
    print("="*70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│  EJEMPLO: Predecir la siguiente palabra para "el gato come"    │
└─────────────────────────────────────────────────────────────────┘

Paso 1: ENTRADA
   Texto: "el gato come"
   └─> Tokenización: [1, 2, 3]

Paso 2: CAPA EMBEDDING
   [1, 2, 3]
   └─> Vector 1: [0.23, -0.45, 0.12, ... ] (50 dims)
   └─> Vector 2: [0.67, 0.34, -0.23, ... ] (50 dims)
   └─> Vector 3: [-0.12, 0.89, 0.45, ... ] (50 dims)

Paso 3: CAPA LSTM
   Secuencia de 3 vectores de 50 dims
   └─> LSTM procesa secuencialmente
   └─> Mantiene estado interno (memoria)
   └─> Salida: Vector de estado [100 dims]

Paso 4: CAPA DENSE + SOFTMAX
   Vector [100 dims]
   └─> Transformación lineal
   └─> Softmax: convierte en probabilidades
   └─> Salida: [0.05, 0.03, 0.52, 0.15, ...] (vocab_size dims)
            └─> Palabra 3 tiene 52% probabilidad
                (la red predice "pescado")

Paso 5: PREDICCIÓN FINAL
   Se selecciona la palabra con mayor probabilidad
   └─> "pescado" (índice 4, prob: 0.52)
   └─> Resultado: "el gato come pescado"
""")
    
    print("\n" + "="*70)
    print(" ANÁLISIS COMPLETADO")
    print("="*70)
    print()

if __name__ == "__main__":
    mostrar_info_modelo()
