"""
PREDICTOR DE PALABRAS CON LSTM - VERSIÓN OPTIMIZADA
Red Neuronal Recurrente para predicción de secuencias de texto
Optimizado para corpus gigantes con preprocesamiento avanzado

Autor: Vincenzo Rojas Carrera
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import os
import re
import unicodedata
from collections import Counter
import gc

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO AVANZADO
# ==========================================

def cargar_texto_desde_archivo(ruta_archivo=None):
    """
    Carga texto desde un archivo o genera uno grande de ejemplo.
    Para usar tu propio texto, pasa la ruta del archivo.
    
    Args:
        ruta_archivo: Path al archivo de texto (txt, opcional)
    
    Returns:
        str: Texto completo cargado
    """
    if ruta_archivo and os.path.exists(ruta_archivo):
        print(f" Cargando texto desde: {ruta_archivo}")
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
        print(f" Texto cargado: {len(texto)} caracteres")
        return texto
    
    # Si no hay archivo, generamos un corpus de ejemplo grande
    print(" Generando corpus de ejemplo...")
    
    # Corpus base con temas variados
    corpus_temas = [
        # Tecnología y IA
        """el modelo aprende cuando analiza secuencias de palabras que dependen del contexto previo.
        la red neuronal ajusta sus pesos mientras procesa información distribuida en el tiempo.
        el sistema mejora su predicción cuando la memoria interna conserva datos relevantes.
        la arquitectura lstm permite recordar información pasada durante el procesamiento secuencial.
        el entrenamiento continuo reduce errores acumulados en la generación de texto.
        la red procesa lenguaje natural sin reglas explícitas pero aprende patrones estadísticos.
        el modelo genera frases coherentes cuando el corpus mantiene una estructura estable.
        la memoria de la red influye directamente en la salida producida para cada palabra.
        el sistema aprende dependencias largas cuando las secuencias son consistentes.
        la predicción mejora si el entrenamiento incluye variaciones sintácticas controladas.""",
        
        # Ciencia y naturaleza
        """el universo contiene miles de millones de galaxias dispersas en el espacio infinito.
        la tierra gira alrededor del sol completando una órbita cada año aproximadamente.
        los océanos cubren más del setenta por ciento de la superficie del planeta.
        las plantas realizan fotosíntesis convirtiendo luz solar en energía química.
        los animales han evolucionado durante millones de años adaptándose al ambiente.
        la biodiversidad representa la variedad de formas de vida en nuestro mundo.
        el clima cambia constantemente afectando ecosistemas y especies vivientes.
        las montañas se forman por movimientos tectónicos de las placas terrestres.
        los ríos transportan agua desde las alturas hasta los mares y océanos.
        la atmósfera protege la vida filtrando radiación solar dañina.""",
        
        # Historia y cultura
        """las civilizaciones antiguas construyeron monumentos que perduran hasta hoy en día.
        el conocimiento se transmitía oralmente antes de la invención de la escritura.
        las revoluciones industriales transformaron completamente la sociedad moderna.
        el arte refleja las emociones y pensamientos de diferentes épocas históricas.
        la música ha evolucionado incorporando instrumentos y estilos muy diversos.
        los idiomas se desarrollan y cambian constantemente a través del tiempo.
        las tradiciones culturales definen la identidad de pueblos y naciones.
        la literatura preserva historias y conocimientos de generaciones pasadas.
        los descubrimientos científicos han revolucionado nuestra comprensión del mundo.
        la filosofía plantea preguntas fundamentales sobre la existencia humana.""",
        
        # Vida cotidiana
        """las personas se despiertan cada mañana para comenzar sus actividades diarias.
        el desayuno proporciona energía necesaria para afrontar el día completo.
        el trabajo ocupa una parte significativa del tiempo de la mayoría.
        las familias se reúnen para compartir momentos importantes y cotidianos.
        los amigos brindan apoyo emocional en momentos buenos y difíciles.
        el ejercicio físico mejora la salud y el bienestar general del cuerpo.
        la alimentación saludable contribuye a mantener un organismo fuerte.
        el descanso adecuado permite recuperar energías gastadas durante el día.
        los hobbies proporcionan entretenimiento y desarrollo de habilidades personales.
        la comunicación fortalece las relaciones entre las personas cercanas.""",
        
        # Educación y aprendizaje
        """la educación formal comienza desde edades tempranas en muchos países.
        los estudiantes adquieren conocimientos mediante la práctica y el estudio.
        los maestros guían el proceso de aprendizaje compartiendo su experiencia.
        las universidades ofrecen programas especializados en diversas disciplinas académicas.
        la lectura expande el vocabulario y mejora la comprensión del lenguaje.
        las matemáticas desarrollan el pensamiento lógico y la capacidad analítica.
        la ciencia explica fenómenos naturales mediante observación y experimentación.
        la tecnología educativa facilita el acceso al conocimiento en línea.
        el aprendizaje continuo es esencial en un mundo que cambia rápidamente.
        las habilidades prácticas complementan el conocimiento teórico adquirido."""
    ]
    
    # Generar un corpus grande repitiendo y variando el contenido
    texto_completo = []
    
    # Repetir cada tema múltiples veces con ligeras variaciones
    for tema in corpus_temas:
        # Agregar el tema original 5 veces
        for _ in range(5):
            texto_completo.append(tema)
    
    texto = " ".join(texto_completo)
    
    print(f" Corpus generado: {len(texto)} caracteres")
    print(f" Palabras aproximadas: {len(texto.split())}")
    
    return texto


def normalizar_unicode(texto):
    """
    Normaliza caracteres Unicode (importante para textos multilingües).
    Convierte caracteres acentuados a su forma NFD normalizada.
    """
    # Normalización NFD (Normalization Form Canonical Decomposition)
    texto = unicodedata.normalize('NFD', texto)
    # Mantener solo caracteres ASCII y espacios
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    return texto

def limpiar_texto_avanzado(texto):
    """Limpieza básica: minusculas, eliminar puntuacion y espacios extra."""
    texto = texto.lower()
    texto = re.sub(r'[\n\r\t]+', ' ', texto)
    texto = re.sub(r'[.,!?;:]', '', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def filtrar_palabras_raras(texto, umbral_frecuencia):
    """
    Opcional: Elimina palabras muy raras que aparecen pocas veces.
    Esto reduce el vocabulario y mejora la generalización.
    
    Args:
        texto: Texto a filtrar
        umbral_frecuencia: Mínimo de apariciones para mantener una palabra
    
    Returns:
        Texto filtrado
    """
    palabras = texto.split()
    
    # Contar frecuencias
    contador = Counter(palabras)
    
    # Filtrar palabras raras
    palabras_filtradas = [
        palabra for palabra in palabras 
        if contador[palabra] >= umbral_frecuencia
    ]
    
    palabras_eliminadas = len(palabras) - len(palabras_filtradas)
    if palabras_eliminadas > 0:
        print(f"  → Palabras raras eliminadas: {palabras_eliminadas}")
    
    return ' '.join(palabras_filtradas)

def filtrar_letras_sueltas(texto, permitir={'y','o','a','e','u','i'}):
    """
    Elimina letras sueltas que no tengan relevancia,
    salvo las incluidas en 'permitir'.
    """
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if len(p) > 1 or p in permitir]
    return ' '.join(palabras_filtradas)


def preprocesar_texto_completo(texto, filtrar_raras=False, umbral_freq=2):
    """
    Pipeline completo de preprocesamiento.
    
    Args:
        texto: Texto crudo
        filtrar_raras: Si eliminar palabras poco frecuentes
        umbral_freq: Umbral de frecuencia para filtrado
    
    Returns:
        Texto preprocesado y listo para tokenización
    """
    print("\n Iniciando preprocesamiento avanzado...")
    
    # 1. Normalizar Unicode
    print("  [1/4] Normalizando caracteres Unicode...")
    texto = normalizar_unicode(texto)
    
    # 2. Limpieza avanzada
    print("  [2/4] Limpiando texto...")
    texto = limpiar_texto_avanzado(texto)
    
    # 3. Filtrar palabras raras (opcional)
    if filtrar_raras:
        print(f"  [3/4] Filtrando palabras con frecuencia < {umbral_freq}...")
        texto = filtrar_palabras_raras(texto, umbral_freq)
    else:
        print("  [3/4] Omitiendo filtrado de palabras raras...")
    
    # 4. Estadísticas finales
    print("  [4/4] Calculando estadísticas...")
    num_caracteres = len(texto)
    num_palabras = len(texto.split())
    palabras_unicas = len(set(texto.split()))
    
    print(f"\n Estadísticas del texto preprocesado:")
    print(f"  • Caracteres: {num_caracteres:,}")
    print(f"  • Palabras totales: {num_palabras:,}")
    print(f"  • Palabras únicas: {palabras_unicas:,}")
    print(f"  • Ratio único/total: {palabras_unicas/num_palabras:.2%}")
    
    return texto


# ==========================================
# 2. TOKENIZACIÓN OPTIMIZADA
# ==========================================

def crear_vocabulario_optimizado(texto, vocab_max=10000, oov_token='<UNK>'):
    """
    Crea vocabulario limitando el tamaño máximo.
    Esto es crucial para corpus gigantes.
    
    Args:
        texto: Texto preprocesado
        vocab_max: Tamaño máximo del vocabulario
        oov_token: Token para palabras fuera del vocabulario
    
    Returns:
        tokenizer, vocab_size
    """
    print(f"\n Creando vocabulario (máx: {vocab_max} palabras)...")
    
    # Tokenizer con límite de vocabulario
    tokenizer = Tokenizer(
        num_words=vocab_max,
        oov_token=oov_token,
        filters='',  # Ya limpiamos antes
        lower=False  # Ya está en minúsculas
    )
    
    tokenizer.fit_on_texts([texto])
    
    # Tamaño real del vocabulario
    vocab_size = min(len(tokenizer.word_index) + 1, vocab_max)
    
    print(f" Vocabulario creado:")
    print(f"  • Palabras únicas encontradas: {len(tokenizer.word_index)}")
    print(f"  • Tamaño del vocabulario usado: {vocab_size}")
    print(f"  • Token OOV: '{oov_token}' (índice: {tokenizer.word_index.get(oov_token, 'N/A')})")
    
    # Mostrar palabras más frecuentes
    print(f"\n  20 palabras de ejemplo:")
    for palabra, indice in list(tokenizer.word_index.items())[:20]:
        print(f"     '{palabra}' → {indice}")
    
    return tokenizer, vocab_size


# ==========================================
# 3. GENERACIÓN EFICIENTE DE SECUENCIAS
# ==========================================

def crear_secuencias_eficiente(texto, tokenizer, longitud_secuencia=10, 
                               stride=1, max_secuencias=None):
    """
    Crea secuencias de entrenamiento de forma eficiente.
    Optimizado para textos grandes.
    
    Args:
        texto: Texto preprocesado
        tokenizer: Tokenizer entrenado
        longitud_secuencia: Longitud de cada secuencia
        stride: Paso entre secuencias (1=todas, 2=cada 2, etc.)
        max_secuencias: Máximo de secuencias a crear (None=todas)
    
    Returns:
        X, y, longitud_secuencia
    """
    print(f"\n  Generando secuencias de entrenamiento...")
    print(f"  • Longitud de secuencia: {longitud_secuencia}")
    print(f"  • Stride (paso): {stride}")
    
    # Convertir texto a secuencia de números
    secuencia_completa = tokenizer.texts_to_sequences([texto])[0]
    total_palabras = len(secuencia_completa)
    
    print(f"  • Total de palabras tokenizadas: {total_palabras:,}")
    
    # Calcular número de secuencias posibles
    num_secuencias_posibles = (total_palabras - longitud_secuencia) // stride
    
    if max_secuencias and num_secuencias_posibles > max_secuencias:
        print(f"  • Limitando a {max_secuencias:,} secuencias (de {num_secuencias_posibles:,} posibles)")
        num_secuencias = max_secuencias
    else:
        num_secuencias = num_secuencias_posibles
    
    # Pre-asignar arrays (más eficiente que append)
    X = np.zeros((num_secuencias, longitud_secuencia), dtype=np.int32)
    y = np.zeros(num_secuencias, dtype=np.int32)
    
    # Generar secuencias con stride
    idx = 0
    for i in range(0, total_palabras - longitud_secuencia, stride):
        if idx >= num_secuencias:
            break
        
        X[idx] = secuencia_completa[i:i + longitud_secuencia]
        y[idx] = secuencia_completa[i + longitud_secuencia]
        idx += 1
        
        # Mostrar progreso cada 10000 secuencias
        if (idx % 10000 == 0):
            print(f"    Procesadas: {idx:,} secuencias...", end='\r')
    
    print(f"\n✓ Secuencias creadas:")
    print(f"  • Total de secuencias: {len(X):,}")
    print(f"  • Forma de X: {X.shape}")
    print(f"  • Forma de y: {y.shape}")
    print(f"  • Memoria usada: ~{(X.nbytes + y.nbytes) / (1024**2):.2f} MB")
    
    # Mostrar ejemplos
    print(f"\n   Ejemplos de secuencias:")
    for i in range(min(3, len(X))):
        entrada_texto = []
        for idx in X[i]:
            for palabra, word_idx in tokenizer.word_index.items():
                if word_idx == idx:
                    entrada_texto.append(palabra)
                    break
        
        salida_texto = None
        for palabra, word_idx in tokenizer.word_index.items():
            if word_idx == y[i]:
                salida_texto = palabra
                break
        
        print(f"     {entrada_texto} → '{salida_texto}'")
    
    return X, y, longitud_secuencia


# ==========================================
# 4. ARQUITECTURA LSTM MEJORADA
# ==========================================

def crear_modelo_optimizado(vocab_size, longitud_secuencia, 
                           embedding_dim=128, lstm_units=256,
                           usar_bidireccional=False, num_capas_lstm=1):
    """
    Arquitectura LSTM optimizada para mejores resultados.
    
    Args:
        vocab_size: Tamaño del vocabulario
        longitud_secuencia: Longitud de secuencias de entrada
        embedding_dim: Dimensión de embeddings (64-256 típico)
        lstm_units: Unidades LSTM (128-512 típico)
        usar_bidireccional: Si usar Bidirectional LSTM
        num_capas_lstm: Número de capas LSTM apiladas
    
    Returns:
        Modelo compilado
    """
    print(f"\n🏗️  Construyendo arquitectura LSTM...")
    print(f"  • Embedding dim: {embedding_dim}")
    print(f"  • LSTM units: {lstm_units}")
    print(f"  • Capas LSTM: {num_capas_lstm}")
    print(f"  • Bidireccional: {usar_bidireccional}")
    
    model = Sequential()
    
    # Capa Embedding
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embedding'
    ))
    
    # Capas LSTM apiladas
    for i in range(num_capas_lstm):
        return_sequences = (i < num_capas_lstm - 1)  # Todas menos la última
        
        if usar_bidireccional:
            model.add(Bidirectional(
                LSTM(lstm_units, 
                     return_sequences=return_sequences,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     name=f'lstm_{i+1}')
            ))
        else:
            model.add(LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=0.2,
                recurrent_dropout=0.2,
                name=f'lstm_{i+1}'
            ))
        
        # Dropout adicional entre capas
        if i < num_capas_lstm - 1:
            model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    # Capa de salida
    model.add(Dense(vocab_size, activation='softmax', name='output'))
    
    print("\n" + "="*70)
    print("ARQUITECTURA DEL MODELO")
    print("="*70)
    model.build(input_shape=(None, longitud_secuencia))
    model.summary()
    
    return model


# ==========================================
# 5. ENTRENAMIENTO OPTIMIZADO
# ==========================================

def entrenar_modelo_optimizado(model, X, y, epocas=50, batch_size=128,
                               validation_split=0.1, nombre_modelo='modelo_lstm'):
    """
    Entrenamiento con callbacks avanzados para mejor rendimiento.
    
    Args:
        model: Modelo a entrenar
        X, y: Datos de entrenamiento
        epocas: Número máximo de épocas
        batch_size: Tamaño del batch (más grande=más rápido en GPU)
        validation_split: Fracción para validación
        nombre_modelo: Nombre base para guardar checkpoints
    
    Returns:
        model, history
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*70)
    
    # Compilar modelo
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print(f"\n Configuración de entrenamiento:")
    print(f"  • Épocas máximas: {epocas}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Validación: {validation_split*100:.0f}%")
    print(f"  • Optimizador: Adam (lr=0.001)")
    print(f"  • Loss: Sparse Categorical Crossentropy")
    
    # Callbacks avanzados
    callbacks = [
        # Guardar mejor modelo
        ModelCheckpoint(
            filepath=f'{nombre_modelo}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=2
        ),
        
        # Early stopping para evitar sobreentrenamiento
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=2
        ),
        
        # Reducir learning rate cuando se estanque
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=2
        )
    ]
    
    print(f"\n Callbacks activados:")
    print(f"  • ModelCheckpoint: Guarda mejor modelo")
    print(f"  • EarlyStopping: Para si no mejora en 10 épocas")
    print(f"  • ReduceLROnPlateau: Reduce LR si se estanca")
    
    print(f"\n Iniciando entrenamiento...\n")
    
    # Entrenar
    history = model.fit(
        X, y,
        epochs=epocas,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2
    )
    
    print("\n" + "="*70)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"  • Épocas ejecutadas: {len(history.history['loss'])}")
    print(f"  • Pérdida final (train): {history.history['loss'][-1]:.4f}")
    print(f"  • Pérdida final (val): {history.history['val_loss'][-1]:.4f}")
    print(f"  • Precisión final (train): {history.history['accuracy'][-1]:.4f}")
    print(f"  • Precisión final (val): {history.history['val_accuracy'][-1]:.4f}")
    
    # Liberar memoria
    gc.collect()
    
    return model, history


# ==========================================
# 6. GUARDAR Y CARGAR
# ==========================================

def guardar_modelo_completo(model, tokenizer, longitud_secuencia, 
                           config_adicional=None, nombre='modelo_lstm'):
    """
    Guarda modelo y metadatos.
    """
    print(f"\n Guardando modelo y configuración...")
    
    # Guardar modelo
    model.save(f'{nombre}.keras')
    
    # Guardar tokenizer
    with open(f'{nombre}_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Configuración completa
    config = {
        'longitud_secuencia': longitud_secuencia,
        'vocab_size': len(tokenizer.word_index) + 1,
    }
    
    if config_adicional:
        config.update(config_adicional)
    
    with open(f'{nombre}_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    # Calcular tamaños
    tamanio_modelo = os.path.getsize(f'{nombre}.keras') / (1024**2)
    tamanio_tokenizer = os.path.getsize(f'{nombre}_tokenizer.pkl') / (1024**2)
    
    print(f"✓ Modelo guardado: {nombre}.keras ({tamanio_modelo:.2f} MB)")
    print(f"✓ Tokenizer guardado: {nombre}_tokenizer.pkl ({tamanio_tokenizer:.2f} MB)")
    print(f"✓ Config guardada: {nombre}_config.pkl")


def cargar_modelo_completo(nombre='modelo_lstm'):
    """
    Carga modelo y metadatos.
    """
    print(f"\n Cargando modelo desde disco...")
    
    model = load_model(f'{nombre}.keras')
    
    with open(f'{nombre}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(f'{nombre}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✓ Modelo cargado: {nombre}.keras")
    print(f"✓ Vocabulario: {config.get('vocab_size', 'N/A')} palabras")
    print(f"✓ Longitud de secuencia: {config['longitud_secuencia']}")
    
    return model, tokenizer, config


# ==========================================
# 7. GENERACIÓN DE TEXTO MEJORADA
# ==========================================
PALABRAS_PENALIZADAS = ["que","de","y","la","a","el","en","no","se","los","con","por","las","lo","le","su","don","del","como"]

def aplicar_penalizacion_dinamica(probs, tokenizer, texto_generado, factor_stopwords=0.3, factor_repeticion=0.5, factor_ultima=0.1):
    """Aplica penalizaciones dinámicas para evitar repeticiones y stopwords."""
    probs = np.array(probs, dtype=np.float64)
    palabras_previas = texto_generado.split()
    indices_previos = [tokenizer.word_index[p] for p in palabras_previas if p in tokenizer.word_index]
    for palabra in PALABRAS_PENALIZADAS:
        idx = tokenizer.word_index.get(palabra)
        if idx is not None:
            probs[idx] *= factor_stopwords
    for idx in indices_previos:
        probs[idx] *= factor_repeticion
    if palabras_previas:
        idx_ultima = tokenizer.word_index.get(palabras_previas[-1])
        if idx_ultima is not None:
            probs[idx_ultima] *= factor_ultima
    probs += 1e-12
    probs /= probs.sum()
    return probs

def generar_texto_mejorado(model, tokenizer, texto_inicial, longitud_secuencia,
                          num_palabras=10, temperatura=1.0, top_k=5):
    """
    Generación de texto con sampling mejorado.
    
    Args:
        model: Modelo entrenado
        tokenizer: Tokenizer usado
        texto_inicial: Frase semilla
        longitud_secuencia: Longitud de contexto
        num_palabras: Palabras a generar
        temperatura: Control de aleatoriedad (0.5=conservador, 1.5=creativo)
        top_k: Considerar solo las top-k palabras más probables
    
    Returns:
        Texto generado
    """
    texto_actual = texto_inicial.lower().strip()
    
    print(f"\n  Generando texto...")
    print(f"  • Semilla: '{texto_inicial}'")
    print(f"  • Palabras a generar: {num_palabras}")
    print(f"  • Temperatura: {temperatura}")
    print(f"  • Top-k: {top_k}")
    print()
    
    for i in range(num_palabras):
        # Tokenizar
        secuencia = tokenizer.texts_to_sequences([texto_actual])[0]
        
        # Tomar últimas N palabras
        if len(secuencia) >= longitud_secuencia:
            secuencia = secuencia[-longitud_secuencia:]
        else:
            secuencia = [0] * (longitud_secuencia - len(secuencia)) + secuencia
        
        secuencia = np.array([secuencia])
        
        # Predecir
        prediccion = model.predict(secuencia, verbose=0)[0]

        # Aplicar penalización dinámica
        prediccion = aplicar_penalizacion_dinamica(prediccion, tokenizer, texto_actual)
        
        # Aplicar temperatura
        prediccion = np.log(prediccion + 1e-10) / temperatura
        prediccion = np.exp(prediccion) / np.sum(np.exp(prediccion))
        
        # Top-k sampling
        top_indices = np.argsort(prediccion)[-top_k:]
        top_probs = prediccion[top_indices]
        top_probs = top_probs / np.sum(top_probs)
        
        # Muestrear
        palabra_idx = np.random.choice(top_indices, p=top_probs)
        
        # Convertir a palabra
        palabra = None
        for w, idx in tokenizer.word_index.items():
            if idx == palabra_idx:
                palabra = w
                break
        
        if palabra:
            texto_actual += " " + palabra
    
    return texto_actual


# ==========================================
# 8. PROGRAMA PRINCIPAL OPTIMIZADO
# ==========================================

def main():
    """
    Pipeline completo optimizado para textos gigantes.
    """
    print("="*70)
    print("PREDICTOR DE PALABRAS CON LSTM - VERSIÓN OPTIMIZADA")
    print("Para corpus gigantes con preprocesamiento avanzado")
    print("="*70)
    
    # ============ CONFIGURACIÓN ============
    ARCHIVO_TEXTO = 'corpus.txt'  # Cambia a 'tu_archivo.txt' para usar tu texto
    NOMBRE_MODELO = 'modelo_lstm_optimizado'
    
    # Parámetros de preprocesamiento
    FILTRAR_RARAS = True  # True para eliminar palabras poco frecuentes
    UMBRAL_FRECUENCIA = 20
    
    # Parámetros de tokenización
    VOCAB_MAX = 30000  # Vocabulario máximo
    
    # Parámetros de secuencias
    LONGITUD_SECUENCIA = 30  # Contexto (más largo = más contexto, más memoria)
    STRIDE = 1  # 1=todas las secuencias, 2=cada 2, etc.
    MAX_SECUENCIAS = 300000  # None=todas, o número para limitar
    
    # Parámetros de arquitectura
    EMBEDDING_DIM = 64         # Dimensión de los vectores de embedding; cada palabra se representa con un vector de 64 valores
    LSTM_UNITS = 256           # Número de unidades en cada capa LSTM; define la "capacidad de memoria" de la red
    USAR_BIDIRECCIONAL = False # Indica si la LSTM procesa la secuencia en ambas direcciones (True) o solo hacia adelante (False)
    NUM_CAPAS_LSTM = 2         # Número de capas LSTM apiladas; más capas permiten aprender patrones más complejos

    # Parámetros de entrenamiento
    EPOCAS = 30                # Número máximo de pasadas sobre todo el conjunto de entrenamiento
    BATCH_SIZE = 64            # Número de secuencias procesadas antes de actualizar los pesos del modelo
    VALIDATION_SPLIT = 0.2     # Porcentaje de los datos reservados para validación durante el entrenamiento

    
    # ============ VERIFICAR MODELO EXISTENTE ============
    modelo_existe = os.path.exists(f'{NOMBRE_MODELO}.keras')
    
    if modelo_existe:
        print(f"\n✓ Modelo existente encontrado: {NOMBRE_MODELO}.keras")
        respuesta = input("¿Deseas (1) Reentrenar o (2) Solo usar el existente? [1/2]: ")
        entrenar = (respuesta == '1')
    else:
        print(f"\n  No se encontró modelo existente.")
        entrenar = True
    
    # ============ ENTRENAMIENTO ============
    if entrenar:
        print("\n" + "="*70)
        print("FASE 1: CARGA Y PREPROCESAMIENTO")
        print("="*70)
        
        # Cargar texto
        texto_crudo = cargar_texto_desde_archivo(ARCHIVO_TEXTO)
        
        # Preprocesar
        texto = preprocesar_texto_completo(
            texto_crudo,
            filtrar_raras=FILTRAR_RARAS,
            umbral_freq=UMBRAL_FRECUENCIA
        )
        
        # Tokenizar
        tokenizer, vocab_size = crear_vocabulario_optimizado(
            texto,
            vocab_max=VOCAB_MAX
        )
        
        # Crear secuencias
        X, y, longitud_secuencia = crear_secuencias_eficiente(
            texto,
            tokenizer,
            longitud_secuencia=LONGITUD_SECUENCIA,
            stride=STRIDE,
            max_secuencias=MAX_SECUENCIAS
        )
        
        print("\n" + "="*70)
        print("FASE 2: CONSTRUCCIÓN Y ENTRENAMIENTO")
        print("="*70)
        
        # Crear modelo
        model = crear_modelo_optimizado(
            vocab_size,
            longitud_secuencia,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            usar_bidireccional=USAR_BIDIRECCIONAL,
            num_capas_lstm=NUM_CAPAS_LSTM
        )
        
        # Entrenar
        model, history = entrenar_modelo_optimizado(
            model, X, y,
            epocas=EPOCAS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            nombre_modelo=NOMBRE_MODELO
        )
        
        # Guardar
        config_adicional = {
            'embedding_dim': EMBEDDING_DIM,
            'lstm_units': LSTM_UNITS,
            'bidireccional': USAR_BIDIRECCIONAL,
            'num_capas': NUM_CAPAS_LSTM
        }
        
        guardar_modelo_completo(
            model, tokenizer, longitud_secuencia,
            config_adicional=config_adicional,
            nombre=NOMBRE_MODELO
        )
        
        # Limpiar memoria
        del X, y, texto, texto_crudo
        gc.collect()
    
    else:
        # Cargar modelo existente
        model, tokenizer, config = cargar_modelo_completo(NOMBRE_MODELO)
        longitud_secuencia = config['longitud_secuencia']
    
    # ============ GENERACIÓN Y PRUEBAS ============
    print("\n" + "="*70)
    print("FASE 3: GENERACIÓN DE TEXTO")
    print("="*70)
    
    # Frases de prueba
    frases_prueba = [
        "el modelo aprende",
        "la red neuronal",
        "el universo contiene",
        "las personas se",
        "la educación formal"
    ]
    
    print("\n📝 Generación con diferentes semillas:\n")
    for frase in frases_prueba:
        resultado = generar_texto_mejorado(
            model, tokenizer, frase, longitud_secuencia,
            num_palabras=8, temperatura=0.8, top_k=5
        )
        print(f"→ {resultado}\n")
    
    # ============ MODO INTERACTIVO ============
    print("\n" + "="*70)
    print("MODO INTERACTIVO")
    print("="*70)
    print("Comandos:")
    print("  • Escribe una frase para generar texto")
    print("  • 'config' para cambiar parámetros de generación")
    print("  • 'salir' para terminar\n")
    
    temperatura = 0.8
    top_k = 5
    num_palabras = 10
    
    while True:
        entrada = input("Texto inicial: ").strip()
        
        if entrada.lower() == 'salir':
            break
        
        elif entrada.lower() == 'config':
            try:
                temperatura = float(input(f"  Temperatura (actual: {temperatura}): ") or temperatura)
                top_k = int(input(f"  Top-k (actual: {top_k}): ") or top_k)
                num_palabras = int(input(f"  Palabras (actual: {num_palabras}): ") or num_palabras)
                print("✓ Configuración actualizada\n")
            except:
                print("❌ Valores inválidos, manteniendo configuración anterior\n")
            continue
        
        if entrada:
            resultado = generar_texto_mejorado(
                model, tokenizer, entrada, longitud_secuencia,
                num_palabras=num_palabras,
                temperatura=temperatura,
                top_k=top_k
            )
            print(f"Resultado: {resultado}\n")
    
    print("\n Programa finalizado!")


if __name__ == "__main__":
    main()
