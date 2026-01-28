"""
PREDICTOR DE PALABRAS CON LSTM - VERSIÓN OPTIMIZADA
Red Neuronal Recurrente para predicción de secuencias de texto
Optimizado para corpus gigantes con preprocesamiento avanzado

Autor: Vincenzo Rojas
Asignatura: Ampliacion de programacion - RA5
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
# 0. Corpus de ejemplo
# ==========================================

corpus_ejemplo = [
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

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO AVANZADO
# ==========================================
def cargar_texto_desde_archivo(ruta_archivo=None):
    """
    Carga texto desde un archivo .txt o genera un corpus de ejemplo.
    Retorna texto concatenado.
    """
    global corpus_ejemplo

    if ruta_archivo is not None and os.path.exists(ruta_archivo):
        if not ruta_archivo.endswith('.txt'):
            raise ValueError("Solo se permiten archivos .txt")
        
        print(f"Cargando texto desde: {ruta_archivo}")
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
        print(f"Texto cargado: {len(texto)} caracteres")
        return texto

    # Generar corpus de ejemplo si no hay archivo
    print("Generando corpus de ejemplo...") 
    texto_completo = []
    for tema in corpus_ejemplo:
        for _ in range(5):
            texto_completo.append(tema)
    texto = " ".join(texto_completo)
    print(f"Corpus generado: {len(texto)} caracteres")
    print(f"Palabras aproximadas: {len(texto.split())}")
    return texto


def normalizar_unicode(texto):
    """Elimina acentos y caracteres especiales de unicode."""
    texto = unicodedata.normalize('NFD', texto)
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
    """Elimina palabras que aparecen menos que 'umbral_frecuencia'."""
    palabras = texto.split()
    contador = Counter(palabras)
    palabras_filtradas = [p for p in palabras if contador[p] >= umbral_frecuencia]
    palabras_eliminadas = len(palabras) - len(palabras_filtradas)
    if palabras_eliminadas > 0:
        print(f"Palabras raras eliminadas: {palabras_eliminadas}")
    return ' '.join(palabras_filtradas)

def filtrar_letras_sueltas(texto, permitir={'y','o','a','e','u','i'}):
    """
    Elimina letras sueltas que no tengan relevancia,
    salvo las incluidas en 'permitir'.
    """
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if len(p) > 1 or p in permitir]
    return ' '.join(palabras_filtradas)

def preprocesar_texto_completo(texto, filtrar_raras=False, umbral_freq=10):
    """Pipeline de preprocesamiento avanzado con eliminación de letras irrelevantes."""
    print("Iniciando preprocesamiento avanzado...")
    texto = normalizar_unicode(texto)
    texto = limpiar_texto_avanzado(texto)
    if filtrar_raras:
        texto = filtrar_palabras_raras(texto, umbral_freq)
    texto = filtrar_letras_sueltas(texto)
    num_caracteres = len(texto)
    num_palabras = len(texto.split())
    palabras_unicas = len(set(texto.split()))
    print(f"Caracteres: {num_caracteres}, Palabras: {num_palabras}, Palabras unicas: {palabras_unicas}")
    return texto

# ==========================================
# 2. TOKENIZACIÓN OPTIMIZADA
# ==========================================
def crear_vocabulario_optimizado(texto, vocab_max=10000, oov_token='<UNK>'):
    """Tokeniza texto y limita el vocabulario."""
    print(f"Creando vocabulario (max: {vocab_max} palabras)...")
    tokenizer = Tokenizer(num_words=vocab_max, oov_token=oov_token, filters='', lower=False)
    tokenizer.fit_on_texts([texto])
    vocab_size = min(len(tokenizer.word_index) + 1, vocab_max)
    print(f"Vocabulario creado: {vocab_size} palabras")
    idx2word = {idx: word for word, idx in tokenizer.word_index.items()}
    return tokenizer, vocab_size, idx2word

# ==========================================
# 3. GENERACIÓN EFICIENTE DE SECUENCIAS
# ==========================================
def crear_secuencias_eficiente(texto, tokenizer, longitud_secuencia=10, stride=1, max_secuencias=None):
    """Genera pares X,y para entrenamiento de LSTM."""
    print("Generando secuencias de entrenamiento...")
    secuencia_completa = tokenizer.texts_to_sequences([texto])[0]
    total_palabras = len(secuencia_completa)
    num_secuencias_posibles = (total_palabras - longitud_secuencia) // stride
    num_secuencias = min(num_secuencias_posibles, max_secuencias) if max_secuencias else num_secuencias_posibles
    X = np.zeros((num_secuencias, longitud_secuencia), dtype=np.int32)
    y = np.zeros(num_secuencias, dtype=np.int32)
    idx = 0
    for i in range(0, total_palabras - longitud_secuencia, stride):
        if idx >= num_secuencias:
            break
        X[idx] = secuencia_completa[i:i+longitud_secuencia]
        y[idx] = secuencia_completa[i+longitud_secuencia]
        idx += 1

    print(f"Secuencias generadas: {len(X)} (Ejemplo de 3 primeras)")
    for i in range(min(3, len(X))):
        entrada_texto = [word for idx in X[i] for word, w_idx in tokenizer.word_index.items() if w_idx == idx]
        salida_texto = next((word for word, w_idx in tokenizer.word_index.items() if w_idx == y[i]), '')
        print(f"   {entrada_texto} -> '{salida_texto}'")
    
    return X, y, longitud_secuencia

# ==========================================
# 4. ARQUITECTURA LSTM MEJORADA
# ==========================================
def crear_modelo_optimizado(vocab_size, longitud_secuencia, embedding_dim=128, lstm_units=256, usar_bidireccional=False, num_capas_lstm=1):
    """Crea un modelo LSTM secuencial optimizado."""
    print("Construyendo arquitectura LSTM...")
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding'))
    for i in range(num_capas_lstm):
        return_sequences = (i < num_capas_lstm - 1)
        if usar_bidireccional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=return_sequences, dropout=0.2, name=f'lstm_{i+1}')))
        else:
            model.add(LSTM(lstm_units, return_sequences=return_sequences, dropout=0.2, name=f'lstm_{i+1}'))
        if i < num_capas_lstm - 1:
            model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    model.add(Dense(vocab_size, activation='softmax', name='output'))

    model.build(input_shape=(None, longitud_secuencia))
    model.summary()
    return model

# ==========================================
# 5. ENTRENAMIENTO OPTIMIZADO
# ==========================================
def entrenar_modelo_optimizado(model, X, y, epocas=50, batch_size=128, validation_split=0.1, nombre_modelo='modelo_lstm'):
    """Entrenamiento con callbacks avanzados y prints de progreso."""
    print("Compilando modelo...")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(filepath=f'{nombre_modelo}_best.keras', monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    print("Iniciando entrenamiento...")
    history = model.fit(X, y, epochs=epocas, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=2)
    print("Entrenamiento completado!")
    return model, history

# ==========================================
# 6. GUARDAR Y CARGAR
# ==========================================
def guardar_modelo_completo(model, tokenizer, longitud_secuencia, config_adicional=None, nombre='modelo_lstm'):
    """Guarda modelo, tokenizer y configuración."""
    print(f"Guardando modelo {nombre}...")
    model.save(f'{nombre}.keras')
    with open(f'{nombre}_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    config = {'longitud_secuencia': longitud_secuencia, 'vocab_size': len(tokenizer.word_index) + 1}
    if config_adicional:
        config.update(config_adicional)
    with open(f'{nombre}_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("Guardado completado!")

def cargar_modelo_completo(nombre='modelo_lstm'):
    """Carga modelo y metadatos desde disco."""
    print(f"Cargando modelo {nombre} desde disco...")
    model = load_model(f'{nombre}.keras')
    with open(f'{nombre}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open(f'{nombre}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    print("Carga completada!")
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

def generar_texto_mejorado(model, tokenizer, texto_inicial, longitud_secuencia, num_palabras=10, temperatura=1.0, top_k=5):
    """Genera texto a partir de una semilla."""
    texto_actual = texto_inicial.lower().strip()
    print(f"Generando texto con semilla: '{texto_inicial}'")
    for i in range(num_palabras):
        secuencia = tokenizer.texts_to_sequences([texto_actual])[0]
        secuencia = ([0]*(longitud_secuencia-len(secuencia)) + secuencia) if len(secuencia)<longitud_secuencia else secuencia[-longitud_secuencia:]
        secuencia = np.array([secuencia])
        prediccion = model.predict(secuencia, verbose=0)[0]
        prediccion = np.log(prediccion + 1e-10)/temperatura
        prediccion = np.exp(prediccion)
        prediccion /= prediccion.sum()
        top_indices = np.argsort(prediccion)[-top_k:]
        top_probs = prediccion[top_indices]
        top_probs /= top_probs.sum()
        palabra_idx = np.random.choice(top_indices, p=top_probs)
        palabra = tokenizer.index_word.get(palabra_idx, '')
        texto_actual += ' ' + palabra
    return texto_actual

# ==========================================
# 8. PROGRAMA PRINCIPAL
# ==========================================
def main():
    print("="*70)
    print("PREDICTOR DE PALABRAS CON LSTM - VERSIÓN OPTIMIZADA")
    print("="*70)
    
    ARCHIVO_TEXTO = 'noticias.txt'
    NOMBRE_MODELO = 'modelo_lstm_optimizado'
    FILTRAR_RARAS = True
    VOCAB_MAX = 30000
    LONGITUD_SECUENCIA = 30
    STRIDE = 1
    MAX_SECUENCIAS = 200000
    EMBEDDING_DIM = 32
    LSTM_UNITS = 64
    USAR_BIDIRECCIONAL = False
    NUM_CAPAS_LSTM = 2
    EPOCAS = 50
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.1

    modelo_existe = os.path.exists(f'{NOMBRE_MODELO}.keras')
    if modelo_existe:
        print(f"Modelo existente encontrado: {NOMBRE_MODELO}.keras")
        respuesta = input("¿Deseas (1) Reentrenar o (2) Solo usar el existente? [1/2]: ")
        entrenar = (respuesta == '1')
    else:
        print("No se encontro modelo existente.")
        entrenar = True

    if entrenar:
        texto_crudo = cargar_texto_desde_archivo(ARCHIVO_TEXTO)
        texto = preprocesar_texto_completo(texto_crudo, filtrar_raras=FILTRAR_RARAS)
        tokenizer, vocab_size, idx2word = crear_vocabulario_optimizado(texto, VOCAB_MAX)
        X, y, longitud_secuencia = crear_secuencias_eficiente(texto, tokenizer, LONGITUD_SECUENCIA, STRIDE, MAX_SECUENCIAS)
        model = crear_modelo_optimizado(vocab_size, longitud_secuencia, EMBEDDING_DIM, LSTM_UNITS, USAR_BIDIRECCIONAL, NUM_CAPAS_LSTM)
        model, history = entrenar_modelo_optimizado(model, X, y, EPOCAS, BATCH_SIZE, VALIDATION_SPLIT, NOMBRE_MODELO)
        guardar_modelo_completo(model, tokenizer, longitud_secuencia, {'embedding_dim':EMBEDDING_DIM,'lstm_units':LSTM_UNITS,'bidireccional':USAR_BIDIRECCIONAL,'num_capas':NUM_CAPAS_LSTM}, NOMBRE_MODELO)
        del X, y, texto, texto_crudo
        gc.collect()
    else:
        model, tokenizer, config = cargar_modelo_completo(NOMBRE_MODELO)
        longitud_secuencia = config['longitud_secuencia']

    # Generación de prueba
    frases_prueba = ["el modelo aprende","la red neuronal","el universo contiene","las personas se","la educacion formal"]
    for frase in frases_prueba:
        resultado = generar_texto_mejorado(model, tokenizer, frase, longitud_secuencia, num_palabras=8, temperatura=0.8, top_k=5)
        print(f"→ {resultado}\n")

    print("Programa finalizado!")

if __name__ == "__main__":
    main()
