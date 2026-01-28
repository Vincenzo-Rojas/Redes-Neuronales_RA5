"""
PREDICTOR DE PALABRAS CON LSTM
Red Neuronal Recurrente para predicción de secuencias de texto
Serie Temporal Discreta

Autor: [Tu nombre y apellidos]
Asignatura: Redes Neuronales - RA5
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# ==========================================
# 1. CARGA Y PREPARACIÓN DEL TEXTO
# ==========================================

def cargar_texto():
    #Carga el texto de entrenamiento.
    
    #Aquí usamos un texto simple para demostrar el concepto.
    texto = """
    el gato come pescado todos los dias. el gato es muy feliz.
    la red neuronal aprende patrones de datos. la red es muy potente.
    el perro juega en el parque. el perro corre muy rapido.
    la inteligencia artificial transforma el mundo. la inteligencia es fascinante.
    el sol brilla en el cielo. el sol da calor.
    la luna ilumina la noche. la luna es hermosa.
    el estudiante aprende redes neuronales. el estudiante trabaja mucho.
    la red lstm recuerda informacion. la red procesa secuencias.
    el ordenador ejecuta programas. el ordenador es rapido.
    la tecnologia avanza cada dia. la tecnologia es increible.
    el aprendizaje profundo usa capas. el aprendizaje mejora con datos.
    la memoria de la red es importante. la memoria ayuda a predecir.
    el modelo neuronal se entrena bien. el modelo genera texto nuevo.
    la prediccion de palabras funciona. la prediccion mejora con practica.
    el texto se procesa correctamente. el texto tiene patrones claros.
    """

    #Aquí usamos un texto complejo.
    '''
    texto = """
    el modelo aprende cuando analiza secuencias de palabras que dependen del contexto previo
    la red neuronal ajusta sus pesos mientras procesa informacion distribuida en el tiempo
    el sistema mejora su prediccion cuando la memoria interna conserva datos relevantes
    la arquitectura lstm permite recordar informacion pasada durante el procesamiento secuencial
    el entrenamiento continuo reduce errores acumulados en la generacion de texto
    la red procesa lenguaje natural sin reglas explicitas pero aprende patrones estadisticos
    el modelo genera frases coherentes cuando el corpus mantiene una estructura estable
    la memoria de la red influye directamente en la salida producida para cada palabra
    el sistema aprende dependencias largas cuando las secuencias son consistentes
    la prediccion mejora si el entrenamiento incluye variaciones sintacticas controladas

    el lenguaje natural presenta relaciones complejas que la red aprende progresivamente
    la secuencia inicial condiciona el desarrollo completo del texto generado
    el modelo interpreta el orden de las palabras como una fuente principal de significado
    la red lstm mantiene estados internos que preservan informacion contextual
    el aprendizaje automatico se basa en ajustar parametros mediante ejemplos repetidos
    el sistema analiza cada palabra teniendo en cuenta el historial procesado
    la red transforma simbolos numericos en representaciones significativas
    el entrenamiento prolongado mejora la estabilidad del modelo generado
    la memoria interna evita que la red pierda informacion importante
    el modelo aprende a anticipar palabras probables en cada contexto

    la coherencia del texto generado depende del equilibrio entre memoria y variacion
    el sistema adapta su salida segun la informacion acumulada previamente
    la red aprende patrones temporales presentes en los datos de entrenamiento
    el modelo procesa frases completas respetando dependencias internas
    el lenguaje secuencial requiere una interpretacion basada en el orden
    la red lstm gestiona dependencias que superan varias palabras consecutivas
    el entrenamiento correcto produce modelos mas robustos
    la prediccion textual refleja probabilidades aprendidas durante el proceso
    el sistema genera texto continuo cuando el contexto se mantiene estable
    la memoria permite conectar ideas separadas en la secuencia

    el aprendizaje profundo captura relaciones no lineales entre palabras
    la red analiza estructuras repetidas presentes en el corpus
    el modelo aprende incluso cuando los datos no son perfectamente regulares
    la secuencia de entrenamiento define el estilo del texto generado
    el sistema mejora cuando se expone a ejemplos variados pero coherentes
    la red ajusta su comportamiento segun el historial reciente
    el modelo aprende representaciones internas del lenguaje
    la prediccion se vuelve mas precisa con datos suficientes
    la memoria interna almacena informacion util para la generacion
    el entrenamiento refuerza patrones frecuentes del lenguaje

    el texto generado mantiene continuidad cuando la red recuerda el contexto
    el sistema procesa informacion palabra a palabra de forma secuencial
    la red lstm evita el olvido rapido de informacion pasada
    el modelo aprende relaciones complejas sin supervision explicita
    la arquitectura del sistema influye en la calidad final del texto
    el lenguaje emerge a partir de estadisticas aprendidas
    la red adapta su salida al estado interno actual
    el entrenamiento repetido estabiliza la generacion
    la memoria actua como un puente entre partes del texto
    el modelo combina informacion pasada y presente

    la prediccion depende del estado interno mantenido por la red
    el sistema aprende estructuras sintacticas a partir de ejemplos
    la red procesa secuencias largas con mayor precision
    el modelo ajusta parametros para minimizar errores
    la coherencia semantica mejora con corpus bien definidos
    el entrenamiento secuencial refuerza dependencias temporales
    la memoria interna sostiene la continuidad del discurso
    el sistema genera texto con patrones reconocibles
    la red aprende progresivamente a mantener consistencia
    el modelo responde segun el contexto acumulado

    el aprendizaje automatico se apoya en datos representativos
    la red lstm conserva informacion relevante a largo plazo
    el texto generado refleja el entrenamiento previo
    el sistema mejora su salida con cada iteracion
    la prediccion se afina mediante ajustes graduales
    la red integra informacion contextual en cada paso
    el modelo aprende sin comprender significado real
    la memoria permite mantener coherencia global
    el entrenamiento adecuado produce texto estable
    el sistema genera secuencias linguisticamente consistentes
    """

    '''

    return texto

def preprocesar_texto(texto):
    """
    Normaliza el texto: minúsculas y limpieza básica.
    """
    texto = texto.lower()
    # Eliminamos signos de puntuación innecesarios
    texto = texto.replace('.', ' .')
    texto = texto.replace(',', ' ,')
    return texto

# ==========================================
# 2. TOKENIZACIÓN Y VOCABULARIO
# ==========================================

def crear_vocabulario(texto):
    """
    Crea un vocabulario: cada palabra única recibe un número entero.
    Esto convierte texto en datos numéricos para la red neuronal.
    """
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts([texto])
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Tamaño del vocabulario: {vocab_size} palabras únicas")
    print(f"Primeras 20 palabras del vocabulario:")
    for palabra, indice in list(tokenizer.word_index.items())[:20]:
        print(f"  '{palabra}' -> {indice}")
    
    return tokenizer, vocab_size

# ==========================================
# 3. CREACIÓN DE SECUENCIAS (SERIE TEMPORAL)
# ==========================================

def crear_secuencias(texto, tokenizer, longitud_secuencia=3):
    """
    Crea secuencias de entrenamiento.
    Cada secuencia de N palabras predice la palabra N+1.
    
    Ejemplo:
        Texto: "el gato come pescado"
        Secuencias generadas:
            [el, gato, come] -> pescado
            [gato, come, pescado] -> (siguiente palabra)
    
    Esto es SERIE TEMPORAL DISCRETA: el orden importa.
    """
    # Convertir texto a secuencia de números
    secuencia_completa = tokenizer.texts_to_sequences([texto])[0]
    
    # Crear ventanas deslizantes
    X = []  # Entrada: secuencias de palabras
    y = []  # Salida: siguiente palabra
    
    for i in range(longitud_secuencia, len(secuencia_completa)):
        # Tomamos N palabras como entrada
        secuencia_entrada = secuencia_completa[i-longitud_secuencia:i]
        # Y la siguiente palabra como salida
        palabra_salida = secuencia_completa[i]
        
        X.append(secuencia_entrada)
        y.append(palabra_salida)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nSecuencias de entrenamiento creadas:")
    print(f"  Total de secuencias: {len(X)}")
    print(f"  Forma de X: {X.shape}")
    print(f"  Forma de y: {y.shape}")
    
    # Mostrar ejemplos
    print(f"\nEjemplos de secuencias:")
    for i in range(min(3, len(X))):
        entrada = [list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(idx)] 
                   for idx in X[i]]
        salida = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(y[i])]
        print(f"  {entrada} -> '{salida}'")
    
    return X, y, longitud_secuencia

# ==========================================
# 4. ARQUITECTURA DE LA RED NEURONAL LSTM
# ==========================================

def crear_modelo(vocab_size, longitud_secuencia, embedding_dim=50, lstm_units=150):
    """
    Crea la arquitectura de la red neuronal recurrente con LSTM.
    
    Capas:
    1. Embedding: Convierte palabras (números) en vectores densos
    2. LSTM: Aprende dependencias temporales en las secuencias
    3. Dense + Softmax: Calcula probabilidades para cada palabra del vocabulario
    
    ¿Por qué LSTM y no RNN simple?
    - Las RNN simples sufren el problema del "vanishing gradient"
    - LSTM tiene puertas (gates) que permiten recordar información a largo plazo
    - LSTM es el estándar para procesamiento de texto y series temporales
    """
    model = Sequential([
        # Capa 1: Embedding
        # Transforma cada palabra en un vector de embedding_dim dimensiones
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=longitud_secuencia,
                 name='embedding'),
        
        # Capa 2: LSTM
        # Procesa la secuencia de palabras y aprende dependencias temporales
        # lstm_units: numero de unidades de memoria interna
        # dropout: desactiva aleatoriamente conexiones de entrada para evitar sobreajuste
        # recurrent_dropout: regulariza las conexiones recurrentes de la memoria interna
        # Esta regularizacion mejora la capacidad de generalizacion del modelo
        # y evita que la red memorice secuencias exactas del entrenamiento
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, name='lstm'),
        
        # Capa 3: Dense con Softmax
        # Convierte el estado de LSTM en probabilidades para cada palabra
        Dense(vocab_size, activation='softmax', name='output')
    ])
    
    print("\n" + "="*60)
    print("ARQUITECTURA DE LA RED NEURONAL")
    print("="*60)
    model.summary()
    
    return model

# ==========================================
# 5. ENTRENAMIENTO DEL MODELO
# ==========================================

def entrenar_modelo(model, X, y, epocas=50, batch_size=32):
    """
    Entrena la red neuronal.
    
    Parámetros:
    - categorical_crossentropy: función de pérdida para clasificación multiclase
    - adam: optimizador adaptativo (eficiente y popular)
    - épocas: número de veces que la red ve todo el dataset
    
    Durante el entrenamiento:
    - La red ajusta sus pesos para minimizar el error
    - En cada época, aprende mejores patrones
    - La pérdida (loss) debe disminuir con el tiempo
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*60)
    
    # Compilar el modelo
    model.compile(
        loss='sparse_categorical_crossentropy',  # Para labels enteros
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print(f"\nParámetros de entrenamiento:")
    print(f"  - Épocas: {epocas}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Optimizador: Adam")
    print(f"  - Función de pérdida: Sparse Categorical Crossentropy")
    print(f"\nIniciando entrenamiento...\n")
    
    # Entrenar
    history = model.fit(
        X, y,
        epochs=epocas,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1  # 10% para validación
    )
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Pérdida final: {history.history['loss'][-1]:.4f}")
    print(f"Precisión final: {history.history['accuracy'][-1]:.4f}")
    
    return model, history

# ==========================================
# 6. GUARDADO DEL MODELO
# ==========================================

def guardar_modelo(model, tokenizer, longitud_secuencia, nombre='modelo_lstm'):
    """
    Guarda el modelo entrenado y el tokenizer.
    Esto permite reutilizar el modelo sin reentrenar.
    """
    # Guardar el modelo
    model.save(f'{nombre}.keras')
    
    # Guardar el tokenizer
    with open(f'{nombre}_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Guardar configuración
    config = {'longitud_secuencia': longitud_secuencia}
    with open(f'{nombre}_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"\n✓ Modelo guardado como '{nombre}.keras'")
    print(f"✓ Tokenizer guardado como '{nombre}_tokenizer.pkl'")
    print(f"✓ Configuración guardada como '{nombre}_config.pkl'")

# ==========================================
# 7. CARGA DEL MODELO
# ==========================================

def cargar_modelo_entrenado(nombre='modelo_lstm'):
    """
    Carga un modelo previamente entrenado.
    """
    # Cargar modelo
    model = load_model(f'{nombre}.keras')
    
    # Cargar tokenizer
    with open(f'{nombre}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Cargar configuración
    with open(f'{nombre}_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✓ Modelo cargado desde '{nombre}.keras'")
    return model, tokenizer, config['longitud_secuencia']

# ==========================================
# 8. GENERACIÓN DE TEXTO (USO DEL MODELO)
# ==========================================

def generar_texto(model, tokenizer, texto_inicial, longitud_secuencia, num_palabras=5):
    """
    Genera texto a partir de una frase inicial.
    
    Proceso:
    1. Toma las últimas N palabras del texto inicial
    2. Predice la siguiente palabra
    3. Añade esa palabra al texto
    4. Repite el proceso
    
    Esto demuestra que la red ha aprendido patrones en el texto.
    """
    # Normalizar texto inicial
    texto_actual = texto_inicial.lower()
    
    print(f"\nTexto inicial: '{texto_inicial}'")
    print(f"Generando {num_palabras} palabras adicionales...\n")
    
    for _ in range(num_palabras):
        # Tokenizar las últimas palabras
        secuencia = tokenizer.texts_to_sequences([texto_actual])[0]
        
        # Tomar solo las últimas N palabras
        if len(secuencia) >= longitud_secuencia:
            secuencia = secuencia[-longitud_secuencia:]
        else:
            # Si es muy corto, rellenar con ceros
            secuencia = [0] * (longitud_secuencia - len(secuencia)) + secuencia
        
        # Preparar entrada
        secuencia = np.array([secuencia])
        
        # Predecir siguiente palabra
        prediccion = model.predict(secuencia, verbose=0)
        palabra_predicha_idx = np.argmax(prediccion)
        
        # Convertir índice a palabra
        for palabra, idx in tokenizer.word_index.items():
            if idx == palabra_predicha_idx:
                texto_actual += " " + palabra
                break
    
    return texto_actual

# ==========================================
# 9. PROGRAMA PRINCIPAL
# ==========================================

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print("="*60)
    print("PREDICTOR DE PALABRAS CON LSTM")
    print("Red Neuronal Recurrente - Serie Temporal Discreta")
    print("="*60)
    
    # Configuración
    LONGITUD_SECUENCIA = 8  # Número de palabras para predecir la siguiente
    EPOCAS = 100
    NOMBRE_MODELO = 'modelo_lstm_predictor'
    
    # Verificar si ya existe un modelo entrenado
    if os.path.exists(f'{NOMBRE_MODELO}.keras'):
        print("\n¡Modelo ya entrenado encontrado!")
        respuesta = input("¿Deseas reentrenar? (s/n): ")
        
        if respuesta.lower() != 's':
            print("\nCargando modelo existente...")
            model, tokenizer, longitud_secuencia = cargar_modelo_entrenado(NOMBRE_MODELO)
        else:
            entrenar = True
    else:
        entrenar = True
    
    if entrenar or respuesta.lower() == 's':
        # Paso 1: Cargar y preprocesar texto
        print("\n[1] Cargando y preprocesando texto...")
        texto = cargar_texto()
        texto = preprocesar_texto(texto)
        
        # Paso 2: Crear vocabulario
        print("\n[2] Creando vocabulario...")
        tokenizer, vocab_size = crear_vocabulario(texto)
        
        # Paso 3: Crear secuencias
        print("\n[3] Creando secuencias de entrenamiento...")
        X, y, longitud_secuencia = crear_secuencias(texto, tokenizer, LONGITUD_SECUENCIA)
        
        # Paso 4: Crear modelo
        print("\n[4] Creando arquitectura de la red...")
        model = crear_modelo(vocab_size, longitud_secuencia)
        
        # Paso 5: Entrenar
        print("\n[5] Entrenando el modelo...")
        model, history = entrenar_modelo(model, X, y, epocas=EPOCAS)
        
        # Paso 6: Guardar
        print("\n[6] Guardando el modelo...")
        guardar_modelo(model, tokenizer, longitud_secuencia, NOMBRE_MODELO)
    
    # Paso 7: Generar texto con diferentes entradas
    print("\n" + "="*60)
    print("GENERACIÓN DE TEXTO - PRUEBAS")
    print("="*60)
    
    # Diferentes frases iniciales para demostrar versatilidad
    frases_prueba = [
        "el gato",
        "la red neuronal",
        "el estudiante",
        "la tecnologia",
        "el modelo"
    ]
    
    print("\nGenerando texto con diferentes entradas:\n")
    for frase in frases_prueba:
        resultado = generar_texto(model, tokenizer, frase, longitud_secuencia, num_palabras=6)
        print(f"→ {resultado}")
        print()
    
    # Modo interactivo
    print("\n" + "="*60)
    print("MODO INTERACTIVO")
    print("="*60)
    print("Introduce frases iniciales para generar texto.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        frase_usuario = input("Texto inicial: ")
        if frase_usuario.lower() == 'salir':
            break
        
        resultado = generar_texto(model, tokenizer, frase_usuario, longitud_secuencia, num_palabras=7)
        print(f"Resultado: {resultado}\n")
    
    print("\n¡Programa finalizado!")

if __name__ == "__main__":
    main()
