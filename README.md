# PREDICTOR DE PALABRAS CON LSTM - VERSION OPTIMIZADA

## Descripcion del Proyecto

Este proyecto implementa un predictor de palabras utilizando una Red Neuronal Recurrente LSTM (Long Short-Term Memory) optimizada para corpus grandes.  

**Nota:** La descripción completa del proyecto, corpus utilizado, preprocesamiento, arquitectura del modelo, entrenamiento y generación de texto se encuentra en la memoria del proyecto:  
- `memoria_RA5.docx`, Capítulos 1 a 7.

## Requisitos del Sistema

### Software Necesario
Python 3.8 o superior
TensorFlow 2.x
NumPy
Instalacion de Dependencias
```bash
pip install tensorflow numpy
```

## Estructura del Proyecto
```bash
VicenzoRojas_RA5/
|
|-- documento
|   |-- memoria_RA5           		# Memoria del proyecto
|
|-- programa
    |-- predictor_palabras_optimizado.py    # Programa principal
    |-- visualizar_modelo.py                # Script de analisis del modelo
    |-- memoria_RA5.docx                    # Documentacion completa
    |-- README.md                           # Este archivo
    |-- noticias_dataset.csv		        # Archivo del dataset original
    |-- convertir.py                        # Script de tranformacion de .csv a .txt
    |-- corpus.txt                          # Archivo de corpus
    |
    |-- (Archivos generados tras entrenamiento)
        |-- modelo_lstm_optimizado.keras           # Modelo entrenado
        |-- modelo_lstm_optimizado_best.keras      # Mejor checkpoint
        |-- modelo_lstm_optimizado_tokenizer.pkl   # Vocabulario
        |-- modelo_lstm_optimizado_config.pkl      # Configuracion
```
## Como Usar el Proyecto
Ejecucion Basica
```bash
python predictor_palabras_optimizado.py
```
### Primera Ejecucion
Si no existe un modelo entrenado, el programa:

- Carga el corpus desde archivo o genera uno de ejemplo
- Preprocesa el texto con tecnicas avanzadas
- Crea el vocabulario (hasta 30,000 palabras)
- Genera secuencias de entrenamiento
- Construye la arquitectura LSTM
- Entrena el modelo (puede tardar minutos u horas)
- Guarda el modelo entrenado
- Realiza generacion de texto de prueba
- Entra en modo interactivo

### Ejecuciones Posteriores
Si ya existe un modelo entrenado:

Modelo existente encontrado: modelo_lstm_optimizado.keras
¿Deseas (1) Reentrenar o (2) Solo usar el existente? [1/2]: 
Opcion 1: Reentrena desde cero

Opcion 2: Carga el modelo existente y pasa directo a generacion

### Uso con Tu Propio Corpus
Para entrenar con tu propio texto, modifica la variable en el codigo:

### En la funcion main()
ARCHIVO_TEXTO = 'mi_corpus.txt'  # Cambia a tu archivo
El archivo debe ser un .txt con codificacion UTF-8.

**Configuracion del Modelo**
Parametros Principales
Los siguientes parametros pueden ajustarse en la funcion main():

**Preprocesamiento**
FILTRAR_RARAS = True           # True para eliminar palabras poco frecuentes del texto
UMBRAL_FRECUENCIA = 20         # Minimo de apariciones para mantener una palabra

**Tokenizacion**
VOCAB_MAX = 30000              # Tamaño maximo del vocabulario

**Secuencias**
LONGITUD_SECUENCIA = 30        # Número de palabras de contexto para cada secuencia
STRIDE = 1                     # Paso entre secuencias (1=todas, 2=cada 2, etc.)
MAX_SECUENCIAS = 300000        # Limite máximo de secuencias generadas (None=todas)

**Arquitectura**
EMBEDDING_DIM = 64             # Dimensión de los vectores de embedding de cada palabra
LSTM_UNITS = 256               # Número de unidades en cada capa LSTM
USAR_BIDIRECCIONAL = False     # Indica si la LSTM procesa la secuencia en ambas direcciones
NUM_CAPAS_LSTM = 2             # Número de capas LSTM apiladas

**Entrenamiento**
EPOCAS = 30                    # Número máximo de pasadas sobre el conjunto de entrenamiento
BATCH_SIZE = 64                # Número de secuencias procesadas antes de actualizar los pesos
VALIDATION_SPLIT = 0.2         # Porcentaje de datos reservados para validación

## Recomendaciones Segun Tamano del Corpus
### Corpus Pequeno (< 100K palabras)

VOCAB_MAX = 5000
LONGITUD_SECUENCIA = 10
EMBEDDING_DIM = 32
LSTM_UNITS = 64
NUM_CAPAS_LSTM = 1
### Corpus Mediano (100K - 1M palabras)

VOCAB_MAX = 10000
LONGITUD_SECUENCIA = 20
EMBEDDING_DIM = 64
LSTM_UNITS = 128
NUM_CAPAS_LSTM = 2
### Corpus Grande (> 1M palabras)

VOCAB_MAX = 30000
LONGITUD_SECUENCIA = 30
STRIDE = 2
MAX_SECUENCIAS = 500000
EMBEDDING_DIM = 128
LSTM_UNITS = 256
NUM_CAPAS_LSTM = 2
USAR_BIDIRECCIONAL = True

## Generacion de Texto
### Modo Automatico
Al finalizar el entrenamiento o cargar un modelo existente, el programa genera automaticamente texto con varias semillas de prueba:

Generando texto con semilla: 'el modelo aprende'
el modelo aprende cuando analiza secuencias de palabras...

### Modo Interactivo
El programa incluye un modo interactivo donde puedes:

#### MODO INTERACTIVO
Comandos:
  - Escribe una frase para generar texto
  - 'config' para cambiar parametros de generacion
  - 'salir' para terminar

Texto inicial: la inteligencia artificial
Generando texto con semilla: 'la inteligencia artificial'
la inteligencia artificial transforma el mundo ...

Texto inicial: config
  Temperatura (actual: 0.8): 1.2
  Top-k (actual: 5): 10
  Palabras (actual: 10): 20
Configuracion actualizada

Texto inicial: salir
Programa finalizado!
## Parametros de Generacion
### Temperatura

0.3-0.5: Conservador, texto predecible
0.8-1.0: Equilibrado (recomendado)
1.2-1.5: Creativo, mas variedad
1.5: Muy aleatorio

### Top-k

1: Determinista (siempre la mas probable)
3-5: Poco variado pero coherente
5-10: Equilibrado (recomendado)
10-20: Mucha variedad

### Numero de Palabras

5-10: Frases cortas
10-20: Parrafos breves
20-50: Texto extenso

## Visualizacion del Modelo
Para analizar el modelo entrenado:

```bash
python visualizar_modelo.py
```
Este script muestra:

- Informacion del vocabulario
- Arquitectura completa de la red
- Detalles de cada capa
- Estadisticas del modelo
- Flujo de datos


## Callbacks de Entrenamiento

La información detallada sobre los callbacks utilizados durante el entrenamiento (ModelCheckpoint, EarlyStopping y ReduceLROnPlateau) se encuentra en la **documentación del proyecto, sección 5.1 "Configuración" y 5.3 "Evolución de epochs" de la memoria RA5**.

## Monitorización del Entrenamiento

La información sobre cómo se monitoriza el entrenamiento, la interpretación de métricas (loss, val_loss, accuracy, val_accuracy) y las señales de problemas como overfitting o underfitting se encuentra en la **documentación del proyecto, sección 5.3 "Evolución de epochs" y 6.4 "Capacidades y limitaciones" de la memoria RA5**.


### Optimización de Memoria y Buenas Prácticas

La información sobre cómo optimizar el uso de memoria para corpus grandes, solucionar problemas de entrenamiento y generación de texto, así como las mejores prácticas durante la preparación del corpus, entrenamiento y generación, está disponible en la **documentación del proyecto, memoria RA5, apartados 4.4, 4.6, 5.3, 7.1–7.2 y 8.3**.

## Informacion del Proyecto
**Autor: Vincenzo Rojas Carrera**
**Asignatura: Ampliacion de Programacion**
**Trabajo: RA5 - Redes Neuronales**