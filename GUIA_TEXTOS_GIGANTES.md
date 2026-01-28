# GUÍA: USO CON TEXTOS GIGANTES

## Versión Optimizada para Corpus Masivos

Esta versión está diseñada para manejar **textos de miles o millones de palabras** con las mismas técnicas de preprocesamiento que usan modelos como GPT.

## Nuevas Características

### 1. Preprocesamiento Avanzado (Estilo GPT)

#### Normalización Unicode

- Convierte caracteres especiales a ASCII
- Maneja textos multilingües correctamente
- Elimina problemas de encoding

#### Limpieza Profunda

- Normalización de espacios en blanco
- Separación correcta de puntuación
- Eliminación de caracteres especiales
- Tokenización optimizada

#### Filtrado Opcional de Palabras Raras

- Elimina palabras que aparecen pocas veces
- Reduce el vocabulario sin perder información
- Mejora la generalización del modelo

### 2. Tokenización Optimizada

- **Vocabulario limitado**: Control de tamaño máximo (default: 10,000)
- **Token OOV**: Manejo de palabras fuera del vocabulario
- **Estadísticas detalladas**: Análisis completo del corpus

### 3. Generación Eficiente de Secuencias

- **Pre-asignación de memoria**: No usa append (mucho más rápido)
- **Stride configurable**: Controla densidad de secuencias
- **Límite de secuencias**: Evita memoria excesiva
- **Progreso en tiempo real**: Muestra avance cada 10,000 secuencias

### 4. Arquitectura Mejorada

- **Bidirectional LSTM**: Opcional, procesa en ambas direcciones
- **Múltiples capas**: Apilamiento de LSTMs para más capacidad
- **Dropout mejorado**: Regularización en más puntos
- **Embeddings más grandes**: Mayor capacidad de representación

### 5. Entrenamiento Avanzado

- **ModelCheckpoint**: Guarda el mejor modelo automáticamente
- **EarlyStopping**: Para si no hay mejora (evita sobreentrenamiento)
- **ReduceLROnPlateau**: Ajusta learning rate dinámicamente
- **Batches grandes**: Aprovecha GPUs mejor (128 vs 32)
- **Validación**: Split automático para monitorizar overfitting

### 6. Generación de Texto Mejorada

- **Sampling con temperatura**: Control de creatividad
- **Top-k sampling**: Más variedad en las predicciones
- **Modo interactivo mejorado**: Cambiar parámetros en tiempo real

## Cómo Usar Tu Propio Texto

### Opción 1: Desde Archivo

```python
# En la función main(), cambia:
ARCHIVO_TEXTO = 'mi_corpus_gigante.txt'
```

El programa cargará automáticamente tu archivo `.txt`.

### Opción 2: Programático

```python
def cargar_texto_desde_archivo(ruta_archivo=None):
    # Aquí puedes agregar tu código para cargar desde:
    # - Múltiples archivos
    # - Base de datos
    # - API web
    # - Scraping
  
    if ruta_archivo:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return f.read()
```

## Configuración para Diferentes Tamaños de Corpus

### Corpus Pequeño (< 100K palabras)

```python
VOCAB_MAX = 5000
LONGITUD_SECUENCIA = 8
STRIDE = 1
EMBEDDING_DIM = 64
LSTM_UNITS = 128
NUM_CAPAS_LSTM = 1
BATCH_SIZE = 64
```

### Corpus Mediano (100K - 1M palabras)

```python
VOCAB_MAX = 10000
LONGITUD_SECUENCIA = 10
STRIDE = 1
EMBEDDING_DIM = 128
LSTM_UNITS = 256
NUM_CAPAS_LSTM = 2
BATCH_SIZE = 128
```

### Corpus Grande (1M - 10M palabras)

```python
VOCAB_MAX = 20000
LONGITUD_SECUENCIA = 12
STRIDE = 2  # Reduce número de secuencias
MAX_SECUENCIAS = 500000  # Limita memoria
EMBEDDING_DIM = 256
LSTM_UNITS = 512
NUM_CAPAS_LSTM = 2
USAR_BIDIRECCIONAL = True
BATCH_SIZE = 256
EPOCAS = 50
```

### Corpus Gigante (> 10M palabras)

```python
VOCAB_MAX = 30000
LONGITUD_SECUENCIA = 15
STRIDE = 3  # Saltar más secuencias
MAX_SECUENCIAS = 1000000  # Limitar a 1M
FILTRAR_RARAS = True  # Activar filtrado
UMBRAL_FRECUENCIA = 3
EMBEDDING_DIM = 256
LSTM_UNITS = 512
NUM_CAPAS_LSTM = 3
USAR_BIDIRECCIONAL = True
BATCH_SIZE = 512  # Si tienes GPU potente
EPOCAS = 30
```

## Requisitos de Memoria

### Estimación de RAM Necesaria

Para **1 millón de secuencias** de longitud 10:

- Arrays X, y: ~40 MB
- Modelo (256 units, vocab 10K): ~100 MB
- Durante entrenamiento: ~500 MB - 2 GB
- **Total recomendado**: 4-8 GB RAM

Para **10 millones de secuencias**:

- Arrays X, y: ~400 MB
- Modelo: ~200 MB
- Durante entrenamiento: ~2-8 GB
- **Total recomendado**: 16-32 GB RAM

### Optimización de Memoria

Si te quedas sin memoria:

1. **Reduce MAX_SECUENCIAS**:

   ```python
   MAX_SECUENCIAS = 100000  # Limita secuencias
   ```
2. **Aumenta STRIDE**:

   ```python
   STRIDE = 3  # Solo 1 de cada 3 secuencias
   ```
3. **Reduce BATCH_SIZE**:

   ```python
   BATCH_SIZE = 32  # Menos memoria por batch
   ```
4. **Activa filtrado de raras**:

   ```python
   FILTRAR_RARAS = True
   UMBRAL_FRECUENCIA = 5  # Más agresivo
   ```

## Parámetros de Generación

### Temperatura

Controla la "creatividad" de las predicciones:

- **0.3-0.5**: Muy conservador, texto predecible
- **0.8-1.0**: Equilibrado (recomendado)
- **1.2-1.5**: Creativo, más variedad
- **> 1.5**: Muy aleatorio, puede ser incoherente

```python
temperatura = 0.8  # Equilibrado
```

### Top-k

Limita las palabras candidatas:

- **1**: Siempre la más probable (determinista)
- **3-5**: Poco variado pero coherente
- **5-10**: Equilibrado (recomendado)
- **10-20**: Mucha variedad
- **> 50**: Casi sin límite

```python
top_k = 5  # Bueno para empezar
```

### Número de Palabras

```python
num_palabras = 10  # Texto corto
num_palabras = 50  # Párrafo
num_palabras = 200  # Texto largo
```

## Monitoreo Durante Entrenamiento

### Métricas Normales

```
Epoch 1/100
loss: 4.5231 - accuracy: 0.1234 - val_loss: 4.3211 - val_accuracy: 0.1456
```

- **Loss**: Debe bajar constantemente
- **Val_loss**: No debe aumentar mucho (overfitting)
- **Accuracy**: Debe subir gradualmente

### Señales de Problemas

❌ **Overfitting**:

```
loss: 0.5 - val_loss: 2.8  # val_loss mucho mayor
```

Solución: Más dropout, menos épocas, más datos

❌ **Underfitting**:

```
loss: 3.2 - val_loss: 3.3  # Ambos altos
```

Solución: Más épocas, modelo más grande, menos dropout

✅ **Entrenamiento Saludable**:

```
loss: 1.2 - val_loss: 1.4  # Similar, bajando
```

## Solución de Problemas

### Problema: "Out of Memory"

```python
# Reduce estos parámetros:
MAX_SECUENCIAS = 50000
BATCH_SIZE = 32
STRIDE = 3
```

### Problema: Entrenamiento muy lento

```python
# Aumenta batch size (si tienes GPU):
BATCH_SIZE = 256

# O reduce datos:
STRIDE = 2
MAX_SECUENCIAS = 100000
```

### Problema: Predicciones repetitivas

```python
# Al generar, usa:
temperatura = 1.2  # Más variedad
top_k = 10  # Más opciones
```

### Problema: Predicciones incoherentes

```python
# Al generar, usa:
temperatura = 0.6  # Más conservador
top_k = 3  # Menos opciones

# Y entrena más:
EPOCAS = 100
```

## Mejores Prácticas

### 1. Preparación del Corpus

- **Tamaño**: Mínimo 50K palabras, ideal > 500K
- **Calidad**: Texto bien escrito, coherente
- **Variedad**: Diferentes temas y estilos
- **Limpieza**: Eliminar spam, código, símbolos raros

### 2. Entrenamiento

- Empieza con pocos datos/épocas para probar
- Monitorea val_loss constantemente
- Usa callbacks (ya incluidos)
- Guarda checkpoints frecuentemente

### 3. Generación

- Prueba diferentes temperaturas
- Usa top-k para evitar palabras raras
- Semillas más largas = mejor contexto
- Experimenta con los parámetros

## Ejemplo Completo: Corpus de Libros

```python
# 1. Preparar tu corpus
# Descarga libros del Proyecto Gutenberg, Wikipedia, etc.

# 2. Unir en un archivo
with open('corpus_gigante.txt', 'w') as output:
    for archivo in ['libro1.txt', 'libro2.txt', 'libro3.txt']:
        with open(archivo, 'r') as f:
            output.write(f.read() + '\n\n')

# 3. Configurar el programa
ARCHIVO_TEXTO = 'corpus_gigante.txt'
VOCAB_MAX = 20000
LONGITUD_SECUENCIA = 12
STRIDE = 2
EMBEDDING_DIM = 256
LSTM_UNITS = 512
NUM_CAPAS_LSTM = 2
EPOCAS = 50
BATCH_SIZE = 128

# 4. Ejecutar
python predictor_palabras_optimizado.py

# 5. Esperar (puede tardar horas)
# 6. Generar texto increíble!
```

## Siguientes Pasos

1. **Recopila tu corpus**: Cuanto más grande, mejor
2. **Configura parámetros**: Según tamaño de corpus
3. **Entrena**: Puede tardar horas/días
4. **Experimenta**: Prueba diferentes temperaturas
5. **Itera**: Ajusta configuración según resultados

## Tips Avanzados

- Usa Google Colab para GPUs gratis
- Guarda checkpoints cada 10 épocas
- Experimenta con Bidirectional LSTM
- Prueba diferentes longitudes de secuencia
- Combina múltiples fuentes de texto
- Considera usar GRU en vez de LSTM (más rápido)
