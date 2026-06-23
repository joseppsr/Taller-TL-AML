# Taller de Valero — Puntos clave y cosas a considerar

> Guía práctica para preparar el taller a partir del transcript.  
> Objetivo: tener claro **qué hay que construir**, **qué entregar** y **qué errores evitar**.

---

## 1. Objetivo general del taller

El taller consiste en construir una aplicación práctica de deep learning sobre un problema de **predicción bancaria de crédito**.

La idea principal es entrenar un modelo que ayude a decidir si a una persona se le debería conceder un crédito o no, teniendo en cuenta:

- Predicción del riesgo de impago.
- Uso de una **capa customizada**.
- Uso de una **función de coste customizada** orientada a aprendizaje justo.
- Optimización automática de arquitectura con **Keras Tuner**.
- Estimación de la **incertidumbre/error de salida** del modelo.

Lo importante no es ganar una competición de accuracy, sino demostrar que se entienden y se aplican correctamente los conceptos.

---

## 2. Entregables

### Obligatorio

- Repositorio de **GitHub** con el código.
- Presentación de resultados de **5 minutos** la semana siguiente.
- Si el grupo no puede presentar en directo:
  - entregar el **PDF de la presentación**;
  - entregar también un **vídeo de 5 minutos** hecho por el grupo.

### Importante

- La presentación o vídeo debe durar **5 minutos**.
- Si un grupo se alarga mucho, por ejemplo 15 minutos, puede tener penalización.
- El trabajo se hace en grupos asignados aleatoriamente.

---

## 3. Dataset que hay que usar

### Dataset correcto

Hay que usar los **datos del taller**, no los datos de ejemplo usados en clase.

Valero insiste explícitamente en:

> No usar los datos de ejemplo. Usar los datos del taller.

### Características del dataset del taller

- Es un dataset de crédito bancario.
- Tiene aproximadamente **300.000 personas**.
- Tiene alrededor de **122 características** por persona.
- La variable objetivo indica si la persona ha tenido problemas devolviendo el crédito.
- La variable sensible principal es el **género**.
- Hay variables financieras, demográficas y de historial crediticio.

Ejemplos de variables mencionadas:

- ingresos anuales;
- cantidad de crédito solicitada;
- anualidad del crédito;
- edad;
- si tiene coche;
- género;
- fuentes externas de evaluación crediticia;
- historial de créditos anteriores;
- pagos e impagos mensuales.

### Datos extra

El dataset incluye tablas adicionales con información complementaria, por ejemplo:

- otros créditos pedidos por la persona;
- si devolvió o no créditos anteriores;
- información mensual de pagos e impagos.

No es obligatorio complicarse usando todas esas tablas. Se puede hacer el taller con la tabla principal. Si el grupo quiere profundizar más, puede incorporar información adicional.

---

## 4. Datos de ejemplo usados por Valero

Valero usa en clase otro dataset solo para explicar el flujo de trabajo.

Ese dataset de ejemplo es similar, pero **no es el que hay que entregar**.

Características del dataset de ejemplo:

- unas **32.000 personas**;
- unas **14 variables de entrada** inicialmente;
- objetivo: predecir si una persona cobra más o menos de 50.000 dólares;
- variables sensibles posibles: género, raza y edad.

Este dataset sirve para entender la metodología, pero el trabajo final debe hacerse con el dataset bancario del taller.

---

## 5. Tareas técnicas obligatorias

El modelo final debe incluir estas cuatro piezas:

1. **Capa customizada**.
2. **Función de coste customizada** para aprendizaje justo.
3. **Keras Tuner** para probar distintas arquitecturas.
4. **Medición o predicción de incertidumbre** en la salida.

---

## 6. Pipeline recomendado

### Paso 1: Explorar los datos

Antes de entrenar modelos, conviene hacer un análisis exploratorio:

- revisar tamaño del dataset;
- revisar columnas disponibles;
- identificar variable objetivo;
- identificar variable sensible;
- mirar distribuciones marginales;
- mirar relaciones entre variables;
- detectar variables categóricas;
- detectar posibles valores nulos o raros;
- revisar desbalance de clases.

Valero recomienda hacer visualizaciones para ver:

- distribución de cada variable;
- relación entre variables;
- relación de variables financieras con el target;
- relación entre género y predicción/target.

---

### Paso 2: Separar train, validation y test

No normalizar todo el dataset antes de separar.

Orden correcto:

1. Separar datos:
   - train: 80%;
   - validation: 10%;
   - test: 10%.
2. Ajustar transformaciones solo con train.
3. Aplicar esas transformaciones a validation y test.

Esto evita leakage de información del validation/test hacia el entrenamiento.

---

### Paso 3: Preprocesar correctamente

Consideraciones importantes:

- Las variables numéricas se pueden estandarizar.
- El `scaler` debe ajustarse solo con train.
- Las variables categóricas no ordinales deben convertirse con **one-hot encoding**.
- No conviene codificar categorías como `0, 1, 2, 3...` si esos números no tienen orden real.
- Después del one-hot encoding, el índice de columnas puede cambiar.
- Si se usa una variable sensible, comprobar bien en qué columna queda tras el preprocesado.
- Algunas transformaciones generan matrices dispersas (`sparse matrices`); si el modelo o ciertos cálculos no las aceptan, convertirlas a array normal.

Ejemplo comentado en clase:

- Género estaba como variable sensible.
- En el ejemplo, tras one-hot encoding de otras variables, la columna del género acabó desplazándose.

---

## 7. Modelo base

Antes de añadir partes custom, conviene construir un modelo base simple.

Para clasificación binaria:

- salida con una neurona;
- activación `sigmoid`;
- loss base: `binary_crossentropy`;
- métrica: `accuracy`.

La sigmoid se usa porque la salida debe estar entre 0 y 1.

Ejemplo conceptual:

```python
model = keras.Sequential([
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Este modelo sirve como baseline para comparar el impacto de la capa custom, la función de coste justa y la incertidumbre.

---

## 8. Capa customizada

El taller pide que haya alguna parte de la arquitectura customizada, por ejemplo una capa propia.

Valero menciona varias ideas:

### Opción 1: ratios financieros

Crear una capa que calcule ratios entre variables financieras, por ejemplo:

- crédito / ingresos;
- anualidad / ingresos;
- deuda / capacidad de pago.

Cuidado: calcular ratios entre todas las variables puede generar demasiadas dimensiones.

### Opción 2: capa de exponentes entrenables

Valero desarrolla como ejemplo una capa que aplica a cada variable de entrada un exponente entrenable.

Idea:

```python
x_i -> x_i ** alpha_i
```

Donde cada `alpha_i` es entrenable.

Consideraciones:

- inicializar los exponentes a `1` para que al principio la capa no cambie los datos;
- restringir los exponentes, por ejemplo entre `0.1` y `3`;
- usar `keras.ops` si se quiere compatibilidad con distintos backends;
- controlar que la capa no desestabilice el entrenamiento.

Interpretación:

- exponentes menores que 1 comprimen valores grandes;
- exponentes mayores que 1 amplifican valores grandes;
- puede ser útil en variables financieras con distribuciones muy asimétricas.

---

## 9. Función de coste customizada para aprendizaje justo

El objetivo de la función de coste justa es reducir la dependencia entre las predicciones del modelo y una variable sensible.

En este taller, la variable sensible principal es el **género**.

### Idea general

Queremos que las predicciones del modelo sean lo más independientes posible del género.

Una forma simple:

```python
loss_total = loss_base + lambda_fair * penalizacion_fairness
```

Donde:

```python
penalizacion_fairness = corr(y_pred, variable_sensible) ** 2
```

La correlación se eleva al cuadrado porque el objetivo no es hacerla negativa, sino acercarla a cero.

### Puntos clave

- Si minimizas directamente la correlación, el modelo podría llevarla hacia `-1`.
- Lo correcto es penalizar su valor absoluto o su cuadrado.
- `lambda_fair` controla cuánto peso se da a la justicia frente a la precisión.
- Un `lambda_fair` más alto suele reducir la correlación con la variable sensible.
- Pero también puede bajar la accuracy.
- Hay que analizar el trade-off entre **accuracy** y **fairness**.

### Trade-off accuracy vs fairness

Valero explica que una forma razonable de trabajar es probar varios valores de `lambda_fair` y observar:

- accuracy;
- correlación con la variable sensible;
- comportamiento en validation;
- posible pérdida de rendimiento.

La decisión final puede depender del criterio del equipo, del contexto o de requisitos regulatorios.

---

## 10. Por qué no basta con eliminar la variable sensible

Una pregunta importante del taller fue:

> Si no queremos discriminar por género o raza, ¿por qué no quitamos directamente esa columna?

Respuesta clave:

Eliminar la variable sensible no garantiza que el modelo deje de discriminar.

Motivo:

- otras variables pueden actuar como proxies;
- por ejemplo, código postal, ingresos, historial o variables socioeconómicas pueden contener información indirecta sobre género, raza u otras variables sensibles.

Por eso puede ser mejor darle al modelo la variable sensible y penalizar explícitamente que las predicciones dependan de ella.

---

## 11. Keras Tuner

Hay que aplicar **Keras Tuner** sobre el modelo.

No hay un número fijo de modelos que haya que entrenar. Lo importante es demostrar que se sabe usar para probar distintas arquitecturas.

Se pueden tunear, por ejemplo:

- número de capas;
- número de neuronas por capa;
- funciones de activación;
- learning rate;
- dropout, si se usa;
- posición de la capa customizada;
- hiperparámetros relacionados con la arquitectura.

Valero deja claro que no se trata de entrenar miles de modelos sin sentido, sino de mostrar que se entiende el proceso.

---

## 12. Incertidumbre o error de salida

El taller pide medir la incertidumbre de la red en la salida.

Valero propone una forma sencilla:

1. Entrenar un primer modelo que predice el target.
2. Calcular el error real de ese modelo:

```python
error = abs(y_pred - y_true)
```

3. Entrenar un segundo modelo para predecir ese error.
4. Usar ese segundo modelo como estimador de incertidumbre.

### Interpretación

El segundo modelo no predice si se concede o no el crédito.

Predice cuándo el primer modelo probablemente se va a equivocar.

Esto puede servir para:

- detectar casos dudosos;
- mandar casos a revisión humana;
- generar alertas;
- identificar perfiles donde el modelo no es fiable.

### Consideraciones

- El error debe ser no negativo.
- La salida del modelo de incertidumbre puede usar `ReLU`.
- Métrica útil: `MAE`.
- Se puede entrenar solo con `X`, o con `X` más la predicción del primer modelo.
- Una predicción cercana a `0.5` ya indica cierta duda, pero no equivale exactamente a una estimación formal del error.

---

## 13. Evaluación recomendada

Para presentar bien el taller, conviene comparar varios aspectos:

### Modelo predictivo

- accuracy;
- loss;
- curva de entrenamiento;
- curva de validación;
- posible overfitting;
- resultados en test final.

### Fairness

- correlación entre predicciones y género;
- comparación antes/después de la función de coste justa;
- efecto de distintos valores de `lambda_fair`;
- trade-off entre accuracy y fairness.

### Incertidumbre

- error real vs error predicho;
- casos donde el modelo anticipa error alto;
- ejemplos de decisiones que deberían ir a revisión humana.

### Keras Tuner

- espacio de búsqueda utilizado;
- mejor arquitectura encontrada;
- comparación con baseline.

---

## 14. Errores técnicos que Valero destacó

### 1. Orden de argumentos en funciones de coste

En funciones simétricas como MSE, a veces da igual pasar primero `y_true` o `y_pred`.

Pero en una función custom no simétrica, el orden importa.

Usar siempre:

```python
loss(y_true, y_pred)
```

y comprobar que internamente se interpreta así.

---

### 2. Dimensiones de `y`

Cuidado con la forma de `y`.

Preferible:

```python
(n, 1)
```

en lugar de:

```python
(n,)
```

Keras puede corregir esto internamente en un modelo, pero si aplicas la función de coste directamente puedes obtener resultados erróneos.

---

### 3. Broadcasting silencioso

Python/Numpy/TensorFlow pueden hacer broadcasting sin avisar.

Esto puede provocar que una resta o una operación se haga con dimensiones incorrectas, generando resultados que parecen válidos pero no lo son.

Comprobar siempre shapes:

```python
print(X_train.shape)
print(y_train.shape)
print(y_pred.shape)
```

---

### 4. Normalización antes del split

No normalizar todo el dataset antes de separar train/validation/test.

Primero se separa; luego se ajusta el scaler con train.

---

### 5. Variables categóricas mal codificadas

No convertir categorías sin orden a números ordinales.

Ejemplo incorrecto:

```python
raza = 0, 1, 2, 3, 4
```

si esos números no representan un orden real.

Mejor:

```python
one-hot encoding
```

---

### 6. Reutilizar modelos ya entrenados

Si se comparan modelos, asegurarse de crear un modelo nuevo desde cero.

Si se reentrena accidentalmente un modelo ya entrenado, las curvas pueden parecer raras o planas.

---

### 7. Código antiguo de `tf.keras`

Valero comenta que algunas sugerencias de IA mezclan código antiguo de `tensorflow.keras` con Keras moderno.

Si se trabaja con Keras 3, conviene usar:

```python
keras.ops
```

cuando sea posible.

---

### 8. Correlación en la loss

Para fairness, no minimizar la correlación bruta.

Usar:

```python
corr(y_pred, s) ** 2
```

o una penalización equivalente que tienda a cero.

---

### 9. División por cero en correlaciones

Al implementar correlaciones diferenciables, añadir un pequeño epsilon en el denominador para evitar divisiones por cero.

---

### 10. Índices de columnas tras one-hot

Después de aplicar one-hot encoding, el índice de la variable sensible puede cambiar.

Verificarlo antes de calcular fairness.

---

## 15. Estructura sugerida para el repositorio

```text
repo/
├── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_baseline.ipynb
│   ├── 03_custom_layer_and_loss.ipynb
│   ├── 04_keras_tuner.ipynb
│   └── 05_uncertainty.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── custom_layers.py
│   ├── custom_losses.py
│   └── evaluation.py
├── reports/
│   └── presentation.pdf
└── requirements.txt
```

El `README.md` debería explicar:

- objetivo del taller;
- dataset usado;
- variable objetivo;
- variable sensible;
- pipeline seguido;
- modelos entrenados;
- principales resultados;
- cómo ejecutar el código.

---

## 16. Estructura recomendada para la presentación de 5 minutos

### Slide 1 — Problema y dataset

- Qué se predice.
- Qué variable sensible se usa.
- Qué datos se han usado.

Tiempo aproximado: 30–45 segundos.

---

### Slide 2 — Preprocesado y EDA

- Split train/validation/test.
- Normalización correcta.
- One-hot encoding.
- Distribuciones importantes.
- Posibles sesgos detectados.

Tiempo aproximado: 45 segundos.

---

### Slide 3 — Modelo y componentes custom

- Baseline.
- Capa customizada.
- Función de coste custom.
- Cómo se penaliza la dependencia con género.

Tiempo aproximado: 1 minuto.

---

### Slide 4 — Keras Tuner y resultados

- Qué hiperparámetros se probaron.
- Mejor arquitectura.
- Accuracy/loss.
- Comparación con baseline.

Tiempo aproximado: 1 minuto.

---

### Slide 5 — Fairness e incertidumbre

- Correlación con variable sensible antes/después.
- Trade-off accuracy vs fairness.
- Estimación de incertidumbre.
- Casos que podrían requerir revisión humana.

Tiempo aproximado: 1–1,5 minutos.

---

## 17. Checklist final antes de entregar

### Dataset

- [ ] He usado los datos del taller, no los datos de ejemplo.
- [ ] He identificado correctamente el target.
- [ ] He identificado correctamente la variable sensible.
- [ ] He revisado las variables principales.

### Preprocesado

- [ ] He separado train/validation/test antes de normalizar.
- [ ] He ajustado el scaler solo con train.
- [ ] He aplicado el scaler a validation y test.
- [ ] He codificado bien las variables categóricas.
- [ ] He comprobado las shapes de `X`, `y` y `s`.

### Modelo

- [ ] Tengo un baseline.
- [ ] Tengo una capa customizada.
- [ ] Tengo una función de coste customizada.
- [ ] He usado Keras Tuner.
- [ ] He estimado la incertidumbre/error de salida.

### Fairness

- [ ] He medido dependencia entre predicciones y género.
- [ ] He probado una penalización de fairness.
- [ ] He comparado accuracy vs fairness.
- [ ] He explicado el valor de `lambda_fair`.

### Evaluación

- [ ] He usado validation durante el desarrollo.
- [ ] He reservado test para la evaluación final.
- [ ] He revisado curvas de entrenamiento.
- [ ] He comprobado overfitting.
- [ ] He incluido resultados claros en la presentación.

### Entrega

- [ ] Código en GitHub.
- [ ] Presentación de 5 minutos.
- [ ] PDF preparado.
- [ ] Vídeo de 5 minutos si el grupo no presenta en directo.
- [ ] README claro.

---

## 18. Qué priorizar

Valero deja claro que se valoran más los conceptos que hacer el proyecto enorme.

Priorizar:

1. Que el pipeline esté bien hecho.
2. Que la capa customizada tenga sentido.
3. Que la función de coste justa esté bien planteada.
4. Que se entienda el trade-off fairness/accuracy.
5. Que Keras Tuner se use correctamente.
6. Que la incertidumbre esté explicada de forma útil.
7. Que la presentación sea clara y no se pase de tiempo.

No hace falta:

- usar todas las tablas extra;
- entrenar miles de modelos;
- buscar el mejor score posible;
- hacer una solución excesivamente compleja.

---

## 19. Resumen ultra corto

Hay que construir un modelo de clasificación para crédito bancario usando los datos del taller. El modelo debe incluir una capa customizada, una función de coste customizada para reducir dependencia con género, optimización con Keras Tuner y una estimación de incertidumbre/error. La entrega es GitHub + presentación de 5 minutos, o PDF + vídeo si no se presenta en directo. Lo más importante es demostrar comprensión técnica y evitar errores de preprocesado, shapes, leakage y fairness mal planteado.
