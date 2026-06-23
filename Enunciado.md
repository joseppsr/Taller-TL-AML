# TALLER: B4-T1

## Diseño de Redes Confiables (Justicia e Incertidumbre)

### Datos básicos

- Grupos de 3 estudiantes.
- Entrega el día 25 junio a través del aula virtual a las 18:00.
- El entregable es un repositorio de GitHub y un documento en formato PDF.
- El día 25 hará una presentación cada grupo de 5 minutos.

## 1. Objetivo de la práctica

Diseñar, entrenar y auditar un modelo de clasificación neuronal para la concesión de créditos asegurando que sus decisiones sean precisas, justas (Fair Learning) y honestas respecto a su nivel de confianza (Incertidumbre). Se desarrollarán modelos que integren restricciones físicas/matemáticas mediante capas customizadas, que penalicen sesgos demográficos mediante funciones de coste customizadas, y que cuantifiquen la varianza de su predicción.

## 2. Contexto del problema

La práctica se desarrolla en forma de taller semi-guiado, estructurado en dos sesiones prácticas de 2,5 horas cada una:

- **Construcción y Optimización (Sesión 1):** Planteamiento del problema, implementación de la arquitectura base y búsqueda de hiperparámetros con Keras Tuner.
- **Incertidumbre y Auditoría (Sesión autónoma):** Implementación de la parte de aprendizaje justo e inferencia para estimar la incertidumbre y análisis de robustez.
- **Exposición de las soluciones (Sesión 2):** Presentación y defensa técnica de los resultados, 5 minutos mas preguntas.

La tarea se entregará justo antes de la sesión 2.

## 3. Datos disponibles

Se utilizará el dataset “Home Credit Default Risk”:

<https://www.kaggle.com/competitions/home-credit-default-risk/overview>

El dataset está centrado en la predicción financiera de perfiles con poco historial crediticio. Se proporciona un notebook esqueleto para la lectura y preprocesado inicial de datos:

`Lectura_datos_Taller_Fairness.ipynb`

Estas serán las variables:

- **Variable Objetivo:** Clasificación binaria (`TARGET`, 1: Dificultades de pago, 0: Pagó a tiempo).
- **Variable Sensible:** Género del solicitante (`CODE_GENDER`). El modelo no debe discriminar basándose en esta característica.
- **Variables de Entrada:** Ingresos, anualidades, y puntuaciones de fuentes externas (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). Estas últimas contienen valores ausentes imputados que serán clave para estudiar la incertidumbre del modelo.

## 4. Tarea del estudiante

El estudiante tiene cuatro tareas principales:

1. **Arquitectura Customizada:** Implementar una capa customizada. Por ejemplo, una capa que calcule internamente el "Ratio de Endeudamiento" (combinando variables financieras de entrada) y aplique una saturación o restricción matemática sobre este ratio antes de pasarlo a las capas densas.
2. **Aprendizaje Justo (FAIR Loss):** Diseñar una función de coste customizada que combine el error de clasificación con una penalización por dependencia estadística entre la variable predicha y la variable sensible.
3. **AutoML:** Utilizar Keras Tuner para encontrar la topología óptima de la red.
4. **Incertidumbre:** Modificar el modelo para que la predicción sobre el conjunto de test devuelva tanto la clase predicha como la varianza/incertidumbre de la misma.

## 5. Entregables

Se deberá entregar un github con los modelos y los resultados, y una presentación (report en pdf) a través del aula virtual.

### Contenido obligatorio del GitHub

- El código de entrenamiento, optimización y evaluación de cada modelo. El código debe generar todas las gráficas y tablas reportadas.
- Un gráfico de dispersión (scatter plot) o Curva de Pareto que muestre el trade-off obtenido por Keras Tuner: Precisión (Eje Y) vs. Medida de Dependencia FAIR (Eje X) para los distintos valores de fairness.
- Un gráfico de distribución de la incertidumbre (varianza de las predicciones) comparando a los usuarios clasificados como "Buen pagador" vs "Mal pagador".
- Para cada entrenamiento final, incluir las curvas de loss donde se vea que el modelo ha convergido.
- Una tabla que resuma los resultados del modelo Base (sin restricciones FAIR) frente al mejor modelo FAIR en el conjunto de test. Remarcar el valor del mejor modelo en test.

### Contenido de la presentación

- Explicación de la restricción matemática introducida en la Capa Custom y la métrica de dependencia seleccionada para la Loss.
- Análisis de la gráfica de Pareto: ¿Cuánto se sacrifica en rendimiento predictivo por conseguir un modelo justo?
- Reflexión sobre la incertidumbre: ¿El modelo muestra mayor incertidumbre en perfiles donde la calidad de las fuentes externas (`EXT_SOURCE`) es peor?

## 6. Criterios de evaluación

- (30 %) Github
- (70 %) Presentación. Se presentará el pdf entregado en el aula virtual. Cada grupo hará una presentación corta (5 minutos) explicando sus resultados.
