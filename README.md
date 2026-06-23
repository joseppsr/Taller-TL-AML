# Taller B4-T1 - Diseño de Redes Confiables

Proyecto de práctica sobre **Diseño de Redes Confiables: Justicia e Incertidumbre** aplicado a un problema de riesgo de crédito con el dataset Home Credit Default Risk.

El trabajo actual cierra el **punto 1** y añade el **punto 2 de aprendizaje justo**: una arquitectura neuronal customizada que incorpora una capa financiera interpretable antes de las capas densas y una FAIR Loss que penaliza la dependencia estadística entre predicción y género.

## Objetivo actual

El notebook principal implementa una red neuronal binaria para predecir `TARGET`:

- `TARGET = 0`: cliente sin dificultades de pago.
- `TARGET = 1`: cliente con dificultades de pago.

La variable sensible `CODE_GENDER` se conserva para auditoría y futuras fases de fairness, pero **no se usa como entrada directa del modelo**.

La aportación principal es la capa customizada `DebtBurdenLayer`, que calcula internamente ratios financieros de endeudamiento:

- cuota anual sobre ingresos;
- crédito total sobre ingresos.

Estos ratios se calculan con variables monetarias imputadas en bruto, no con variables escaladas, y después se saturan suavemente con `tanh`.

## Archivos principales

- `Taller_B4_T1.ipynb`: notebook principal final con punto 1 y punto 2.
- `application_train.csv`: dataset de entrada. No se versiona en Git por tamaño; debe colocarse en la raíz del proyecto o en `data/application_train.csv`.
- `requirements.txt`: dependencias necesarias.
- `outputs/`: artefactos generados por el notebook.

## Outputs generados

El notebook crea la carpeta `outputs/` y guarda:

- `outputs/ext_source_missingness_summary.csv`: resumen de valores ausentes en `EXT_SOURCE_1`, `EXT_SOURCE_2` y `EXT_SOURCE_3`.
- `outputs/training_history_punto1_custom_layer.csv`: histórico de entrenamiento.
- `outputs/metrics_punto1_custom_layer.csv`: tabla final de métricas de test.
- `outputs/fair_results_all_sweeps.csv`: resultados de los sweeps FAIR de Pearson² y CKA RBF en validation y test.
- `outputs/best_fair_proba_test.npy`: probabilidades del mejor modelo FAIR seleccionado sobre test.
- `outputs/best_fair_config.json`: configuración y métricas del mejor modelo FAIR seleccionado con validation.
- `outputs/comparison_base_vs_fair.csv`: comparación final entre modelo base y mejor modelo FAIR.
- `outputs/fair_group_gaps_base_vs_fair.csv`: gaps por grupo sensible en test.

## Crear el entorno

Desde una terminal, en la carpeta del proyecto:

```bash
python -m venv .venv
```

Activa el entorno.

En Windows:

```bat
.venv\Scripts\activate
```

En Linux, WSL o macOS:

```bash
source .venv/bin/activate
```

Instala dependencias:

```bat
pip install -r requirements.txt
```

Para abrir Jupyter:

```bat
jupyter notebook
```

## Flujo del notebook

El notebook ejecuta los siguientes pasos:

1. Importación de librerías y fijación de semillas.
2. Carga de columnas seleccionadas de `application_train.csv`.
3. Exclusión de `CODE_GENDER == "XNA"`.
4. Conversión de `DAYS_BIRTH` a `AGE_YEARS`.
5. Separación de `TARGET`, `CODE_GENDER` y variables predictoras.
6. Split explícito en train, validation y test.
7. Imputación de nulos ajustada solo con train.
8. Escalado ajustado solo con train.
9. Preparación de una rama financiera en bruto.
10. Definición de `DebtBurdenLayer`.
11. Construcción del modelo funcional Keras con dos entradas.
12. Entrenamiento con `class_weight`, `EarlyStopping` y `ReduceLROnPlateau`.
13. Curvas de loss y AUC.
14. Evaluación final sobre test.
15. Guardado de métricas y resultados auxiliares.
16. Auditoría FAIR del modelo base frente a `CODE_GENDER`.
17. Entrenamiento de modelos con FAIR Loss para varios valores de `lambda_fair`.
18. Selección del mejor compromiso usando validation.
19. Gráfico de trade-off entre rendimiento y dependencia FAIR.
20. Comparación final en test entre modelo base y mejor FAIR.

## Métricas calculadas

La tabla final incluye:

- `test_auc`
- `test_accuracy`
- `balanced_accuracy`
- `recall_class_1`
- `precision_class_1`
- `f1_class_1`
- `tn`
- `fp`
- `fn`
- `tp`

Dado que `TARGET` está muy desbalanceada, la `accuracy` no debe interpretarse sola. Son especialmente importantes AUC, balanced accuracy, recall de clase 1 y precision de clase 1.

## Estado actual

Los puntos 1 y 2 quedan preparados para continuar con las siguientes fases de la práctica:

- Keras Tuner o AutoML.
- Incertidumbre predictiva.
- Curva de Pareto entre rendimiento y justicia.

El punto 2 deja implementada una primera versión de aprendizaje justo mediante FAIR Loss. La selección de `lambda_fair` se realiza con validation y test queda reservado para la comparación final.

La arquitectura actual no debe interpretarse como un sistema final de concesión de crédito, sino como una base experimental para construir modelos más confiables.
