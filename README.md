# Taller B4-T1 - Diseño de Redes Confiables

Proyecto de práctica sobre **Diseño de Redes Confiables: Justicia e Incertidumbre** aplicado a un problema de riesgo de crédito con el dataset Home Credit Default Risk.

**Componentes del equipo:**

- Albert Martin
- Josep Pérez Segura
- Rodolfo Villena

El notebook cubre los cuatro bloques de la práctica: arquitectura customizada con capa financiera, aprendizaje justo mediante FAIR Loss, optimización automática con Keras Tuner y estimación de incertidumbre predictiva.

## Objetivo

El notebook principal `Taller_B4_T1_Final.ipynb` implementa una red neuronal binaria para predecir `TARGET`:

- `TARGET = 0`: cliente sin dificultades de pago.
- `TARGET = 1`: cliente con dificultades de pago.

La variable sensible `CODE_GENDER` se conserva para auditoría y penalización de dependencia, pero **no se usa como entrada directa del modelo**.

La aportación principal del bloque 1 es la capa customizada `DebtBurdenLayer`, que calcula internamente ratios financieros de endeudamiento:

- cuota anual sobre ingresos: `AMT_ANNUITY / AMT_INCOME_TOTAL`;
- crédito total sobre ingresos: `AMT_CREDIT / AMT_INCOME_TOTAL`.

Estos ratios se calculan con variables monetarias imputadas en bruto, no con variables escaladas, y después se saturan suavemente con `tanh`.

## Bloques implementados

1. **Arquitectura customizada** mediante `DebtBurdenLayer` con dos ramas de entrada: variables escaladas y variables financieras en bruto.
2. **Aprendizaje justo** mediante funciones de pérdida FAIR (`Pearson²` y `CKA RBF`) que penalizan la dependencia estadística entre predicciones y `CODE_GENDER`.
3. **AutoML con Keras Tuner** (Hyperband) para optimizar arquitectura, dropout, learning rate y `lambda_fair` con un objetivo combinado rendimiento/justicia.
4. **Estimación de incertidumbre** mediante un modelo auxiliar que predice el error absoluto `|ŷ − y|` y MC Dropout como método complementario.

## Archivos principales

- `Taller_B4_T1_Final.ipynb`: notebook principal final con los cuatro bloques.
- `Presentacion_Taller_B4_T1.pptx`: presentación final del taller.
- `application_train.csv`: dataset de entrada. No se versiona en Git por tamaño; debe colocarse en la raíz del proyecto, en `data/application_train.csv` o en Google Drive si se ejecuta desde Colab.
- `outputs/`: artefactos generados por el notebook.

## Outputs generados

El notebook crea la carpeta `outputs/` y guarda:

- `outputs/ext_source_missingness_summary.csv`: resumen de valores ausentes en `EXT_SOURCE_1`, `EXT_SOURCE_2` y `EXT_SOURCE_3`.
- `outputs/training_history_punto1_custom_layer.csv`: histórico de entrenamiento del modelo base.
- `outputs/metrics_punto1_custom_layer.csv`: tabla final de métricas de test del modelo base.
- `outputs/fair_results_all_sweeps.csv`: resultados de los sweeps FAIR de Pearson² y CKA RBF en validation y test.
- `outputs/best_fair_proba_test.npy`: probabilidades del mejor modelo FAIR seleccionado sobre test.
- `outputs/best_fair_config.json`: configuración y métricas del mejor modelo FAIR seleccionado con validation.
- `outputs/comparison_base_vs_fair.csv`: comparación final entre modelo base y mejor modelo FAIR.
- `outputs/fair_group_gaps_base_vs_fair.csv`: gaps por grupo sensible en test.
- `outputs/test_predictions_with_uncertainty.csv`: tabla final con probabilidad predicha, clase, incertidumbre (modelo de error y MC Dropout), missingness de `EXT_SOURCE` y `CODE_GENDER` por cliente.

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

## Uso en Google Colab

Si se ejecuta el notebook en Colab, el runtime parte normalmente de `/content` y no puede ver los archivos locales del equipo. El notebook busca automáticamente el dataset en varias ubicaciones y puede montar Google Drive.

La opción recomendada es guardar el CSV con este nombre exacto:

```text
application_train.csv
```

Y dejarlo en Google Drive en:

```text
Mi unidad/data/application_train.csv
```

En Colab esa ruta se resuelve como:

```text
/content/drive/MyDrive/data/application_train.csv
```

También se puede subir temporalmente a:

```text
/content/data/application_train.csv
```

o definir la variable de entorno `HOME_CREDIT_DATA` con la ruta completa al CSV.

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
9. Preparación de una rama financiera en bruto y flags de missingness para `EXT_SOURCE`.
10. Definición de `DebtBurdenLayer`.
11. Construcción del modelo funcional Keras con dos entradas.
12. Entrenamiento con `class_weight`, `EarlyStopping` y `ReduceLROnPlateau`.
13. Curvas de loss y AUC.
14. Evaluación final sobre test.
15. Guardado de métricas y resultados auxiliares.
16. Auditoría FAIR del modelo base frente a `CODE_GENDER`.
17. Entrenamiento de modelos con FAIR Loss Pearson² para varios valores de `lambda_fair`.
18. Entrenamiento de modelos con FAIR Loss CKA RBF para varios valores de `lambda_fair`.
19. Selección del mejor compromiso usando validation.
20. Gráfico de trade-off entre rendimiento y dependencia FAIR.
21. Comparación final en test entre modelo base y mejor FAIR.
22. Estimación de incertidumbre PRE-tuner: modelo auxiliar de error y MC Dropout.
23. Búsqueda de arquitectura e hiperparámetros con Keras Tuner (Hyperband).
24. Frontera de Pareto AUC vs dependencia FAIR explorada por el tuner.
25. Reentrenamiento reproducible del mejor trial y tabla comparativa base / FAIR pre-tuner / FAIR tuneado.
26. Reevaluación de incertidumbre POST-tuner y comparación PRE vs POST.
27. Exportación de tabla final de predicciones con incertidumbre.

## Métricas calculadas

La tabla final incluye:

- `test_auc`
- `test_accuracy`
- `balanced_accuracy`
- `recall_class_1`
- `precision_class_1`
- `f1_class_1`
- `tn`, `fp`, `fn`, `tp`

Para la auditoría de justicia se reportan además:

- `pearson2`: correlación lineal `ŷ↔s` al cuadrado.
- `cka_rbf`: dependencia no lineal vía kernel gaussiano.
- probabilidad media predicha por grupo (`CODE_GENDER`).
- tasa de predicción positiva y recall por grupo.
- gaps entre grupos en test.

Dado que `TARGET` está muy desbalanceada, la `accuracy` no debe interpretarse sola. Son especialmente importantes AUC, balanced accuracy, recall de clase 1 y precision de clase 1.


## Entrega

Archivos en el repositorio principal del git:

- `Taller_B4_T1_Final.ipynb`
- `Presentacion_Taller_B4_T1.pptx`
- `README.md`
