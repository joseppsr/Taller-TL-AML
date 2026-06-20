# Taller B4-T1 - Diseño de Redes Confiables

Proyecto de práctica sobre **Diseño de Redes Confiables: Justicia e Incertidumbre** aplicado a un problema de riesgo de crédito con el dataset Home Credit Default Risk.

El trabajo actual cierra el **punto 1 de la práctica**: una arquitectura neuronal customizada que incorpora una capa financiera interpretable antes de las capas densas.

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

- `Taller_B4_T1_punto1_custom_layer.ipynb`: notebook principal modificado.
- `application_train.csv`: dataset de entrada.
- `requirements.txt`: dependencias necesarias.
- `crear_environment_windows.bat`: script para crear el entorno en Windows.
- `crear_environment.sh`: script para crear el entorno en Linux, WSL o macOS.
- `prompt0.txt`, `prompt1.txt`, `prompt2.txt`: prompts usados para guiar la evolución del notebook.
- `Lectura_datos_Taller_B4_T1.ipynb`: notebook base original.
- `Taller_B4_T1.pdf`: enunciado de la práctica.

## Outputs generados

El notebook crea la carpeta `outputs/` y guarda:

- `outputs/ext_source_missingness_summary.csv`: resumen de valores ausentes en `EXT_SOURCE_1`, `EXT_SOURCE_2` y `EXT_SOURCE_3`.
- `outputs/training_history_punto1_custom_layer.csv`: histórico de entrenamiento.
- `outputs/metrics_punto1_custom_layer.csv`: tabla final de métricas de test.

## Crear el entorno en Windows

Desde PowerShell o CMD, en la carpeta del proyecto:

```bat
crear_environment_windows.bat
```

El script crea el entorno en una ruta corta para evitar problemas de Windows Long Path con TensorFlow:

```text
C:\Users\rvill\.venvs\miax-b4-t1
```

Para activar el entorno en otra consola:

```bat
"C:\Users\rvill\.venvs\miax-b4-t1\Scripts\activate.bat"
```

Después vuelve a la carpeta del proyecto:

```bat
cd /d "C:\Users\rvill\Desktop\MIAX\4 Inteligencia Artificial Avanzada\26 - 06 - 18 Taller_B4_T1. Valero Laparra"
```

Y abre Jupyter:

```bat
jupyter notebook
```

En el notebook selecciona el kernel:

```text
Python (MIAX B4-T1)
```

## Crear el entorno en Linux, WSL o macOS

```bash
bash crear_environment.sh
source .venv/bin/activate
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

El punto 1 queda preparado para continuar con las siguientes fases de la práctica:

- FAIR loss.
- Auditoría por variable sensible.
- Keras Tuner o AutoML.
- Incertidumbre predictiva.
- Curva de Pareto entre rendimiento y justicia.

La arquitectura actual no debe interpretarse como un sistema final de concesión de crédito, sino como una base experimental para construir modelos más confiables.
