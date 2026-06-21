# Taller B4-T1 — Diseño de Redes Confiables

Proyecto de práctica sobre **Diseño de Redes Confiables: Justicia e Incertidumbre** aplicado a un problema de riesgo de crédito con el dataset Home Credit Default Risk.

El notebook principal implementa los **puntos 1 y 2** de la práctica: una arquitectura neuronal customizada con capa financiera interpretable (Punto 1) y cinco estrategias de aprendizaje justo mediante penalizaciones de justicia en la función de pérdida (Punto 2).

## Objetivo

El notebook entrena una red neuronal binaria para predecir `TARGET`:

- `TARGET = 0`: cliente sin dificultades de pago.
- `TARGET = 1`: cliente con dificultades de pago.

La variable sensible `CODE_GENDER` se conserva para auditoría de justicia y se usa en el Punto 2 como variable protegida, pero **no se usa como entrada directa del modelo**.

---

## Punto 1 — Arquitectura customizada con capa financiera

La aportación principal es la capa customizada `DebtBurdenLayer`, que calcula internamente ratios financieros de endeudamiento:

- cuota anual sobre ingresos (`AMT_ANNUITY / AMT_INCOME_TOTAL`);
- crédito total sobre ingresos (`AMT_CREDIT / AMT_INCOME_TOTAL`).

Estos ratios se calculan con variables monetarias imputadas en bruto, no con variables escaladas, y después se saturan suavemente con `tanh`.

El modelo usa la API funcional de Keras con dos ramas de entrada que se concatenan antes de las capas densas:

- **Rama escalada**: variables predictoras normalizadas.
- **Rama financiera**: salida de `DebtBurdenLayer`.

---

## Punto 2 — Aprendizaje Justo: cinco penalizaciones de justicia

Se evalúan cinco funciones de pérdida que añaden un término de penalización a la BCE estándar, barriendo `λ ∈ {0.5, 1.0, 2.0, 5.0, 10.0}`:

| Sweep | Loss | Justificación |
|---|---|---|
| 1 — DP only | `BCE + λ·DP_gap` | Referencia: penaliza `\|E[ŷ\|s=1] − E[ŷ\|s=0]\|` (diferencias de media) |
| 2 — CKA lineal solo | `BCE + λ·CKA_lin` | `CKA_lin = corr(ŷ,s)²` — más fuerte que DP gap (correlación completa) |
| 3 — MI solo | `BCE + λ·MI` | MI captura cualquier dependencia estadística; subsume al DP gap |
| 4 — DP + CKA RBF | `BCE + λ·DP_gap + λ·CKA_RBF` | Combinación complementaria: DP gap controla medias O(n), CKA RBF añade forma distribucional O(n²) |
| 5 — CKA RBF solo | `BCE + λ·CKA_RBF` | HSIC con kernel gaussiano — penalización de justicia más general |

**Trick de implementación**: `CODE_GENDER` se concatena como segunda columna de `y_true` para pasarla a la función de pérdida (la API de Keras solo expone `y_true` e `y_pred`). Se usa `sample_weight` en lugar de `class_weight` para manejar el desbalance de clases con esta forma compuesta.

**Métricas de dependencia evaluadas** en test para cada modelo:

- `DP gap`: diferencia de medias entre grupos.
- `CKA lineal`: cuadrado de la correlación de Pearson entre ŷ y s.
- `CKA RBF`: HSIC con kernel gaussiano (γ = mediana heurística en evaluación, 0.5 fijo en entrenamiento).
- `MI suave`: aproximación diferenciable con soft histogram gaussiano (K = 20 bins).

---

## Archivos principales

- `Taller_B4_T1_punto1_custom_layer.ipynb`: notebook principal (Punto 1 + Punto 2).
- `data/application_train.csv`: dataset de entrada.
- `requirements.txt`: dependencias necesarias.
- `crear_environment_windows.bat`: script para crear el entorno en Windows.
- `crear_environment.sh`: script para crear el entorno en Linux, WSL o macOS.
- `prompt0.txt`, `prompt1.txt`, `prompt2.txt`: prompts usados para guiar la evolución del notebook.
- `Lectura_datos_Taller_B4_T1.ipynb`: notebook base original.
- `Taller_B4_T1.pdf`: enunciado de la práctica.

---

## Outputs generados

El notebook crea la carpeta `outputs/` y guarda:

**Punto 1**
- `ext_source_missingness_summary.csv`: resumen de valores ausentes en `EXT_SOURCE_1/2/3`.
- `training_history_punto1_custom_layer.csv`: histórico de entrenamiento (loss y AUC por época).
- `metrics_punto1_custom_layer.csv`: tabla final de métricas de test del modelo base.

**Punto 2**
- `sweep_dp_only.csv`, `sweep_cka_lin_only.csv`, `sweep_mi_only.csv`, `sweep_dp_cka_rbf.csv`, `sweep_cka_rbf_only.csv`: resultados de cada sweep (AUC, DP gap, CKA, MI por valor de λ).
- `summary_all_fair_approaches.csv`: tabla comparativa — modelo base + mejor modelo de cada uno de los cinco sweeps, con las cuatro métricas de dependencia.
- `pareto_all_approaches.png`: curva de Pareto AUC vs DP gap y AUC vs CKA RBF para los cinco sweeps.
- `pareto_front_all.csv`: puntos no dominados del frente de Pareto.

---

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

---

## Flujo del notebook

**Preprocesado común**
1. Importación de librerías y fijación de semillas.
2. Carga de columnas seleccionadas de `application_train.csv`.
3. Exclusión de `CODE_GENDER == "XNA"`.
4. Conversión de `DAYS_BIRTH` a `AGE_YEARS`.
5. Separación de `TARGET`, `CODE_GENDER` y variables predictoras.
6. Split explícito en train, validation y test.
7. Imputación de nulos ajustada solo con train.
8. Escalado ajustado solo con train.
9. Preparación de una rama financiera en bruto.

**Punto 1**

10. Definición de `DebtBurdenLayer`.
11. Construcción del modelo funcional Keras con dos entradas.
12. Entrenamiento con `class_weight`, `EarlyStopping` y `ReduceLROnPlateau`.
13. Curvas de loss y AUC.
14. Evaluación final sobre test.
15. Guardado de métricas y resultados auxiliares.

**Punto 2**

16. Preparación de `y_combined` (TARGET ‖ CODE_GENDER) y `sample_weight`.
17. Definición de las cinco funciones de pérdida y métricas de evaluación.
18. Cinco sweeps de entrenamiento (un modelo por valor de λ por sweep).
19. Tabla comparativa final: seis modelos con las cuatro métricas de dependencia.
20. Curva de Pareto con todos los puntos de los cinco sweeps.

---

## Métricas de test calculadas (Punto 1)

- `test_auc`
- `test_accuracy`
- `balanced_accuracy`
- `recall_class_1`
- `precision_class_1`
- `f1_class_1`
- `tn`, `fp`, `fn`, `tp`

Dado que `TARGET` está muy desbalanceada, la `accuracy` no debe interpretarse sola. Son especialmente importantes AUC, balanced accuracy, recall de clase 1 y precision de clase 1.

---

## Estado del proyecto

| Punto | Estado |
|---|---|
| 1 — Arquitectura customizada | Completado |
| 2 — Aprendizaje Justo (5 penalizaciones) | Completado |
| 3 — AutoML con Keras Tuner | Pendiente |
| 4 — Incertidumbre predictiva | Pendiente |

La arquitectura actual no debe interpretarse como un sistema final de concesión de crédito, sino como una base experimental para construir modelos más confiables.
