# README – Aplicación de Evaluación de LLMs en Tres en Raya

Este repositorio contiene la implementación completa de una **aplicación experimental para evaluar el desempeño de Modelos de Lenguaje Grande (LLMs) en el juego Tres en Raya (Tic-Tac-Toe)**. El proyecto abarca desde la generación y etiquetado de jugadas, la interacción automática entre modelos, el proceso de *fine-tuning*, hasta el post-procesamiento y análisis estadístico de los resultados.

El enfoque principal es **comparar un modelo base (Sin Fine-Tuning)** frente a **un modelo con Fine-Tuning**, evaluando validez de jugadas, estrategias empleadas y resultados de las partidas. En este README se describe únicamente **el primer modelo LoRA**, ignorando cualquier implementación alternativa.

---

## 📂 Descripción del Repositorio

```
├── dataset_enriquecido.py
├── torneo_sfn.py
├── torneo_cfn.py
├── entrenar_modelo.py
├── procesar_resultados.py
├── graficar_comparaciones.py
├── requirements.txt
├── Consolidated_Move_Records.csv
├── data/
│   ├── dataset_enriquecido.csv
│   ├── reportes.csv
│   ├── reportes_ft.csv
│   ├── reportes_sft.csv
│   ├── resultados_tictactoe_sft.csv
│   └── resultados_tictactoe_ft.csv
└── plots/
    └── *.png
```

---

## 🏷️ Script de Etiquetado de Jugadas

### `dataset_enriquecido.py`

Este script se encarga de **generar un dataset enriquecido** a partir de los registros crudos de partidas (`Consolidated_Move_Records.csv`). Sus funciones principales son:

* Validar cada jugada según las reglas del Tres en Raya.
* Detectar y marcar **jugadas inválidas**.
* Clasificar cada jugada válida en una de las siguientes estrategias:

  * Ofensiva
  * Defensiva
  * Greedy
  * Aleatoria
  * Inválida
* Generar métricas agregadas por partida y por jugador (X / O).

📤 **Salida principal**:

* `data/dataset_enriquecido.csv`

Este archivo constituye la **base de todo el análisis posterior**.

---

## 🤖 Script de Interacción entre Modelos (Juego Automático)

### `torneo_sfn.py` – Torneo Sin Fine-Tuning

Implementa la lógica para que los modelos **jueguen partidas completas entre sí** utilizando el modelo base (sin fine-tuning). El script:

* Controla el turno de los jugadores (X y O).
* Solicita jugadas al modelo.
* Valida las respuestas.
* Registra cada jugada y el resultado final de la partida.

📤 **Salida**:

* Archivos CSV con resultados de partidas sin fine-tuning.

---

### `torneo_cfn.py` – Torneo Con Fine-Tuning

Funciona de manera análoga a `torneo_sfn.py`, pero utilizando el **modelo con Fine-Tuning (primer LoRA)**. Permite una comparación directa bajo las mismas condiciones experimentales.

📤 **Salida**:

* `data/resultados_tictactoe_ft.csv`

---

## 🧠 Script de Fine-Tuning

### `entrenar_modelo.py`

Este script implementa el proceso de **Fine-Tuning del modelo base**, utilizando el dataset enriquecido. Sus tareas incluyen:

* Carga y preparación del dataset.
* Configuración del entrenamiento con LoRA.
* Ajuste del modelo para mejorar:

  * Cumplimiento de reglas
  * Reducción de jugadas inválidas
  * Coherencia estratégica

⚠️ **Nota**: Solo se utiliza el **primer modelo LoRA**, ignorando implementaciones adicionales.

---

## 📊 Script de Post-Procesamiento de Resultados

### `procesar_resultados.py`

Este script consolida los resultados obtenidos de los torneos y genera métricas finales para el análisis estadístico:

* Conteo de jugadas válidas e inválidas.
* Porcentajes de victorias, derrotas y empates.
* Distribución de estrategias.
* Comparación entre modelos con y sin fine-tuning.

📤 **Salidas principales**:

* `reportes.csv`
* `reportes_sft.csv`
* `reportes_ft.csv`

---

## 📈 Visualización de Resultados

### `graficar_comparaciones.py`

Genera gráficos comparativos que permiten visualizar el impacto del fine-tuning, incluyendo:

* Resultados globales (victorias, empates, derrotas).
* Comparación de jugadas válidas vs inválidas.
* Distribución de estrategias por modelo y rol (X / O).

📤 **Salida**:

* Imágenes `.png` almacenadas en el directorio `plots/`.

---

## ⚙️ Requisitos

Instalar dependencias con:

```bash
pip install -r requirements.txt
```

---

## ▶️ Flujo de Ejecución Recomendado

1. Generar dataset enriquecido:

   ```bash
   python dataset_enriquecido.py
   ```
2. Ejecutar torneos sin fine-tuning:

   ```bash
   python torneo_sfn.py
   ```
3. Entrenar el modelo (fine-tuning):

   ```bash
   python entrenar_modelo.py
   ```
4. Ejecutar torneos con fine-tuning:

   ```bash
   python torneo_cfn.py
   ```
5. Procesar resultados:

   ```bash
   python procesar_resultados.py
   ```
6. Generar gráficos:

   ```bash
   python graficar_comparaciones.py
   ```


## 👤 Autores

**Yoel Bermeo, Anthony Vega**
Proyecto académico – Evaluación de LLMs en Tres en Raya
