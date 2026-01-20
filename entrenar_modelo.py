"""
torneo_lora_llama.py
Minitorneo de Tres en Raya con LLaMA-3.2-3B + LoRA REAL
Entrena usando dataset_enriquecido.csv y luego juega
"""

# ==============================================================
# IMPORTS
# ==============================================================

import os
# import uuid
import time
# import re
import csv
from ast import literal_eval
# from datetime import datetime, timezone

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ==============================================================
# CONFIG GENERAL
# ==============================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_enriquecido.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "resultados_tictactoe_lora.csv")
LORA_DIR = os.path.join(BASE_DIR, "lora_llama_tictactoe2")

GEN_CONFIG = {
    "temperature": 0.4,
    "top_p": 1.0,
    "max_new_tokens": 64
}

# ==============================================================
# CARGA MODELO + TOKENIZER
# ==============================================================

print("⏳ Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("⏳ Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

# ==============================================================
# CONFIGURACIÓN LoRA
# ==============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================================================
# DATASET ENRIQUECIDO → TEXTO DE ENTRENAMIENTO
# ==============================================================

def construir_prompt_entrenamiento(row):

    if not row["board"] or str(row["board"]).lower() == "nan":
        return None

    try:
        board_raw = literal_eval(row["board"])
    except Exception:
        return None

    board = [["b" for _ in range(3)] for _ in range(3)]
    for item in board_raw:
        if isinstance(item, list) and len(item) == 4 and item[0] == "cell":
            i, j = int(item[1]) - 1, int(item[2]) - 1
            board[i][j] = item[3]

    tablero_txt = "\n".join(
        " ".join(c if c != "b" else "-" for c in fila)
        for fila in board
    )

    estrategia = row["strategy"]

    # 🔥 REGLAS EXPLÍCITAS
    reglas = {
        "greedy": "Esta jugada gana la partida inmediatamente y SIEMPRE debe priorizarse.",
        "defensiva": "Esta jugada bloquea una victoria inmediata del rival y es obligatoria si existe.",
        "ofensiva": "Esta jugada mejora la posición o crea amenazas futuras.",
        "aleatoria": "Esta jugada solo es aceptable si no existe jugada greedy ni defensiva.",
        "invalida": "Esta jugada es ILEGAL y NUNCA debe realizarse bajo ninguna circunstancia."
    }

    return f"""
Eres un jugador experto de Tres en Raya y sigues reglas estrictas.

REGLAS ABSOLUTAS:
- Nunca hagas una jugada inválida y prioriza ganar o bloquear.
- Prioriza ganar con jugadas greedy y ofensivas.
- Si no puedes ganar, bloquea al rival con jugadas defensivas.
- Solo usa jugadas aleatorias si es la ultima opcion válida.

Estado del tablero:
{tablero_txt}

Jugador actual: {row['player'].upper()}

Movimiento elegido:
{row['move']}

Tipo de jugada: {estrategia}

Regla general:
{reglas.get(estrategia, "")}

Razón específica de esta jugada:
{row["reason"] if row["reason"] else "No aplica."}
""".strip()


def cargar_dataset_lora(csv_path, max_ejemplos=1000): # 1000 filas del dataset
    ejemplos = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_ejemplos:
                break
            texto = construir_prompt_entrenamiento(row)
            ejemplos.append({"text": texto})
    return ejemplos
 

print("📂 Cargando dataset enriquecido...")
dataset_raw = cargar_dataset_lora(DATASET_CSV, max_ejemplos=1010)
dataset = Dataset.from_list(dataset_raw)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
dataset = dataset.filter(
    lambda x: x["text"] is not None and x["text"].strip() != ""
)
dataset = dataset.map(tokenize, remove_columns=["text"])

# ==============================================================
# ENTRENAMIENTO LoRA
# ==============================================================

training_args = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=False,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("🔥 ENTRENANDO LoRA...")
trainer.train()

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

print("🔥 PARAMETROS MODELO ENTRENADO...")
model.print_trainable_parameters()

model.eval()

# ==============================================================
# UTILIDADES DEL JUEGO (IGUAL QUE TU CÓDIGO)
# ==============================================================

def tablero_vacio():
    return [["b" for _ in range(3)] for _ in range(3)]

def legal_moves(board):
    return [["mark", str(i+1), str(j+1)]
            for i in range(3) for j in range(3) if board[i][j] == "b"]

def board_to_cells(board, control):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(["cell", str(i+1), str(j+1), board[i][j]])
    cells.append(["control", control])
    return cells

def verificar_ganador(tab):
    for i in range(3):
        if tab[i][0] == tab[i][1] == tab[i][2] != "b":
            return tab[i][0]
        if tab[0][i] == tab[1][i] == tab[2][i] != "b":
            return tab[0][i]
    if tab[0][0] == tab[1][1] == tab[2][2] != "b":
        return tab[0][0]
    if tab[0][2] == tab[1][1] == tab[2][0] != "b":
        return tab[0][2]
    return ""

# ==============================================================
# LLAMADA AL MODELO
# ==============================================================

def llamar_modelo(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inicio = time.time()
    outputs = model.generate(**inputs, **GEN_CONFIG)
    fin = time.time()
    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return texto, fin - inicio

