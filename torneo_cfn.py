"""
torneo_sn.py "sin fine tunning"

Partidas de Tres en Raya de Llama entre varios modelos de lenguaje.

- Recolecta jugadas de un torneo round-robin (cada modelo vs todos).
- Registra:
    * id_match, board, legalMoves, move, valid, win
    * player (x/o), model, reason, execution_time, timestamp
    * strategy (greedy, defensiva, ofensiva, aleatoria, invalida)
    * coherent (1 si la explicación coincide con la jugada aplicada)
    * hiperparámetros usados en la llamada (temperature, top_p, etc.)

El JSONL generado es la entrada para:
    - procesar_resultados.py
    - graficas_resultados.py
"""

import os
import uuid
import time
import re
import csv
import torch
from ast import literal_eval
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================
# CONFIGURACIÓN GENERAL
# =========================

# Clave de OpenRouter (poner la tuya o usar variable de entorno)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODELOS = {
    # =====================
    # Nivel medio
    # =====================
    "LLaMA-3.1-8b": "meta-llama/llama-3.1-8B-instruct",

    "Mistral: Pixtral 12B":"mistralai/pixtral-12b",
    "Mistral: Ministral 8B":"mistralai/ministral-8b",
    "GPT-3.5 Turbo 16k":"openai/gpt-3.5-turbo-16k",
    "DeepSeek: DeepSeek V3.1":"deepseek/deepseek-chat-v3.1",
    "Google: Gemini 2.0 Flash Lite":"google/gemini-2.0-flash-lite-001",

    "Mistral-7B": "mistralai/mistral-7b-instruct-v0.3",
    "Qwen-2.5-7B": "qwen/qwen-2.5-7b-instruct",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    
    # Medio-alto
    "Mixtral-8x7B": "mistralai/mixtral-8x7b-instruct",
    # =====================
    # Nivel medio–alto / alto
    # =====================
    "DeepSeek: DeepSeek V3.2": "deepseek/deepseek-v3.2",
    "OpenAI: ChatGPT-4o":"openai/chatgpt-4o-latest",
    "Meta: Llama 3.3 70B Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Google: Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Amazon: Nova Pro 1.0":"amazon/nova-pro-v1",
}

# Hiperparámetros fijos de generación (metodología reproducible)
GEN_CONFIG = {
    "temperature": 0.4,   # Baja aleatoriedad → decisiones más deterministas
    "top_p": 1.0,         # Sin recorte probabilístico adicional
    "max_tokens": 64,     # Límite suficiente para jugada + razón
    "presence_penalty": 0.0,   # No forzar diversidad artificial
    "frequency_penalty": 0.0,  # No penalizar repeticiones
}

BASE_MODEL_NAME = "LLaMA-3.2-3B"
LORA_MODEL_ID = "llama-3.2-3b-lora-tictactoe"
BASE_MODEL_ID = LORA_MODEL_ID

# =========================
# MODELO LLaMA LoRA (LOCAL)
# =========================

LORA_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR = "./lora_llama_tictactoe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_lora = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
tokenizer_lora.pad_token = tokenizer_lora.eos_token

base_model_lora = AutoModelForCausalLM.from_pretrained(
    LORA_BASE_MODEL,
    dtype=torch.float16,
    device_map=None,
).to("cuda")

lora_model = PeftModel.from_pretrained(
    base_model_lora,
    LORA_DIR
).to("cuda")
lora_model.eval()


# Rutas de salida
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_DIR, "resultados_tictactoe_ft.csv")
# Para después del fine-tuning puedes cambiar el nombre a resultados_tictactoe_despues_ft.jsonl

# =========================
# INCLUIR COLUMNAS EXTRA
# =========================
def incluir_columnas_extra(include_extra_columns: bool) -> List[str]:
    base_columns = [
        'id_match','legalMoves','board','move','valid','win','is_terminal',
        'player','model','reason','execution_time','timestamp','strategy'
    ]
    if include_extra_columns:
        extra_columns = ['coherent','turn','temperature','top_p','max_tokens']
        return base_columns + extra_columns
    return base_columns

# =========================
# UTILIDADES DEL JUEGO
# =========================

def tablero_vacio() -> List[List[str]]:
    return [["b" for _ in range(3)] for _ in range(3)]


def legal_moves(board: List[List[str]]) -> List[List[str]]:
    """
    Devuelve solo las casillas vacías en formato ['mark','fila','col'].
    """
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "b":
                moves.append(["mark", str(i + 1), str(j + 1)])
    return moves


def board_to_cells(board: List[List[str]], control: str) -> List[List[str]]:
    """
    Convierte el tablero a una lista de celdas + ['control', jugador],
    siguiendo el formato del dataset de la inge.
    """
    cells: List[List[str]] = []
    for i in range(3):
        for j in range(3):
            cells.append(["cell", str(i + 1), str(j + 1), board[i][j]])
    cells.append(["control", control])
    return cells


def verificar_ganador(tab: List[List[str]]) -> str:
    """
    Devuelve 'x', 'o' o '' según haya ganador o no.
    """
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


# =========================
# CLASIFICACIÓN ESTRATÉGICA
# =========================

def check_winning_move(board: List[List[str]], fila: int, col: int, simbolo: str) -> bool:
    """
    Verifica si colocar 'simbolo' en (fila,col) da victoria inmediata.
    """
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = simbolo
    return verificar_ganador(b) == simbolo


def check_block_move(board: List[List[str]], fila: int, col: int, simbolo: str) -> bool:
    """
    Verifica si colocar en (fila,col) bloquea una victoria inmediata
    del oponente.
    """
    rival = "o" if simbolo == "x" else "x"
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = rival
    return verificar_ganador(b) == rival

def count_two_in_row(board, simbolo):
    count = 0
    lines = []

    # filas
    for i in range(3):
        lines.append([board[i][0], board[i][1], board[i][2]])

    # columnas
    for j in range(3):
        lines.append([board[0][j], board[1][j], board[2][j]])

    # diagonales
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if line.count(simbolo) == 2 and line.count("b") == 1:
            count += 1

    return count

def creates_fork(board, fila, col, simbolo):
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = simbolo
    return count_two_in_row(b, simbolo) >= 2

def blocks_fork(board, fila, col, simbolo):
    rival = "o" if simbolo == "x" else "x"
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = rival
    return count_two_in_row(b, rival) >= 2

def etiquetar_estrategia(board_before, fila, col, simbolo):

    rival = "o" if simbolo == "x" else "x"

    # 1. Greedy: gana inmediatamente
    if check_winning_move(board_before, fila, col, simbolo):
        return "greedy"

    # 2. Defensiva fuerte: bloquea victoria inmediata
    if check_block_move(board_before, fila, col, simbolo):
        return "defensiva"

    # 3. Defensiva avanzada: bloquea fork del rival
    if blocks_fork(board_before, fila, col, simbolo):
        return "defensiva"

    # 4. Ofensiva fuerte: crea fork propio
    if creates_fork(board_before, fila, col, simbolo):
        return "ofensiva"

    # 5. Ofensiva media: crea amenaza (2 en línea)
    b = [row[:] for row in board_before]
    b[fila - 1][col - 1] = simbolo
    if count_two_in_row(b, simbolo) == 1:
        return "ofensiva"

    # 6. Control estratégico (centro o esquinas)
    if (fila, col) == (2, 2):
        return "ofensiva"
    if (fila, col) in [(1,1),(1,3),(3,1),(3,3)]:
        return "ofensiva"

    # 7. Último recurso
    return "aleatoria"


# =========================
# CLASIFICACIÓN JUGADA INVÁLIDA
# =========================
def clasificar_invalidez(texto, fila, col, legal):
    # No se pudo extraer nada
    if fila == -1 or col == -1:
        return "invalida"

    # Fuera del tablero
    if fila not in [1,2,3] or col not in [1,2,3]:
        return "invalida"

    # Casilla ocupada
    if ["mark", str(fila), str(col)] not in legal:
        return "invalida"

    return "invalida"

# =========================
# COHERENCIA TEXTO–JUGADA
# =========================

def extraer_jugada_de_reason(reason: str) -> Tuple[int, int]:
    """
    Intenta extraer (fila, col) desde 'reason'.

    Soporta:
      - ['mark',2,3]
      - ['mark','2','3']
      - (2,3)
      - "2,3" o "2;3"

    Si no puede, devuelve (-1, -1).
    """
    if not isinstance(reason, str):
        return -1, -1

    # Intentar evaluar un fragmento entre corchetes
    if "[" in reason and "]" in reason:
        try:
            frag = reason[reason.index("["):reason.index("]") + 1]
            mov = literal_eval(frag)
            if isinstance(mov, list) and len(mov) >= 3:
                return int(mov[1]), int(mov[2])
        except Exception:
            pass

    # Buscar "2,3" o "2;3"
    m = re.search(r"(\d)\s*[,;]\s*(\d)", reason)
    if m:
        return int(m.group(1)), int(m.group(2))

    return -1, -1


# =========================
# PROMPT Y LLAMADA AL MODELO
# =========================

def construir_prompt(board: List[List[str]], jugador_actual: str, historial_texto: str) -> str:
    texto_tablero = "\n".join(
        " ".join(c if c != "b" else "-" for c in fila) for fila in board
    )
    otro = "O" if jugador_actual == "x" else "X"

    prompt = f"""
Estamos jugando Tres en Raya (Tic-Tac-Toe).

OBJETIVO:
Tu objetivo principal es GANAR la partida.
Prioriza:
1. Jugadas que ganen inmediatamente.
2. Jugadas que creen una amenaza futura (fork).
3. Bloquear al rival solo si no puedes ganar.
4. Defiendete en medida de lo posible.
5. Esta prohibido hacer jugadas invalidas

Siempre que exista una jugada ganadora, debe ser elegida.
Empatar es una estrategia subóptima.
Empatar es el último recurso.

Estado actual del tablero:
{texto_tablero}

Tu símbolo es '{jugador_actual.upper()}'. El oponente usa '{otro}'.

Jugadas previas:
{historial_texto or "Aún no hay jugadas previas."}

Debes responder SIEMPRE con el siguiente formato:
['mark', fila, columna] seguido de una explicación breve de tu estrategia.

Ejemplo:
['mark', 2, 2] Decido marcar el centro porque controla más líneas.

Las filas y columnas van de 1 a 3. Elige SIEMPRE una casilla libre.
"""
    return prompt


def llamar_modelo(model_id: str, prompt: str):
    """
    Llama a un modelo remoto (OpenRouter) o local (LoRA).
    """
    inicio = time.time()

    # ---------- MODELO LoRA LOCAL ----------
    if model_id == LORA_MODEL_ID:
        try:
            inputs = tokenizer_lora(
              prompt,
              return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    temperature=GEN_CONFIG["temperature"],
                    top_p=GEN_CONFIG["top_p"],
                    max_new_tokens=GEN_CONFIG["max_tokens"],
                    pad_token_id=tokenizer_lora.eos_token_id
                )

            decoded = tokenizer_lora.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # eliminar el prompt del inicio
            if decoded.startswith(prompt):
                texto = decoded[len(prompt):].strip()
            else:
                texto = decoded.strip()

            fin = time.time()
            return texto.strip(), fin - inicio

        except Exception as e:
            fin = time.time()
            return f"ERROR LORA: {e}", fin - inicio

    # ---------- MODELOS REMOTOS ----------
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=GEN_CONFIG["temperature"],
            top_p=GEN_CONFIG["top_p"],
            max_tokens=GEN_CONFIG["max_tokens"],
            presence_penalty=GEN_CONFIG["presence_penalty"],
            frequency_penalty=GEN_CONFIG["frequency_penalty"],
            timeout=30,
        )

        fin = time.time()
        texto = resp.choices[0].message.content.strip()
        return texto, fin - inicio

    except Exception as e:
        fin = time.time()
        return f"ERROR API: {e}", fin - inicio



# =========================
# LÓGICA DE UNA PARTIDA
# =========================

def jugar_partida(modelo_x: str, modelo_o: str, include_extra_columns: bool) -> List[Dict[str, Any]]:
    """
    Ejecuta una partida X vs O y devuelve una lista de registros
    (uno por jugada) listos para escribir en JSONL.
    """
    partida_id = str(uuid.uuid4())
    tablero = tablero_vacio()
    jugador_actual = "x"
    modelos_partida = {"x": modelo_x, "o": modelo_o}

    registros: List[Dict[str, Any]] = []
    historial_descriptivo: List[str] = []
    intentos_invalidos = {"x": 0, "o": 0}
    ganador = ""

    for turno in range(1, 10):  # máximo 9 jugadas
        modelo = modelos_partida[jugador_actual]
        legal = legal_moves(tablero)
        control_siguiente = "o" if jugador_actual == "x" else "x"
        historial_texto = "\n".join(historial_descriptivo)

        prompt = construir_prompt(tablero, jugador_actual, historial_texto)

        texto, exec_time = llamar_modelo(modelo, prompt)

        # Tablero antes de aplicar jugada (para estrategia)
        tablero_antes = [row[:] for row in tablero]

        # Extraer jugada propuesta
        fila = col = -1
        try:
            if "[" in texto and "]" in texto:
                frag = texto[texto.index("["):texto.index("]") + 1]
                mov = literal_eval(frag)
                fila = int(mov[1])
                col = int(mov[2])
        except Exception:
            fila = col = -1

        move_list = ["mark", str(fila), str(col)]
        valido = int(move_list in legal)

        if valido:
            intentos_invalidos[jugador_actual] = 0
            tablero[fila - 1][col - 1] = jugador_actual
        else:
            intentos_invalidos[jugador_actual] += 1

        if not valido:
            historial_descriptivo.append(
                "Tu jugada fue inválida. "
                "Debes responder ÚNICAMENTE con el formato exacto: "
                "['mark', fila, columna] usando una jugada legal."
            )

        # Estrategia basada en tablero antes de la jugada
        if valido:
            strategy = etiquetar_estrategia(tablero_antes, fila, col, jugador_actual)
        else:
            strategy = clasificar_invalidez(texto, fila, col, legal)

        # Coherencia texto–jugada
        fila_decl, col_decl = extraer_jugada_de_reason(texto)
        coherent = int(
            valido
            and fila_decl == fila
            and col_decl == col
        )

        # Registro de la jugada
        registro = {
            "id_match": partida_id,
            "legalMoves": legal,
            "board": board_to_cells(tablero, control_siguiente),
            "move": move_list,
            "valid": valido,
            "win": 0,  # se rellena al final
            "is_terminal": 0, 
            "player": jugador_actual,
            "model": modelo,
            "reason": texto,
            "execution_time": exec_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "coherent": coherent,
            "turn": turno,
            "temperature": GEN_CONFIG["temperature"],
            "top_p": GEN_CONFIG["top_p"],
            "max_tokens": GEN_CONFIG["max_tokens"],
        }
        
        if not include_extra_columns:
            del registro["coherent"]
            del registro["turn"]
            del registro["temperature"]
            del registro["top_p"]
            del registro["max_tokens"]
        
        registros.append(registro)

        # Actualizar historial descriptivo
        if valido:
            desc = f"Turno {turno}: {jugador_actual.upper()} jugó {move_list}."
        else:
            desc = f"Turno {turno}: {jugador_actual.upper()} intentó jugada inválida {move_list}."
        historial_descriptivo.append(desc)

        # Derrota automática por 10 jugadas inválidas
        if intentos_invalidos[jugador_actual] >= 10:
            ganador = "o" if jugador_actual == "x" else "x"
            break

        # Ganador normal
        ganador_actual = verificar_ganador(tablero)
        if ganador_actual:
            ganador = ganador_actual
            break

        # Empate si el tablero está lleno
        if all(c != "b" for fila_tab in tablero for c in fila_tab):
            break

        # Cambiar de jugador
        jugador_actual = "o" if jugador_actual == "x" else "x"

    # Marcar win=1 para el ganador
    # Marcar win=1 solo en la ÚLTIMA jugada válida del ganador
    if ganador:
        for r in reversed(registros):
            if r["player"] == ganador and r["valid"] == 1:
                r["win"] = 1
                break

    # Marcar como terminal SOLO la última jugada válida
    for r in reversed(registros):
        if r["valid"] == 1:
            r["is_terminal"] = 1
            break

    return registros

# =========================
# TORNEO
# =========================
n_partidas = 10
def ejecutar_torneo(modelos: Dict[str, str], archivo_salida: str, include_extra_columns: bool) -> None:
    """
    Ejecuta un minitorneo round-robin y escribe los resultados en un archivo CSV.
    """
    nombres = list(modelos.keys())
    cabeceras = incluir_columnas_extra(include_extra_columns)
    
    # Verifica las cabeceras
    print("Cabeceras del CSV:", cabeceras)
    
    with open(archivo_salida, "w", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=cabeceras)
        writer.writeheader()

        for nombre_oponente, modelo_oponente in modelos.items():

            print(f"\n▶ Llama-3.2-3B(LoRA) vs {nombre_oponente}")

            # ---- 10 partidas: BASE como X ----
            for i in range(1, n_partidas + 1):
                print(f"  Partida {i}/{n_partidas} → Llama-3.2-3B(LoRA) (X) vs {nombre_oponente} (O)")
                registros = jugar_partida(BASE_MODEL_ID, modelo_oponente, include_extra_columns)

                for r in registros:
                    fila = {k: r[k] for k in cabeceras if k in r}
                    writer.writerow(fila)

            # ---- 10 partidas: BASE como O ----
            for i in range(1, n_partidas + 1):
                print(f"  Partida {i}/{n_partidas} → {nombre_oponente} (X) vs Llama-3.2-3B(LoRA) (O)")
                registros = jugar_partida(modelo_oponente, BASE_MODEL_ID, include_extra_columns)

                for r in registros:
                    fila = {k: r[k] for k in cabeceras if k in r}
                    writer.writerow(fila)

# =========================
# PROGRAMA PRINCIPAL
# =========================

def main():
    include_extra_columns = False  # Cambia a False si no quieres incluir las columnas adicionales
    print("Iniciando minitorneo de Tres en Raya...")
    print(f"Modelos participantes: {list(MODELOS.keys())}")
    print(f"Hiperparámetros: {GEN_CONFIG}")
    ejecutar_torneo(MODELOS, OUTPUT_CSV, include_extra_columns)
    print(f"Torneo completado. Resultados guardados en {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

