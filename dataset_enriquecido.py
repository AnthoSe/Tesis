import pandas as pd
import ast
import os
from typing import List

# DATASET ENRIQUECIDO DE SOLO PARTIDAS VALIDAS

# =====================================================
# VERIFICADOR DE GANADOR
# =====================================================

def verificar_ganador(board: List[List[str]]) -> str:
    for i in range(3):
        if board[i][0] != "b" and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]

    for j in range(3):
        if board[0][j] != "b" and board[0][j] == board[1][j] == board[2][j]:
            return board[0][j]

    if board[1][1] != "b":
        if board[0][0] == board[1][1] == board[2][2]:
            return board[1][1]
        if board[2][0] == board[1][1] == board[0][2]:
            return board[1][1]

    return ""


# =====================================================
# ESTRATEGIAS
# =====================================================

def check_winning_move(board, fila, col, simbolo):
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = simbolo
    return verificar_ganador(b) == simbolo


def check_block_move(board, fila, col, simbolo):
    rival = "o" if simbolo == "x" else "x"
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = rival
    return verificar_ganador(b) == rival

def creates_fork(board, fila, col, simbolo):
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = simbolo
    return count_two_in_row(b, simbolo) >= 2

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


def reduces_opponent_threat(board, fila, col, simbolo):
    rival = "o" if simbolo == "x" else "x"

    amenazas_antes = count_two_in_row(board, rival)

    b = [row[:] for row in board]
    b[fila - 1][col - 1] = simbolo

    amenazas_despues = count_two_in_row(b, rival)

    return amenazas_despues < amenazas_antes


def etiquetar_estrategia(board, fila, col, simbolo):

    # 1. Greedy
    if check_winning_move(board, fila, col, simbolo):
        return "greedy"

    # 2. Defensiva crítica
    if check_block_move(board, fila, col, simbolo):
        return "defensiva"

    # 3. Defensiva estructural (NUEVA)
    if reduces_opponent_threat(board, fila, col, simbolo):
        return "defensiva"
      
    # 4. Ofensiva fuerte: crea fork propio
    if creates_fork(board, fila, col, simbolo):
        return "ofensiva"

    # 4. Ofensiva fuerte / posicional
    if (fila, col) == (2, 2):
        return "ofensiva"
    if (fila, col) in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        return "ofensiva"

    # 5. Último recurso
    return "aleatoria"



# =====================================================
# CONVERSIÓN BOARD
# =====================================================

def convertir_board(board_list):
    matrix = [["b" for _ in range(3)] for _ in range(3)]
    for item in board_list:
        if item[0] == "cell":
            fila = int(item[1])
            col = int(item[2])
            matrix[fila - 1][col - 1] = item[3]
    return matrix

def reconstruir_board_previo(board, fila, col):
    if not (1 <= fila <= 3 and 1 <= col <= 3):
        return None  # movimiento corrupto
    b = [row[:] for row in board]
    b[fila - 1][col - 1] = "b"
    return b



# =====================================================
# VALIDACIONES REALES
# =====================================================

def contar_fichas(board):
    x = sum(row.count("x") for row in board)
    o = sum(row.count("o") for row in board)
    return x, o


def tablero_consistente(board):
    x, o = contar_fichas(board)
    return abs(x - o) <= 1


def es_jugada_legal(board, fila, col, jugador):
    if not (1 <= fila <= 3 and 1 <= col <= 3):
        return False

    if board[fila - 1][col - 1] != "b":
        return False

    if not tablero_consistente(board):
        return False

    if verificar_ganador(board) != "":
        return False

    x_count, o_count = contar_fichas(board)

    if jugador == "x" and x_count != o_count:
        return False
    if jugador == "o" and x_count != o_count + 1:
        return False

    return True


# =====================================================
# CARGA DATASET
# =====================================================

df = pd.read_csv("Consolidated_Move_Records.csv", nrows=1535)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_DATASET = os.path.join(DATA_DIR, "dataset_enriquecido.csv")
OUTPUT_REPORT = os.path.join(DATA_DIR, "reportes.csv")
OUTPUT_MODELS = os.path.join(DATA_DIR, "reportes_por_modelo.csv")

columns = ["id_match", "player", "move", "win", "board", "reason"]
df = df[columns]

rows_validas = []

# =====================================================
# PROCESAMIENTO PRINCIPAL
# =====================================================

rows_enriquecidas = []
jugadas_invalidas = 0

for _, row in df.iterrows():

    row_new = row.copy()
    row_new["strategy"] = "invalida"   # 👈 valor por defecto SIEMPRE

    # =========================
    # BOARD
    # =========================
    try:
        board_list = ast.literal_eval(row["board"])
        board = convertir_board(board_list)
    except:
        jugadas_invalidas += 1
        rows_enriquecidas.append(row_new)
        continue

    # =========================
    # MOVE
    # =========================
    try:
        move = ast.literal_eval(row["move"])
        if not (
            isinstance(move, list)
            and len(move) == 3
            and move[0] == "mark"
            and move[1].isdigit()
            and move[2].isdigit()
        ):
            raise ValueError
        fila = int(move[1])
        col = int(move[2])
    except:
        jugadas_invalidas += 1
        rows_enriquecidas.append(row_new)
        continue

    jugador = row["player"]

    # =========================
    # BOARD PREVIO
    # =========================
    board_before = reconstruir_board_previo(board, fila, col)
    if board_before is None:
        jugadas_invalidas += 1
        rows_enriquecidas.append(row_new)
        continue

    # =========================
    # LEGALIDAD REAL
    # =========================
    legal = es_jugada_legal(board_before, fila, col, jugador)

    if not legal:
        estrategia = "invalida"
        jugadas_invalidas += 1
    else:
        estrategia = etiquetar_estrategia(board_before, fila, col, jugador)

    # =========================
    # ESTRATEGIA (SOLO SI ES LEGAL)
    # =========================
    row_new["strategy"] = etiquetar_estrategia(
        board_before, fila, col, jugador
    )

    rows_enriquecidas.append(row_new)


# =====================================================
# DATASET FINAL
# =====================================================

df_enriched = pd.DataFrame(rows_enriquecidas)
df_enriched.to_csv(OUTPUT_DATASET, index=False)

# =====================================================
# CONTEO DE ESTRATEGIAS
# =====================================================

conteo = df_enriched["strategy"].value_counts()

total_jugadas = len(df_enriched)
invalidas = conteo.get("invalida", 0)
validas = total_jugadas - invalidas

greedy = conteo.get("greedy", 0)
ofensiva = conteo.get("ofensiva", 0)
defensiva = conteo.get("defensiva", 0)
aleatoria = conteo.get("aleatoria", 0)

print("\n📊 ===== RESUMEN DE ESTRATEGIAS =====")
print(f"Total jugadas      : {total_jugadas}")
print(f"Jugadas válidas    : {validas}")
print(f"Jugadas inválidas  : {invalidas}")
print("----------------------------------")
print(f"Greedy             : {greedy}")
print(f"Ofensiva           : {ofensiva}")
print(f"Defensiva          : {defensiva}")
print(f"Aleatoria          : {aleatoria}")
print("==================================\n")

print(f"📊 Dataset enriquecido generado: {OUTPUT_DATASET}")

# =====================================================
# REPORTE GLOBAL
# =====================================================

report = {}

# Totales generales
report["total_moves"] = len(df_enriched)
report["total_matches"] = df_enriched["id_match"].nunique()

# Válidas / inválidas
report["valid_moves"] = (df_enriched["strategy"] != "invalida").sum()
report["invalid_moves"] = (df_enriched["strategy"] == "invalida").sum()

# Conteo global de estrategias
report["greedy"] = (df_enriched["strategy"] == "greedy").sum()
report["defensiva"] = (df_enriched["strategy"] == "defensiva").sum()
report["ofensiva"] = (df_enriched["strategy"] == "ofensiva").sum()
report["aleatoria"] = (df_enriched["strategy"] == "aleatoria").sum()

# =====================================================
# MÉTRICAS POR JUGADOR
# =====================================================

for p in ["x", "o"]:
    df_p = df_enriched[df_enriched["player"] == p]

    report[f"{p}_moves"] = len(df_p)
    report[f"{p}_wins"] = df_p["win"].sum()
    report[f"{p}_losses"] = df_enriched[
        (df_enriched["player"] != p) & (df_enriched["win"] == 1)
    ].shape[0]
    report[f"{p}_draws"] = (
        report[f"{p}_moves"]
        - report[f"{p}_wins"]
        - report[f"{p}_losses"]
    )

    for strat in ["greedy", "defensiva", "ofensiva", "aleatoria"]:
        report[f"{p}_{strat}"] = (df_p["strategy"] == strat).sum()


pd.DataFrame([report]).to_csv(OUTPUT_REPORT, index=False)
print(f"📊 Reporte global generado: {OUTPUT_REPORT}")


# =====================================================
# REPORTE POR MODELO
# =====================================================

df_model = df_enriched.merge(
    pd.read_csv("Consolidated_Move_Records.csv")[["id_match", "model"]],
    on="id_match",
    how="left"
)

model_reports = []

for model, df_m in df_model.groupby("model"):

    r = {"model": model, "total_moves": len(df_m)}

    for p in ["x", "o"]:
        df_p = df_m[df_m["player"] == p]
        r[f"{p}_moves"] = len(df_p)
        r[f"{p}_wins"] = df_p["win"].sum()

        for strat in ["greedy", "defensiva", "ofensiva", "aleatoria"]:
            count = (df_p["strategy"] == strat).sum()
            r[f"{p}_{strat}"] = count
            r[f"{p}_{strat}_pct"] = round((count / len(df_p)) * 100, 2) if len(df_p) else 0

    model_reports.append(r)

pd.DataFrame(model_reports).to_csv(OUTPUT_MODELS, index=False)
print(f"📈 Reporte por modelo generado: {OUTPUT_MODELS}")

