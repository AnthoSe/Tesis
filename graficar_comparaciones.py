import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import wilcoxon, permutation_test
from statsmodels.stats.descriptivestats import sign_test


# =======================================================
# RUTAS
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================================================
# CARGA DE DATOS
# =======================================================
df_sft = pd.read_csv(os.path.join(DATA_DIR, "reportes_sft.csv"))
df_ft  = pd.read_csv(os.path.join(DATA_DIR, "reportes_ft.csv"))

# =======================================================
# FILTRAR SOLO OPONENTES (opcional)
# =======================================================
MODELOS_REFERENCIA = [
    "meta-llama/llama-3.1-8B-instruct",
    "mistralai/pixtral-12b",
    "mistralai/ministral-8b",
    "openai/gpt-3.5-turbo-16k",
    "deepseek/deepseek-chat-v3.1",
    "google/gemini-2.0-flash-lite-001",
    "mistralai/mistral-7b-instruct-v0.3",
    "qwen/qwen-2.5-7b-instruct",
    "google/gemma-2-9b-it",
    "mistralai/mixtral-8x7b-instruct",
    "deepseek/deepseek-v3.2",
    "openai/chatgpt-4o-latest",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
    "amazon/nova-pro-v1",
]

df_sft = df_sft[df_sft["opponent_model"].isin(MODELOS_REFERENCIA)]
df_ft  = df_ft[df_ft["opponent_model"].isin(MODELOS_REFERENCIA)]

# =======================================================
# ORDENAR MODELOS SEGÚN LISTA DE REFERENCIA
# =======================================================
# orden_modelos = pd.CategoricalDtype(
#     categories=MODELOS_REFERENCIA,
#     ordered=True
# )

# df_sft["opponent_model"] = df_sft["opponent_model"].astype(orden_modelos)
# df_ft["opponent_model"]  = df_ft["opponent_model"].astype(orden_modelos)

# df_sft = df_sft.sort_values("opponent_model").reset_index(drop=True)
# df_ft  = df_ft.sort_values("opponent_model").reset_index(drop=True)

# =======================================================
# FUNCIÓN BASE (MISMO ESTILO LEGACY)
# =======================================================
def grafico_barra_apilado(df, cols, labels, titulo, nombre, colores):
    modelos = df["opponent_model"].tolist()
    y = np.arange(len(modelos))
    acumulado = np.zeros(len(modelos))

    plt.figure(figsize=(10, 6))

    for col, lab, color in zip(cols, labels, colores):
        valores = df[col].fillna(0)
        plt.barh(y, valores, left=acumulado, label=lab, color=color)

        for i, v in enumerate(valores):
            if v > 0:
                plt.text(
                    acumulado[i] + v / 2,
                    i,
                    f"{v:.1f}%",
                    va="center",
                    ha="center",
                    fontsize=9
                )

        acumulado += valores

    plt.yticks(y, modelos)
    plt.xlabel("Porcentaje")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.xlim(0, 100)
    
    plt.gca().invert_yaxis()  # 👈 INVERTIR
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, nombre))
    plt.close()
    print(f"✔ Gráfico guardado: {nombre}")

# =======================================================
# RESULTADOS (WIN / DRAW / LOSS)
# =======================================================
def graficos_resultados(df, sufijo, rol=None):
    pref = "" if rol is None else f"{rol}_"
    titulo_rol = "Global" if rol is None else f"como {rol.upper()}"

    grafico_barra_apilado(
        df,
        [f"{pref}win_pct", f"{pref}draw_pct", f"{pref}loss_pct"],
        ["Win", "Draw", "Loss"],
        f"Resultados {titulo_rol} LLaMA-3.2-3B({sufijo})",
        f"resultados_{titulo_rol}_{sufijo}.png".replace(" ", "_"),
        ["#54a0ff", "#feca57", "#ff6b6b"]
    )

# =======================================================
# ESTRATEGIAS
# =======================================================
def graficos_estrategias(df, sufijo, rol=None):
    pref = "" if rol is None else f"{rol}_"
    titulo_rol = "Global" if rol is None else f"como {rol.upper()}"

    grafico_barra_apilado(
        df,
        [
            f"{pref}ofensiva_pct",
            f"{pref}defensiva_pct",
            f"{pref}aleatoria_pct",
            f"{pref}greedy_pct",
            f"{pref}invalida_pct",
        ],
        ["Ofensiva", "Defensiva", "Aleatoria", "Greedy", "Inválida"],
        f"Estrategias {titulo_rol} LLaMA-3.2-3B({sufijo})",
        f"estrategias_{titulo_rol}_{sufijo}.png".replace(" ", "_"),
        ["#ff6b6b", "#feca57", "#1dd1a1", "#7C59E3", "#576574"]
    )

# =======================================================
# VALIDAS VS INVALIDAS
# =======================================================
def graficos_validas_invalidas(df, sufijo, rol=None):
    pref = "" if rol is None else f"{rol}_"
    titulo_rol = "Global" if rol is None else f"como {rol.upper()}"

    grafico_barra_apilado(
        df,
        [f"{pref}valid_pct", f"{pref}invalid_pct"],
        ["Válidas", "Inválidas"],
        f"Jugadas válidas vs inválidas {titulo_rol} LLaMA-3.2-3B({sufijo})",
        f"validas_invalidas_{titulo_rol}_{sufijo}.png".replace(" ", "_"),
        ["#1dd1a1", "#ff6b6b"]
    )

# =======================================================
# GENERAR TODOS LOS GRÁFICOS
# =======================================================
for df, sufijo in [(df_sft, "SinFT"), (df_ft, "ConFT")]:
    # Resultados
    graficos_resultados(df, sufijo)
    graficos_resultados(df, sufijo, rol="x")
    graficos_resultados(df, sufijo, rol="o")

    # Estrategias
    graficos_estrategias(df, sufijo)
    graficos_estrategias(df, sufijo, rol="x")
    graficos_estrategias(df, sufijo, rol="o")

    # Válidas vs inválidas
    graficos_validas_invalidas(df, sufijo)
    graficos_validas_invalidas(df, sufijo, rol="x")
    graficos_validas_invalidas(df, sufijo, rol="o")


# =======================================================
# PRUEBAS ESTADÍSTICAS SIN FT vs CON FT 
# =======================================================

# Alinear por oponente (CRÍTICO)
df_sft_stats = df_sft.sort_values("opponent_model").reset_index(drop=True)
df_ft_stats  = df_ft.sort_values("opponent_model").reset_index(drop=True)

# Verificación de seguridad
assert all(df_sft_stats["opponent_model"] == df_ft_stats["opponent_model"]), \
    "ERROR: los oponentes no están alineados entre SFT y FT"

metricas_test = [
    "valid_pct",
    "win_pct",
    "invalid_pct",
    "ofensiva_pct",
    "defensiva_pct",
]

print("\nPRUEBAS ESTADÍSTICAS SIN FT vs CON FT (Wilcoxon pareado)\n")

for m in metricas_test:
    x = df_sft_stats[m].values
    y = df_ft_stats[m].values

    if np.allclose(x, y):
        print(f"{m}: sin diferencias (valores idénticos)")
        continue

    stat, p = wilcoxon(x, y, zero_method="wilcox")
    print(f"{m}: W={stat:.3f}, p-valor={p:.4f}")

# =======================================================
# TAMAÑO DEL EFECTO (r)
# =======================================================

def effect_size_r(w, n):
    z = (w - (n * (n + 1) / 4)) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return abs(z) / math.sqrt(n)

print("\nTAMAÑO DEL EFECTO (r)\n")

n = len(df_sft_stats)

for m in metricas_test:
    x = df_sft_stats[m].values
    y = df_ft_stats[m].values

    if np.allclose(x, y):
        continue

    w, _ = wilcoxon(x, y)
    r = effect_size_r(w, n)

    print(f"{m}: r={r:.3f}")

# =======================================================
# PRUEBA DE SIGNOS
# =======================================================

print("\nPRUEBA DE SIGNOS\n")

for m in metricas_test:
    x = df_sft_stats[m].values
    y = df_ft_stats[m].values

    stat, p = sign_test(y - x)
    print(f"{m}: estadístico={stat}, p-valor={p:.4f}")

# =======================================================
# PERMUTATION TEST
# =======================================================

print("\nPERMUTATION TEST\n")

for m in metricas_test:
    x = df_sft_stats[m].values
    y = df_ft_stats[m].values

    res = permutation_test(
        (x, y),
        statistic=lambda a, b: np.mean(b - a),
        permutation_type="pairings",
        n_resamples=10000,
        alternative="two-sided"
    )

    print(
        f"{m}: diff_media={np.mean(y - x):.3f}, "
        f"p-valor={res.pvalue:.4f}"
    )

