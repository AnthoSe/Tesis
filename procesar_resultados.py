import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def generar_reporte_llama(path_csv, output_csv, llama_model_id, llama_version):
    df = pd.read_csv(path_csv)

    df_terminal = df[(df["is_terminal"] == 1) & (df["valid"] == 1)]
    df_llama = df[df["model"] == llama_model_id]

    pct = lambda x, t: round((x / t) * 100, 2) if t > 0 else 0
    estrategias = ["greedy", "defensiva", "ofensiva", "aleatoria", "invalida"]

    reportes = []

    for opponent in df["model"].unique():
        if opponent == llama_model_id:
            continue

        matches = df_llama[
            df_llama["id_match"].isin(
                df[df["model"] == opponent]["id_match"].unique()
            )
        ]["id_match"].unique()

        if len(matches) == 0:
            continue

        # =====================
        # RESULTADOS GLOBALES
        # =====================
        wins = losses = draws = 0

        for match_id in matches:
            df_match = df_terminal[df_terminal["id_match"] == match_id]
            if df_match.empty:
                continue

            last_move = df_match.sort_values("timestamp").iloc[-1]

            if last_move["win"] == 1 and last_move["model"] == llama_model_id:
                wins += 1
            elif last_move["win"] == 1:
                losses += 1
            else:
                draws += 1

        total_games = wins + losses + draws

        df_moves_all = df_llama[df_llama["id_match"].isin(matches)]
        total_moves = len(df_moves_all)
        
        valid_moves = (df_moves_all["valid"] == 1).sum()
        invalid_moves = (df_moves_all["valid"] == 0).sum()

        strat_counts_all = {
            s: (df_moves_all["strategy"] == s).sum()
            for s in estrategias 
        }

        # =====================
        # RESULTADOS POR ROL
        # =====================
        role_data = {}

        for role in ["x", "o"]:
            role_matches = df_llama[
                (df_llama["player"] == role) &
                (df_llama["id_match"].isin(matches))
            ]["id_match"].unique()

            rw = rl = rd = 0

            for match_id in role_matches:
                df_match = df_terminal[df_terminal["id_match"] == match_id]
                if df_match.empty:
                    continue

                last_move = df_match.sort_values("timestamp").iloc[-1]

                if last_move["win"] == 1 and last_move["model"] == llama_model_id:
                    rw += 1
                elif last_move["win"] == 1:
                    rl += 1
                else:
                    rd += 1

            df_moves_role = df_llama[
                (df_llama["id_match"].isin(role_matches)) &
                (df_llama["player"] == role)
            ]

            total_moves_role = len(df_moves_role)
            valid_moves_role = (df_moves_role["valid"] == 1).sum()
            invalid_moves_role = (df_moves_role["valid"] == 0).sum()

            strat_counts_role = {
                s: (df_moves_role["strategy"] == s).sum()
                for s in estrategias
            }

            role_data[role] = {
                "games": len(role_matches),
                "wins": rw,
                "losses": rl,
                "draws": rd,
                "moves": total_moves_role,
                "valid_moves": valid_moves_role,
                "invalid_moves": invalid_moves_role,
                "strategies": strat_counts_role
            }

        # =====================
        # CONSTRUIR FILA FINAL
        # =====================
        reporte = {
            "llama_version": llama_version,
            "opponent_model": opponent,

            # Global
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_pct": pct(wins, total_games),
            "loss_pct": pct(losses, total_games),
            "draw_pct": pct(draws, total_games),
            "valid_moves": valid_moves,
            "invalid_moves": invalid_moves,
            "valid_pct": pct(valid_moves, total_moves),
            "invalid_pct": pct(invalid_moves, total_moves),

        }

        for s in estrategias:
            reporte[f"{s}_count"] = strat_counts_all[s]
            reporte[f"{s}_pct"] = pct(strat_counts_all[s], total_moves)


        # X y O
        for role in ["x", "o"]:
            rd = role_data[role]

            reporte[f"{role}_games"] = rd["games"]
            reporte[f"{role}_wins"] = rd["wins"]
            reporte[f"{role}_losses"] = rd["losses"]
            reporte[f"{role}_draws"] = rd["draws"]

            reporte[f"{role}_win_pct"] = pct(rd["wins"], rd["games"])
            reporte[f"{role}_loss_pct"] = pct(rd["losses"], rd["games"])
            reporte[f"{role}_draw_pct"] = pct(rd["draws"], rd["games"])

            reporte[f"{role}_valid_moves"] = rd["valid_moves"]
            reporte[f"{role}_invalid_moves"] = rd["invalid_moves"]

            reporte[f"{role}_valid_pct"] = pct(rd["valid_moves"], rd["moves"])
            reporte[f"{role}_invalid_pct"] = pct(rd["invalid_moves"], rd["moves"])

            for s in estrategias:
                reporte[f"{role}_{s}_count"] = rd["strategies"][s]
                reporte[f"{role}_{s}_pct"] = pct(
                    rd["strategies"][s], rd["moves"]
                )

        reportes.append(reporte)

    df_report = pd.DataFrame(reportes)
    df_report.to_csv(output_csv, index=False)
    print(f"✔ Reporte generado: {output_csv}")

# IDs de LLaMA
LLAMA_SFT_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_FT_ID = "llama-3.2-3b-lora-tictactoe"

generar_reporte_llama(
    os.path.join(DATA_DIR, "resultados_tictactoe_sft.csv"),
    os.path.join(DATA_DIR, "reportes_sft.csv"),
    LLAMA_SFT_ID,
    "sft"
)

generar_reporte_llama(
    os.path.join(DATA_DIR, "resultados_tictactoe_ft.csv"),
    os.path.join(DATA_DIR, "reportes_ft.csv"),
    LLAMA_FT_ID,
    "ft"
)
