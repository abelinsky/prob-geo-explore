"""Визуализация портфельной оптимизации"""

from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    block_ranked_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked.csv",
        "--block_ranked",
        "-b",
        exists=True,
        help="Путь до файла block_ranked.csv (с p_success)",
    ),
    portfolio_solution_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "portfolio_solution.csv",
        "--portfolio_solution_path",
        "-s",
        exists=True,
        help="Путь до файла portfolio_solution.csv (результат солвера)",
    ),
    mode: str = typer.Option(
        "stochastic",
        "--mode",
        "-m",
        help="Режим: stochastic, robust",
    ),
    out_prefix: str = typer.Option(
        "fig_portfolio",
        "--out_prefix",
        "-op",
        help="Название файла",
    ),
    postT0_eval: str = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_postT0_eval.csv",
        "--postT0_eval",
        "-pe",
        help="Название файла",
    ),
):
    logger.info("Ранжирование блоков...")

    ranked = pd.read_csv(block_ranked_path)
    sol = pd.read_csv(portfolio_solution_path)

    # Make sure we have key columns
    if "block_id" not in ranked.columns or "p_success" not in ranked.columns:
        raise ValueError("ranked must contain block_id and p_success")
    if "block_id" not in sol.columns:
        raise ValueError("solution must contain block_id")

    # Merge to ensure we have p_success in solution too
    solm = sol.merge(
        ranked[["block_id", "p_success"]], on="block_id", how="left"
    )

    # ---------- Plot 1: p_success vs EMV (all blocks) ----------
    # EMV column name differs by mode
    emv_col = "emv_musd" if "emv_musd" in ranked.columns else None
    if emv_col is None and "emv_low_musd" in ranked.columns:
        emv_col = "emv_low_musd"
    # If EMV not present in ranked, try from solution itself
    if emv_col is None:
        if "emv_musd" in sol.columns:
            emv_col = "emv_musd"
            ranked = ranked.merge(
                sol[["block_id", "emv_musd"]], on="block_id", how="left"
            )
        elif "emv_low_musd" in sol.columns:
            emv_col = "emv_low_musd"
            ranked = ranked.merge(
                sol[["block_id", "emv_low_musd"]], on="block_id", how="left"
            )
        else:
            raise ValueError(
                "No EMV column found. Run optimizer with economics columns saved in solution."
            )

    plt.figure()
    plt.scatter(ranked["p_success"], ranked[emv_col], s=10)
    plt.xlabel("Predicted probability p_success")
    plt.ylabel(f"{emv_col} (MUSD)")
    plt.title(f"Probability vs EMV ({mode})")
    plt.tight_layout()
    p1 = f"{out_prefix}_1_prob_vs_emv.png"
    plt.savefig(p1, dpi=300)
    plt.show()

    # ---------- Plot 2: Contribution of chosen blocks to total EMV ----------
    # Sort chosen by EMV and show cumulative contribution
    solm2 = solm.copy()
    if emv_col not in solm2.columns:
        # pull from solution if exists
        if emv_col in sol.columns:
            solm2 = sol.merge(
                ranked[["block_id", "p_success"]], on="block_id", how="left"
            )
        else:
            # try to pull emv from ranked
            solm2 = solm2.merge(
                ranked[["block_id", emv_col]], on="block_id", how="left"
            )

    solm2 = (
        solm2.dropna(subset=[emv_col])
        .sort_values(emv_col, ascending=False)
        .reset_index(drop=True)
    )
    solm2["cum_emv"] = solm2[emv_col].cumsum()
    total_emv = float(solm2[emv_col].sum()) if len(solm2) else 0.0
    solm2["cum_share"] = (
        solm2["cum_emv"] / total_emv if total_emv != 0 else np.nan
    )

    plt.figure()
    plt.plot(solm2.index + 1, solm2["cum_share"])
    plt.xlabel("Top-N chosen blocks (sorted by EMV contribution)")
    plt.ylabel("Cumulative share of portfolio EMV")
    plt.title(f"EMV Concentration in Portfolio ({mode})")
    plt.tight_layout()
    p2 = f"{out_prefix}_2_emv_concentration.png"
    plt.savefig(p2, dpi=300)
    plt.show()

    # ---------- Plot 3: Portfolio bar chart (top 25 EMV blocks) ----------
    topn = solm2.head(25).copy()
    if len(topn) > 0:
        plt.figure()
        plt.bar(range(len(topn)), topn[emv_col])
        plt.xticks(range(len(topn)), topn["block_id"], rotation=90)
        plt.xlabel("Chosen blocks (top 25 by EMV)")
        plt.ylabel(f"{emv_col} (MUSD)")
        plt.title(f"Top contributors in chosen portfolio ({mode})")
        plt.tight_layout()
        p3 = f"{out_prefix}_3_top_contributors_bar.png"
        plt.savefig(p3, dpi=300)
        plt.show()
    else:
        p3 = None

    # ---------- Plot 4 (optional): Backtest on post-T0 truth ----------
    # Requires block_ranked_postT0_eval.csv from earlier validation code
    p4 = None
    if postT0_eval:
        post = pd.read_csv(postT0_eval)
        if "y_postT0" not in post.columns:
            raise ValueError("postT0_eval must contain y_postT0 (0/1)")
        chosen_ids = set(solm["block_id"].tolist())
        post["chosen"] = post["block_id"].apply(
            lambda x: 1 if x in chosen_ids else 0
        )

        chosen_rate = (
            float(post.loc[post["chosen"] == 1, "y_postT0"].mean())
            if post["chosen"].sum() > 0
            else np.nan
        )
        base_rate = float(post["y_postT0"].mean())
        lift = chosen_rate / base_rate if base_rate > 0 else np.nan

        # simple 2-bar plot
        plt.figure()
        plt.bar([0, 1], [base_rate, chosen_rate])
        plt.xticks([0, 1], ["Base rate", "Chosen portfolio"])
        plt.ylabel("Post-T0 discovery rate")
        plt.title(f"Portfolio backtest (lift={lift:.2f})")
        plt.tight_layout()
        p4 = f"{out_prefix}_4_portfolio_backtest.png"
        plt.savefig(p4, dpi=300)
        plt.show()

        print(
            f"Backtest: base_rate={base_rate:.4f}, chosen_rate={chosen_rate:.4f}, lift={lift:.2f}"
        )

    print("Saved figures:")
    print(" -", p1)
    print(" -", p2)
    if p3:
        print(" -", p3)
    if p4:
        print(" -", p4)

    logger.success("Рисунки сформированы и сохранены")


if __name__ == "__main__":
    app()
