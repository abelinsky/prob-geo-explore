"""Сравнение Problog и Markov Logic Network"""

from loguru import logger
import typer
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    block_ranked_mln_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_mln.csv",
        "--block_ranked_mln",
        "-b",
        exists=True,
        help="Путь до файла, куда будут записаны ранжированные блоки",
    ),
    fig1: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_compare_dist_problog_vs_mln.png",
        "--fig1",
        "-f1",
        help="Путь до файла, куда будет записана диаграмма сравнения распределений",
    ),
    fig2: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_compare_rank_problog_vs_mln.png",
        "--fig2",
        "-f2",
        help="Путь до файла, куда будет записана диаграмма сравнения результатов ранжирования",
    ),
):
    logger.info(
        "Сравнение результатов расчетов Problog и Markov Logic Network"
    )
    df = (
        pd.read_csv(block_ranked_mln_path)
        .dropna(subset=["p_success", "p_success_mln"])
        .copy()
    )

    # 1) Distribution comparison
    plt.figure()
    plt.hist(df["p_success"], bins=40, alpha=0.7, label="ProbLog")
    plt.hist(df["p_success_mln"], bins=40, alpha=0.7, label="MLN (logistic)")
    plt.xlabel("Probability")
    plt.ylabel("Number of blocks")
    plt.title("Probability distribution: ProbLog vs MLN (logistic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=300)

    # 2) Ranking curves
    p1 = df["p_success"].sort_values(ascending=False).reset_index(drop=True)
    p2 = (
        df["p_success_mln"].sort_values(ascending=False).reset_index(drop=True)
    )

    plt.figure()
    plt.plot(p1.index + 1, p1, label="ProbLog")
    plt.plot(p2.index + 1, p2, label="MLN (logistic)")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.title("Ranking curves: ProbLog vs MLN (logistic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=300)

    # 3) Correlation
    corr = float(df[["p_success", "p_success_mln"]].corr().iloc[0, 1])
    logger.info(f"Корреляция Пирсона: {corr:.4f}")

    spearman = spearmanr(df["p_success"], df["p_success_mln"])
    logger.info(f"Корреляция Спирмана: {spearman.statistic:.4f}")  # type: ignore

    logger.success(
        "Сравнение результатов Problog и Markov Logic Network выполнено"
    )


if __name__ == "__main__":
    app()
