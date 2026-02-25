"""Сравнение ProbLog, MLN (logistic) и plingo"""

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
        "-m",
        exists=True,
        help="CSV с p_success (ProbLog) и p_success_mln",
    ),
    block_ranked_plingo_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_plingo.csv",
        "--block_ranked_plingo",
        "-p",
        exists=True,
        help="CSV с p_success_plingo",
    ),
    fig_dist: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_compare_dist_problog_vs_mln_vs_plingo.png",
        "--fig_dist",
        help="Гистограммы распределений",
    ),
    fig_rank: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_compare_rank_problog_vs_mln_vs_plingo.png",
        "--fig_rank",
        help="Кривые ранжирования",
    ),
    fig_scatter_pm: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_scatter_problog_vs_plingo.png",
        "--fig_scatter_pm",
        help="Scatter ProbLog vs plingo",
    ),
    fig_scatter_mp: Path = typer.Option(
        PROCESSED_DATA_DIR / "fig_scatter_mln_vs_plingo.png",
        "--fig_scatter_mp",
        help="Scatter MLN vs plingo",
    ),
):
    logger.info("Сравнение ProbLog, MLN и plingo")

    # В mln-файле уже есть block_id + p_success + p_success_mln
    df = pd.read_csv(block_ranked_mln_path).copy()

    # В plingo-файле block_id + p_success_plingo
    dfp = pd.read_csv(block_ranked_plingo_path).copy()

    # Merge
    if "block_id" not in df.columns:
        raise ValueError(
            "block_ranked_mln.csv должен содержать column block_id"
        )
    if "block_id" not in dfp.columns:
        raise ValueError(
            "block_ranked_plingo.csv должен содержать column block_id"
        )

    df = df.merge(dfp, on="block_id", how="inner")

    # Drop missing
    df = df.dropna(
        subset=["p_success", "p_success_mln", "p_success_plingo"]
    ).copy()
    logger.info(f"Общих блоков для сравнения: {len(df)}")

    # 1) Distributions
    plt.figure()
    plt.hist(df["p_success"], bins=40, alpha=0.6, label="ProbLog")
    plt.hist(df["p_success_mln"], bins=40, alpha=0.6, label="MLN")
    plt.hist(
        df["p_success_plingo"], bins=40, alpha=0.6, label="plingo (LPMLN)"
    )
    plt.xlabel("Вероятность")
    plt.ylabel("Число блоков")
    plt.title("Распределение вероятностей")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dist, dpi=300)

    # 2) Ranking curves
    p_problog = (
        df["p_success"].sort_values(ascending=False).reset_index(drop=True)
    )
    p_mln = (
        df["p_success_mln"].sort_values(ascending=False).reset_index(drop=True)
    )
    p_plingo = (
        df["p_success_plingo"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    plt.figure()
    plt.plot(p_problog.index + 1, p_problog, label="ProbLog")
    plt.plot(p_mln.index + 1, p_mln, label="MLN")
    plt.plot(p_plingo.index + 1, p_plingo, label="plingo (LPMLN)")
    plt.xlabel("Ранг")
    plt.ylabel("Вероятность")
    plt.title("Кривые ранжирования")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_rank, dpi=300)

    # 3) Correlations
    pearson = df[["p_success", "p_success_mln", "p_success_plingo"]].corr(
        method="pearson"
    )
    logger.info("Корреляция Пирсона:\n" + pearson.to_string())

    # Spearman pairwise (scipy)
    s_pm = spearmanr(df["p_success"], df["p_success_mln"])
    s_pp = spearmanr(df["p_success"], df["p_success_plingo"])
    s_mp = spearmanr(df["p_success_mln"], df["p_success_plingo"])
    logger.info(f"Спирман ProbLog vs MLN:   {s_pm.statistic:.4f}")  # type: ignore
    logger.info(f"Спирман ProbLog vs plingo:{s_pp.statistic:.4f}")  # type: ignore
    logger.info(f"Спирман MLN vs plingo:    {s_mp.statistic:.4f}")  # type: ignore

    # 4) Scatter plots (очень хорошо смотрится в монографии)
    plt.figure()
    plt.scatter(df["p_success"], df["p_success_plingo"], s=8, alpha=0.5)
    plt.xlabel("ProbLog p_success")
    plt.ylabel("plingo p_success_plingo")
    plt.title("Scatter: ProbLog vs plingo")
    plt.tight_layout()
    plt.savefig(fig_scatter_pm, dpi=300)

    plt.figure()
    plt.scatter(df["p_success_mln"], df["p_success_plingo"], s=8, alpha=0.5)
    plt.xlabel("MLN p_success_mln")
    plt.ylabel("plingo p_success_plingo")
    plt.title("Scatter: MLN vs plingo")
    plt.tight_layout()
    plt.savefig(fig_scatter_mp, dpi=300)

    logger.success("Сравнение ProbLog, MLN и plingo выполнено")


if __name__ == "__main__":
    app()
