from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import typer
from loguru import logger
import numpy as np

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


def _compute_precision_at_k(ranked_path, truth_df, prob_col, max_k=500):
    ranked = pd.read_csv(ranked_path)

    if "block_id" not in ranked.columns:
        raise ValueError(f"{ranked_path} must contain block_id")
    if prob_col not in ranked.columns:
        raise ValueError(f"{prob_col} not found in {ranked_path}")

    if "post_t0_discovery" in truth_df.columns:
        ycol = "post_t0_discovery"
    elif "y_postT0" in truth_df.columns:
        ycol = "y_postT0"
    else:
        raise ValueError("truth_df must contain post_t0_discovery or y_postT0")

    merged = ranked[["block_id", prob_col]].merge(
        truth_df[["block_id", ycol]],
        on="block_id",
        how="left",
    )

    merged[ycol] = merged[ycol].fillna(0).astype(int)

    p = (
        pd.to_numeric(merged[prob_col], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy()
    )
    y = merged[ycol].to_numpy()

    order = np.argsort(-p)
    y_sorted = y[order]

    max_k = min(max_k, len(y_sorted))
    ks = np.arange(1, max_k + 1)

    cumsum_hits = np.cumsum(y_sorted[:max_k])
    precision_at_k = cumsum_hits / ks

    return ks, precision_at_k


def plot_lift_all_methods(
    problog_path: Path,
    mln_path: Path,
    plingo_path: Path,
    truth_path: Path,
    out_png: Path,
    max_k: int = 500,
):

    truth = pd.read_csv(truth_path)

    plt.figure()

    # ProbLog
    ks, prec_problog = _compute_precision_at_k(
        problog_path, truth, "p_success", max_k
    )
    plt.plot(ks, prec_problog, label=f"ProbLog (P@100={prec_problog[99]:.3f})")

    # MLN
    ks, prec_mln = _compute_precision_at_k(
        mln_path, truth, "p_success_mln", max_k
    )
    plt.plot(ks, prec_mln, label=f"MLN (P@100={prec_mln[99]:.3f})")

    # plingo
    ks, prec_plingo = _compute_precision_at_k(
        plingo_path, truth, "p_success_plingo", max_k
    )
    plt.plot(ks, prec_plingo, label=f"plingo (P@100={prec_plingo[99]:.3f})")

    # Base rate
    # if "post_t0_discovery" in truth.columns:
    #     base_rate = truth["post_t0_discovery"].mean()
    # else:
    #     base_rate = truth["y_postT0"].mean()

    # plt.plot(
    #     ks,
    #     np.full_like(ks, base_rate, dtype=float),
    #     linestyle="--",
    #     label=f"Base rate ({base_rate:.3f})",
    # )

    plt.xlabel("K (top-ranked blocks)")
    plt.ylabel("Post-T0 discovery rate")
    plt.title("Lift curves: ProbLog vs MLN vs plingo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


@app.command()
def main(
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--data_dir",
        "-dd",
        exists=True,
        help="Путь до файла с результатами расчетов",
    ),
):

    summary_files = sorted(data_dir.glob("portfolio_compare_summary_*.csv"))
    overlap_files = sorted(
        data_dir.glob("portfolio_compare_overlap_jaccard_*.csv")
    )

    if not summary_files:
        raise FileNotFoundError("No portfolio_compare_summary_*.csv found")
    if not overlap_files:
        raise FileNotFoundError(
            "No portfolio_compare_overlap_jaccard_*.csv found"
        )

    summary = pd.read_csv(summary_files[-1])
    overlap = pd.read_csv(overlap_files[-1])

    print("Using summary:", summary_files[-1].name)
    print("Using overlap:", overlap_files[-1].name)

    # -------------------------------------------------------
    # 1. Portfolio objective (EMV)
    # -------------------------------------------------------
    plt.figure()

    if (
        "worst_case_emv" in summary.columns
        and summary["worst_case_emv"].notna().any()
    ):
        value_col = "worst_case_emv"
    else:
        value_col = "emv"

    plt.bar(summary["method"], summary[value_col])
    plt.xlabel("Method")
    plt.ylabel("Portfolio objective (MUSD)")
    plt.title("Portfolio EMV by probabilistic reasoning method")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DATA_DIR / "fig_portfolio_objective_bar.png", dpi=300
    )
    plt.close()

    # -------------------------------------------------------
    # 2. Portfolio size
    # -------------------------------------------------------
    plt.figure()

    plt.bar(summary["method"], summary["chosen_n"])
    plt.xlabel("Method")
    plt.ylabel("Number of selected blocks")
    plt.title("Portfolio size by method")
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "fig_portfolio_size_bar.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # 3. Portfolio overlap (Jaccard heatmap)
    # -------------------------------------------------------
    pivot = overlap.pivot(index="a", columns="b", values="jaccard")

    plt.figure()
    plt.imshow(pivot.values)
    plt.xticks(range(len(pivot.columns)), pivot.columns)  # type: ignore
    plt.yticks(range(len(pivot.index)), pivot.index)  # type: ignore
    plt.colorbar()
    plt.title("Portfolio overlap (Jaccard similarity)")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DATA_DIR / "fig_portfolio_overlap_heatmap.png", dpi=300
    )
    plt.close()

    # -------------------------------------------------------
    # 4. Method ranking by EMV
    # -------------------------------------------------------
    sorted_summary = summary.sort_values(value_col, ascending=False)

    plt.figure()
    plt.plot(sorted_summary["method"], sorted_summary[value_col], marker="o")
    plt.xlabel("Method")
    plt.ylabel("Portfolio objective (MUSD)")
    plt.title("Ranking of methods by portfolio objective")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DATA_DIR / "fig_portfolio_method_ranking.png", dpi=300
    )
    plt.close()

    # -------------------------------------------------------
    # 5. Budget–Objective Frontier (all methods)
    # -------------------------------------------------------

    budgets = np.linspace(100, 1000, 10)  # можно изменить диапазон
    methods = ["problog", "mln", "plingo"]

    plt.figure()

    for method in methods:
        objectives = []

        for B in budgets:
            # cmd = (
            #     f"python optimize_portfolio.py compare-all "
            #     f"--mode robust --budget {int(B)} --temp_plingo 3.0"
            # )
            # Предполагается, что compare_all уже создал summary
            # Читаем последний summary
            summary_files = sorted(
                data_dir.glob("portfolio_compare_summary_*.csv")
            )
            summary = pd.read_csv(summary_files[-1])

            row = summary[summary["method"] == method]
            if "worst_case_emv" in row.columns:
                val = row["worst_case_emv"].values[0]
            else:
                val = row["emv"].values[0]

            objectives.append(val)

        plt.plot(budgets, objectives)

    plt.xlabel("Budget (MUSD)")
    plt.ylabel("Portfolio objective (MUSD)")
    plt.title("Budget–Objective Frontier by Method")
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "fig_budget_frontier.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # 6. Portfolio Lift by Method
    # -------------------------------------------------------

    # features = pd.read_csv(data_dir / "block_features.csv")

    summary_files = sorted(data_dir.glob("portfolio_compare_*_B*.csv"))
    latest_prefix = summary_files[-1].name.split("_summary")[0]

    methods = ["problog", "mln", "plingo"]

    plt.figure()

    for method in methods:
        sol_file = sorted(data_dir.glob(f"{latest_prefix}_{method}_*.csv"))[-1]
        sol = pd.read_csv(sol_file)

        # merged = sol.merge(features, on="block_id", how="left")
        truth = pd.read_csv(
            PROCESSED_DATA_DIR / "block_ranked_postT0_eval.csv"
        )

        merged = sol.merge(
            truth[["block_id", "y_postT0"]], on="block_id", how="left"
        )
        merged["y_postT0"] = merged["y_postT0"].fillna(0).astype(int)
        hit_rate = merged["y_postT0"].mean()

        if "y_postT0" not in merged.columns:
            continue

        hit_rate = merged["y_postT0"].mean()

        plt.scatter(method, hit_rate)

    plt.xlabel("Method")
    plt.ylabel("Post-T0 success rate")
    plt.title("Portfolio Lift by Method")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DATA_DIR / "fig_portfolio_lift_by_method.png", dpi=300
    )
    plt.close()

    # -------------------------------------------------------
    # 7. Scatter EMV_plingo vs EMV_problog
    # -------------------------------------------------------

    merged = pd.read_csv(sorted(data_dir.glob("block_ranked_mln*.csv"))[-1])
    plingo = pd.read_csv(sorted(data_dir.glob("block_ranked_plingo*.csv"))[-1])

    merged = merged.merge(
        plingo[["block_id", "p_success_plingo"]], on="block_id"
    )

    # Строим EMV для каждого метода
    V0 = 5000.0
    C0 = 50.0

    merged["emv_problog"] = merged["p_success"] * V0 - C0
    merged["emv_plingo"] = merged["p_success_plingo"] * V0 - C0

    plt.figure()
    plt.scatter(merged["emv_problog"], merged["emv_plingo"], s=8)
    plt.xlabel("EMV (ProbLog)")
    plt.ylabel("EMV (plingo)")
    plt.title("EMV comparison: ProbLog vs plingo")
    plt.tight_layout()
    plt.savefig(
        PROCESSED_DATA_DIR / "fig_scatter_emv_problog_vs_plingo.png", dpi=300
    )
    plt.close()

    # -------------------------------------------------------
    # 8. Lift curves
    # -------------------------------------------------------
    plot_lift_all_methods(
        problog_path=PROCESSED_DATA_DIR / "block_ranked.csv",
        mln_path=PROCESSED_DATA_DIR / "block_ranked_mln.csv",
        plingo_path=PROCESSED_DATA_DIR / "block_ranked_plingo.csv",
        truth_path=PROCESSED_DATA_DIR / "block_ranked_postT0_eval.csv",
        out_png=PROCESSED_DATA_DIR / "fig_lift_all_methods.png",
        max_k=500,
    )

    logger.success(f"Рисунки сохранены в {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    app()
