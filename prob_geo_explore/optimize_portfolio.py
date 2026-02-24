"""Оптимизация портфеля проектов."""

from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Binary,
    maximize,
    SolverFactory,
    value,
)


from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


def build_demo_economics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds demo Value and Cost in consistent units (e.g., million USD).
    You can tune constants for your monograph narrative.
    """
    # Base value of a commercial success (e.g., expected NPV of a successful discovery)
    V0 = 5000.0  # million USD

    # Multipliers from simple proxies
    near_mult = np.where(
        df.get("near_field", 0).fillna(0).astype(int) == 1, 1.25, 1.0
    )
    deep_mult = np.where(
        df.get("deepwater_penalty", 0).fillna(0).astype(int) == 1, 0.75, 1.0
    )

    df["value_musd"] = V0 * near_mult * deep_mult

    # Costs: exploration well
    C0 = 50.0  # million USD baseline
    deep_extra = np.where(
        df.get("deepwater_penalty", 0).fillna(0).astype(int) == 1, 30.0, 0.0
    )
    df["cost_musd"] = C0 + deep_extra

    # Expected monetary value
    df["emv_musd"] = df["p_success"] * df["value_musd"] - df["cost_musd"]
    return df


def solve_stochastic_knapsack(
    df: pd.DataFrame, budget_musd: float, solver: str = "cbc"
):
    """
    Maximize sum x_i * EMV_i subject to sum x_i * cost_i <= budget.
    """
    model = ConcreteModel()

    blocks = df["block_id"].tolist()
    idx = range(len(blocks))

    emv = df["emv_musd"].to_numpy()
    cost = df["cost_musd"].to_numpy()

    model.x = Var(idx, domain=Binary)

    model.budget = Constraint(
        expr=sum(model.x[i] * cost[i] for i in idx) <= budget_musd
    )

    model.obj = Objective(
        expr=sum(model.x[i] * emv[i] for i in idx), sense=maximize
    )

    opt = SolverFactory(solver)
    opt.solve(model, tee=False)

    chosen = [blocks[i] for i in idx if value(model.x[i]) > 0.5]
    total_cost = float(sum(cost[i] for i in idx if value(model.x[i]) > 0.5))
    total_emv = float(sum(emv[i] for i in idx if value(model.x[i]) > 0.5))

    return chosen, total_cost, total_emv


def solve_robust_knapsack(
    df: pd.DataFrame,
    budget_musd: float,
    p_delta: float = 0.15,
    solver: str = "cbc",
):
    """
    Robust version: uncertain probability p in [p*(1-delta), p*(1+delta)] clipped to [0,1].
    We maximize worst-case EMV => use p_low = p*(1-delta).
    Equivalent to using conservative EMV_low.

    This is the simplest robust counterpart and is very explainable in a monograph.
    """
    p = df["p_success"].to_numpy()
    p_low = np.clip(p * (1.0 - p_delta), 0.0, 1.0)

    df2 = df.copy()
    df2["emv_low_musd"] = p_low * df2["value_musd"] - df2["cost_musd"]

    model = ConcreteModel()
    blocks = df2["block_id"].tolist()
    idx = range(len(blocks))

    emv_low = df2["emv_low_musd"].to_numpy()
    cost = df2["cost_musd"].to_numpy()

    model.x = Var(idx, domain=Binary)
    model.budget = Constraint(
        expr=sum(model.x[i] * cost[i] for i in idx) <= budget_musd
    )
    model.obj = Objective(
        expr=sum(model.x[i] * emv_low[i] for i in idx), sense=maximize
    )

    opt = SolverFactory(solver)
    opt.solve(model, tee=False)

    chosen = [blocks[i] for i in idx if value(model.x[i]) > 0.5]
    total_cost = float(sum(cost[i] for i in idx if value(model.x[i]) > 0.5))
    total_emv_low = float(
        sum(emv_low[i] for i in idx if value(model.x[i]) > 0.5)
    )

    return chosen, total_cost, total_emv_low, df2


@app.command()
def main(
    block_ranked_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked.csv",
        "--block_ranked",
        "-bl",
        exists=True,
        help="Путь до файла с block_ranked.csv",
    ),
    budget: float = typer.Option(
        800,
        "--budget",
        "-b",
        help="Бюджет ГРР",
    ),
    solver: str = typer.Option(
        "cbc",
        "--solver",
        "-s",
        help="Солвер",
    ),
    mode: str = typer.Option(
        "stochastic",
        "--mode",
        "-m",
        help="Режим: stochastic, robust",
    ),
    p_delta: float = typer.Option(
        0.30,
        "--p_delta",
        "-p",
        help="Уровень надежности",
    ),
    out_csv: str = typer.Option(
        PROCESSED_DATA_DIR / "portfolio_solution.csv",
        "--out_csv",
        "-os",
        help="Путь до portfolio_solution.csv",
    ),
    out_txt: str = typer.Option(
        PROCESSED_DATA_DIR / "portfolio_summary.txt",
        "--out_txt",
        "-ot",
        help="Путь до portfolio_summary.txt",
    ),
    frontier: bool = typer.Option(
        False,
        "--frontier",
        "-fr",
        help="Бюджет -> EMV",
    ),
    bmin: float = typer.Option(
        200.0,
        "--bmin",
        "-bmin",
        help="Min бюджет",
    ),
    bmax: float = typer.Option(
        1200.0,
        "--bmax",
        "-bmax",
        help="Max бюджет",
    ),
    bstep: float = typer.Option(
        100.0,
        "--bstep",
        "-bstep",
        help="Шаг",
    ),
):
    logger.info("Оптимизация портфеля проектов...")

    df = pd.read_csv(block_ranked_path)
    if "block_id" not in df.columns or "p_success" not in df.columns:
        raise ValueError("Input must contain block_id and p_success")

    # economics
    df = build_demo_economics(df)
    logger.info("EMV stats (MUSD):")
    logger.info(df["emv_musd"].describe())
    logger.info("Positive EMV count:", (df["emv_musd"] > 0).sum())
    logger.info("Max EMV:", df["emv_musd"].max())

    if mode == "stochastic":
        chosen, total_cost, total_emv = solve_stochastic_knapsack(
            df, budget, solver=solver
        )
        chosen_df = df[df["block_id"].isin(chosen)].copy()
        chosen_df = chosen_df.sort_values("emv_musd", ascending=False)

        chosen_df.to_csv(out_csv, index=False, encoding="utf-8")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("MODE: STOCHASTIC (max expected EMV)\n")
            f.write(f"BUDGET (MUSD): {budget:.1f}\n")
            f.write(f"CHOSEN BLOCKS: {len(chosen)}\n")
            f.write(f"TOTAL COST (MUSD): {total_cost:.1f}\n")
            f.write(f"TOTAL EXPECTED EMV (MUSD): {total_emv:.1f}\n")

        logger.info(
            f"Результаты стохастической оптимизации: {out_csv}, {out_txt}"
        )
        logger.info(
            f"Выбрано={len(chosen)}, стоимость={total_cost:.1f}, ожидаемый EMV={total_emv:.1f}"
        )

    else:
        chosen, total_cost, total_emv_low, df2 = solve_robust_knapsack(
            df, budget, p_delta=p_delta, solver=solver
        )
        chosen_df = df2[df2["block_id"].isin(chosen)].copy()
        chosen_df = chosen_df.sort_values("emv_low_musd", ascending=False)

        chosen_df.to_csv(out_csv, index=False, encoding="utf-8")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("MODE: ROBUST (max worst-case EMV)\n")
            f.write(f"BUDGET (MUSD): {budget:.1f}\n")
            f.write(
                f"P UNCERTAINTY: p_low = p*(1-delta), delta={p_delta:.2f}\n"
            )
            f.write(f"CHOSEN BLOCKS: {len(chosen)}\n")
            f.write(f"TOTAL COST (MUSD): {total_cost:.1f}\n")
            f.write(f"TOTAL WORST-CASE EMV (MUSD): {total_emv_low:.1f}\n")

        print(f"Результаты робастной оптимизации: {out_csv}, {out_txt}")
        print(
            f"Выбрано={len(chosen)}, стоимость={total_cost:.1f}, пессимистический EMV={total_emv_low:.1f}"
        )

    if frontier:
        logger.info("Расчет  Бюджет -> EMV")
        budgets = np.arange(bmin, bmax + 1e-9, bstep)
        emvs = []
        costs = []
        ns = []
        for B in budgets:
            if mode == "stochastic":
                chosen, total_cost, total_emv = solve_stochastic_knapsack(
                    df, B, solver=solver
                )
                emvs.append(total_emv)
                costs.append(total_cost)
                ns.append(len(chosen))
            else:
                chosen, total_cost, total_emv_low, _df2 = (
                    solve_robust_knapsack(
                        df, B, p_delta=p_delta, solver=solver
                    )
                )
                emvs.append(total_emv_low)
                costs.append(total_cost)
                ns.append(len(chosen))

        # Plot
        plt.figure()
        plt.plot(budgets, emvs, marker="o")
        plt.xlabel("Budget (MUSD)")
        plt.ylabel("Portfolio objective (MUSD)")
        plt.title(f"Efficient frontier: Budget vs objective ({mode})")
        plt.tight_layout()
        out_fig = f"fig_frontier_{mode}.png"
        plt.savefig(out_fig, dpi=300)
        # plt.show()

        # Save table
        pd.DataFrame(
            {
                "budget_musd": budgets,
                "objective_musd": emvs,
                "cost_musd": costs,
                "n_blocks": ns,
            }
        ).to_csv(f"frontier_{mode}.csv", index=False, encoding="utf-8")

        print("Saved:", out_fig, f"frontier_{mode}.csv")

    logger.success("Результаты экспортированы в файл")


if __name__ == "__main__":
    app()
