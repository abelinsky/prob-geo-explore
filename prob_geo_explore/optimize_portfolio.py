"""Оптимизация портфеля проектов."""

from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np

from pyomo.environ import (
    ConcreteModel,
    Var,
    Binary,  # type: ignore
    Constraint,
    Objective,
    maximize,
    value,
)
from pyomo.opt import SolverFactory

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


def _find_ranked_file(
    pattern: str, data_dir: Path = PROCESSED_DATA_DIR
) -> Path:
    """Найти единственный файл в PROCESSED_DATA_DIR по шаблону."""
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Не найдены файлы, соответствующие паттерну {pattern} в каталоге {data_dir}"
        )
    # берём самый свежий/последний по имени
    return files[-1]


def _load_ranked_merged(data_dir: Path) -> pd.DataFrame:
    """
    Загружает ranked-файлы и приводит к единой таблице:
    block_id, p_success, p_success_mln, p_success_plingo
    """
    # Обычно у тебя уже есть block_ranked_mln.csv (ProbLog+MLN)
    f_mln = _find_ranked_file("block_ranked_mln*.csv", data_dir=data_dir)
    df = pd.read_csv(f_mln).copy()

    # Добавляем plingo
    f_pl = _find_ranked_file("block_ranked_plingo*.csv", data_dir=data_dir)
    dfp = pd.read_csv(f_pl).copy()

    if "block_id" not in df.columns:
        raise ValueError(f"{f_mln.name} must contain block_id")
    if "block_id" not in dfp.columns:
        raise ValueError(f"{f_pl.name} must contain block_id")

    # Нормализуем имена колонок вероятностей
    # (подстрой под свои названия, если отличаются)
    if "p_success" not in df.columns:
        raise ValueError(f"{f_mln.name} must contain p_success (ProbLog)")
    if "p_success_mln" not in df.columns:
        # иногда у тебя может называться p_success_logistic или p_mln
        cand = [c for c in df.columns if "mln" in c.lower()]
        raise ValueError(
            f"{f_mln.name} must contain p_success_mln. Found mln-like cols: {cand}"
        )

    if "p_success_plingo" not in dfp.columns:
        cand = [c for c in dfp.columns if "plingo" in c.lower()]
        raise ValueError(
            f"{f_pl.name} must contain p_success_plingo. Found plingo-like cols: {cand}"
        )

    merged = df.merge(
        dfp[["block_id", "p_success_plingo"]], on="block_id", how="inner"
    )
    merged = merged.dropna(
        subset=["p_success", "p_success_mln", "p_success_plingo"]
    ).copy()

    logger.info(
        f"Merged ranked table: {len(merged)} blocks | "
        f"from {f_mln.name} + {f_pl.name}"
    )
    return merged


def _apply_temperature(p: pd.Series, temp: float) -> pd.Series:
    """
    Температурная калибровка (logit / temp), чтобы "охладить" слишком уверенные вероятности.
    temp=1.0 -> без изменений; temp>1.0 -> ближе к 0.5.
    """
    if temp is None or float(temp) == 1.0:
        return p
    eps = 1e-6
    x = p.clip(eps, 1 - eps).astype(float)
    logit = np.log(x / (1 - x))
    x2 = 1 / (1 + np.exp(logit / float(temp)))
    return pd.Series(x2, index=p.index)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_demo_economics(
    df: pd.DataFrame,
    block_features_path: "Path | None" = None,
    data_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """
    Builds demo Value and Cost in million USD.
    Works even if df contains only: block_id, p_success.

    It merges in minimal context features from block_features.csv:
      - near_field
      - deepwater_penalty
    (Optionally can be extended with water depth, distance to field, etc.)
    """
    if "block_id" not in df.columns or "p_success" not in df.columns:
        raise ValueError(
            "build_demo_economics expects columns: block_id, p_success"
        )

    # Locate features file
    if block_features_path is None:
        block_features_path = data_dir / "block_features.csv"

    feat = pd.read_csv(block_features_path)

    if "block_id" not in feat.columns:
        raise ValueError("block_features.csv must contain block_id")

    # Keep only what economics needs (extend if you want)
    keep_cols = ["block_id"]
    for c in ["near_field", "deepwater_penalty"]:
        if c in feat.columns:
            keep_cols.append(c)

    feat = feat[keep_cols].copy()

    # Merge (left keeps df ordering)
    out = df.merge(feat, on="block_id", how="left")

    # Default missing to 0
    near = out.get("near_field", 0)
    deep = out.get("deepwater_penalty", 0)
    near = pd.to_numeric(near, errors="coerce").fillna(0).astype(int)  # type: ignore
    deep = pd.to_numeric(deep, errors="coerce").fillna(0).astype(int)  # type: ignore

    # ----- Demo economics -----
    V0 = 5000.0  # million USD: base value of success
    near_mult = np.where(near == 1, 1.25, 1.0)
    deep_mult = np.where(deep == 1, 0.75, 1.0)
    out["value_musd"] = V0 * near_mult * deep_mult

    C0 = 50.0  # million USD: base exploration well cost
    deep_extra = np.where(deep == 1, 30.0, 0.0)
    out["cost_musd"] = C0 + deep_extra

    out["p_success"] = (
        pd.to_numeric(out["p_success"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    out["emv_musd"] = out["p_success"] * out["value_musd"] - out["cost_musd"]

    # Optional: sanity log
    miss = (
        int(out["near_field"].isna().sum())
        if "near_field" in out.columns
        else len(out)
    )
    if miss > 0:
        logger.warning(
            f"build_demo_economics: {miss} blocks missing near_field/deepwater info (filled as 0)"
        )

    return out


def solve_stochastic_knapsack(
    df: pd.DataFrame, budget_musd: float, solver: str = "cbc"
) -> dict:
    """
    Maximize sum x_i * EMV_i subject to sum x_i * cost_i <= budget.

    Returns a dict compatible with compare_all():
      {
        "chosen_blocks": [...],
        "total_cost_musd": float,
        "emv": float,
        "status": str,
        "solver": str,
        "budget_musd": float,
      }
    """
    required = {"block_id", "emv_musd", "cost_musd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"solve_stochastic_knapsack: missing columns: {missing}"
        )

    model = ConcreteModel()

    blocks = df["block_id"].tolist()
    idx = range(len(blocks))

    emv = df["emv_musd"].to_numpy(dtype=float)
    cost = df["cost_musd"].to_numpy(dtype=float)

    model.x = Var(idx, domain=Binary)

    model.budget = Constraint(
        expr=sum(model.x[i] * cost[i] for i in idx) <= float(budget_musd)
    )

    model.obj = Objective(
        expr=sum(model.x[i] * emv[i] for i in idx), sense=maximize
    )

    opt = SolverFactory(solver)
    results = opt.solve(model, tee=False)

    chosen_idx = [i for i in idx if value(model.x[i]) > 0.5]  # type: ignore
    chosen = [blocks[i] for i in chosen_idx]

    total_cost = float(sum(cost[i] for i in chosen_idx))
    total_emv = float(sum(emv[i] for i in chosen_idx))

    return {
        "chosen_blocks": chosen,
        "total_cost_musd": total_cost,
        "emv": total_emv,
        "status": str(
            getattr(results, "solver", {}).get("termination_condition", "")
        ),
        "solver": solver,
        "budget_musd": float(budget_musd),
    }


def solve_robust_knapsack(
    df: pd.DataFrame,
    budget_musd: float,
    p_delta: float = 0.15,
    solver: str = "cbc",
) -> dict:
    """
    Robust version: uncertain probability p in [p*(1-delta), p*(1+delta)] clipped to [0,1].
    We maximize worst-case EMV => use p_low = p*(1-delta).

    Returns a dict compatible with compare_all():
      {
        "chosen_blocks": [...],
        "total_cost_musd": float,
        "worst_case_emv": float,
        "p_delta": float,
        "status": str,
        "solver": str,
        "budget_musd": float,
        "df": pd.DataFrame   # df with emv_low_musd + p_low (for diagnostics)
      }
    """
    required = {"block_id", "p_success", "value_musd", "cost_musd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"solve_robust_knapsack: missing columns: {missing}")

    p = df["p_success"].to_numpy(dtype=float)
    p_low = np.clip(p * (1.0 - float(p_delta)), 0.0, 1.0)

    df2 = df.copy()
    df2["p_low"] = p_low
    df2["emv_low_musd"] = df2["p_low"] * df2["value_musd"].astype(float) - df2[
        "cost_musd"
    ].astype(float)

    model = ConcreteModel()
    blocks = df2["block_id"].tolist()
    idx = range(len(blocks))

    emv_low = df2["emv_low_musd"].to_numpy(dtype=float)
    cost = df2["cost_musd"].to_numpy(dtype=float)

    model.x = Var(idx, domain=Binary)
    model.budget = Constraint(
        expr=sum(model.x[i] * cost[i] for i in idx) <= float(budget_musd)
    )
    model.obj = Objective(
        expr=sum(model.x[i] * emv_low[i] for i in idx), sense=maximize
    )

    opt = SolverFactory(solver)
    results = opt.solve(model, tee=False)

    chosen_idx = [i for i in idx if value(model.x[i]) > 0.5]  # type: ignore
    chosen = [blocks[i] for i in chosen_idx]

    total_cost = float(sum(cost[i] for i in chosen_idx))
    total_emv_low = float(sum(emv_low[i] for i in chosen_idx))

    return {
        "chosen_blocks": chosen,
        "total_cost_musd": total_cost,
        "worst_case_emv": total_emv_low,
        "p_delta": float(p_delta),
        "status": str(
            getattr(results, "solver", {}).get("termination_condition", "")
        ),
        "solver": solver,
        "budget_musd": float(budget_musd),
        "df": df2,
    }


@app.command()
def main(
    mode: str = typer.Option("robust", "--mode", help="stochastic | robust"),
    budget_musd: float = typer.Option(600.0, "--budget", help="Бюджет (MUSD)"),
    delta: float = typer.Option(
        0.3, "--delta", help="Робастность: p_low = p*(1-delta)"
    ),
    # Температуры: обычно для ProbLog/MLN =1, для plingo можно 2..5 если он слишком уверенный
    temp_problog: float = typer.Option(1.0, "--temp_problog"),
    temp_mln: float = typer.Option(1.0, "--temp_mln"),
    temp_plingo: float = typer.Option(3.0, "--temp_plingo"),
    out_prefix: str = typer.Option("portfolio_compare", "--out_prefix"),
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--data_dir",
        "-dd",
        help="Путь до каталога с результатами расчетов",
    ),
):
    """
    Сравнение портфельных решений для ProbLog / MLN / plingo при одном бюджете.
    Сохраняет:
      - portfolio_solution_<method>.csv
      - portfolio_summary.csv
      - portfolio_overlap_jaccard.csv
    """
    mode = mode.strip().lower()
    if mode not in ("stochastic", "robust"):
        raise ValueError("mode must be stochastic or robust")

    ranked = _load_ranked_merged(data_dir)

    methods = [
        ("problog", "p_success", temp_problog),
        ("mln", "p_success_mln", temp_mln),
        ("plingo", "p_success_plingo", temp_plingo),
    ]

    solutions = {}
    summaries = []

    for method, col, temp in methods:
        df = (
            ranked[["block_id", col]].rename(columns={col: "p_success"}).copy()
        )
        df["p_success"] = _apply_temperature(df["p_success"], temp)

        econ = build_demo_economics(df)

        if mode == "stochastic":
            sol = solve_stochastic_knapsack(econ, budget_musd=budget_musd)
            obj_name = "emv"
        else:
            sol = solve_robust_knapsack(
                econ, budget_musd=budget_musd, p_delta=delta
            )
            obj_name = "worst_case_emv"

        # sol должен содержать выбранные блоки и агрегаты
        # Я предполагаю, что у тебя sol возвращает dict с:
        #   chosen_blocks (list[str]) и total_cost, objective
        # Если структура другая — смотри комментарий ниже.
        chosen = sol["chosen_blocks"]
        solutions[method] = set(chosen)

        out_sol = (
            PROCESSED_DATA_DIR
            / f"{out_prefix}_{method}_{mode}_B{int(budget_musd)}.csv"
        )
        pd.DataFrame({"block_id": chosen}).to_csv(
            out_sol, index=False, encoding="utf-8"
        )

        summaries.append(
            {
                "method": method,
                "mode": mode,
                "budget_musd": budget_musd,
                "delta": delta if mode == "robust" else np.nan,
                "temp": temp,
                "chosen_n": len(chosen),
                "total_cost_musd": float(sol.get("total_cost_musd", np.nan)),
                obj_name: float(
                    sol.get(obj_name, sol.get("objective", np.nan))
                ),
            }
        )

        logger.info(
            f"{method}: chosen={len(chosen)} "
            f"cost={summaries[-1]['total_cost_musd']:.2f} "
            f"{obj_name}={summaries[-1][obj_name]:.2f}"
        )

    # summary table
    out_sum = (
        PROCESSED_DATA_DIR
        / f"{out_prefix}_summary_{mode}_B{int(budget_musd)}.csv"
    )
    pd.DataFrame(summaries).to_csv(out_sum, index=False, encoding="utf-8")

    # overlap matrix (Jaccard)
    rows = []
    keys = [m[0] for m in methods]
    for a in keys:
        for b in keys:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "jaccard": _jaccard(solutions[a], solutions[b]),
                }
            )

    out_j = (
        PROCESSED_DATA_DIR
        / f"{out_prefix}_overlap_jaccard_{mode}_B{int(budget_musd)}.csv"
    )
    pd.DataFrame(rows).to_csv(out_j, index=False, encoding="utf-8")

    logger.success(
        f"Saved: {out_sum.name}, {out_j.name} and per-method solutions"
    )


# @app.command()
# def main(
#     block_ranked_path: Path = typer.Option(
#         PROCESSED_DATA_DIR / "block_ranked.csv",
#         "--block_ranked",
#         "-bl",
#         exists=True,
#         help="Путь до файла с block_ranked.csv",
#     ),
#     budget: float = typer.Option(
#         800,
#         "--budget",
#         "-b",
#         help="Бюджет ГРР",
#     ),
#     solver: str = typer.Option(
#         "cbc",
#         "--solver",
#         "-s",
#         help="Солвер",
#     ),
#     mode: str = typer.Option(
#         "stochastic",
#         "--mode",
#         "-m",
#         help="Режим: stochastic, robust",
#     ),
#     p_delta: float = typer.Option(
#         0.30,
#         "--p_delta",
#         "-p",
#         help="Уровень надежности",
#     ),
#     out_csv: str = typer.Option(
#         PROCESSED_DATA_DIR / "portfolio_solution.csv",
#         "--out_csv",
#         "-os",
#         help="Путь до portfolio_solution.csv",
#     ),
#     out_txt: str = typer.Option(
#         PROCESSED_DATA_DIR / "portfolio_summary.txt",
#         "--out_txt",
#         "-ot",
#         help="Путь до portfolio_summary.txt",
#     ),
#     frontier: bool = typer.Option(
#         False,
#         "--frontier",
#         "-fr",
#         help="Бюджет -> EMV",
#     ),
#     bmin: float = typer.Option(
#         200.0,
#         "--bmin",
#         "-bmin",
#         help="Min бюджет",
#     ),
#     bmax: float = typer.Option(
#         1200.0,
#         "--bmax",
#         "-bmax",
#         help="Max бюджет",
#     ),
#     bstep: float = typer.Option(
#         100.0,
#         "--bstep",
#         "-bstep",
#         help="Шаг",
#     ),
# ):
#     logger.info("Оптимизация портфеля проектов...")

#     df = pd.read_csv(block_ranked_path)
#     if "block_id" not in df.columns or "p_success" not in df.columns:
#         raise ValueError("Input must contain block_id and p_success")

#     # economics
#     df = build_demo_economics(df)
#     logger.info("EMV stats (MUSD):")
#     logger.info(df["emv_musd"].describe())
#     logger.info(f"Positive EMV count: {(df["emv_musd"] > 0).sum()}")
#     logger.info(f"Max EMV: {df["emv_musd"].max()}")

#     if mode == "stochastic":
#         chosen, total_cost, total_emv = solve_stochastic_knapsack(
#             df, budget, solver=solver
#         )
#         chosen_df = df[df["block_id"].isin(chosen)].copy()
#         chosen_df = chosen_df.sort_values("emv_musd", ascending=False)

#         chosen_df.to_csv(out_csv, index=False, encoding="utf-8")
#         with open(out_txt, "w", encoding="utf-8") as f:
#             f.write("MODE: STOCHASTIC (max expected EMV)\n")
#             f.write(f"BUDGET (MUSD): {budget:.1f}\n")
#             f.write(f"CHOSEN BLOCKS: {len(chosen)}\n")
#             f.write(f"TOTAL COST (MUSD): {total_cost:.1f}\n")
#             f.write(f"TOTAL EXPECTED EMV (MUSD): {total_emv:.1f}\n")

#         logger.info(
#             f"Результаты стохастической оптимизации: {out_csv}, {out_txt}"
#         )
#         logger.info(
#             f"Выбрано={len(chosen)}, стоимость={total_cost:.1f}, ожидаемый EMV={total_emv:.1f}"
#         )

#     else:
#         chosen, total_cost, total_emv_low, df2 = solve_robust_knapsack(
#             df, budget, p_delta=p_delta, solver=solver
#         )
#         chosen_df = df2[df2["block_id"].isin(chosen)].copy()
#         chosen_df = chosen_df.sort_values("emv_low_musd", ascending=False)

#         chosen_df.to_csv(out_csv, index=False, encoding="utf-8")
#         with open(out_txt, "w", encoding="utf-8") as f:
#             f.write("MODE: ROBUST (max worst-case EMV)\n")
#             f.write(f"BUDGET (MUSD): {budget:.1f}\n")
#             f.write(
#                 f"P UNCERTAINTY: p_low = p*(1-delta), delta={p_delta:.2f}\n"
#             )
#             f.write(f"CHOSEN BLOCKS: {len(chosen)}\n")
#             f.write(f"TOTAL COST (MUSD): {total_cost:.1f}\n")
#             f.write(f"TOTAL WORST-CASE EMV (MUSD): {total_emv_low:.1f}\n")

#         logger.info(f"Результаты робастной оптимизации: {out_csv}, {out_txt}")
#         logger.info(
#             f"Выбрано={len(chosen)}, стоимость={total_cost:.1f}, пессимистический EMV={total_emv_low:.1f}"
#         )

#     if frontier:
#         logger.info("Расчет  Бюджет -> EMV")
#         budgets = np.arange(bmin, bmax + 1e-9, bstep)
#         emvs = []
#         costs = []
#         ns = []
#         for B in budgets:
#             if mode == "stochastic":
#                 chosen, total_cost, total_emv = solve_stochastic_knapsack(
#                     df, B, solver=solver
#                 )
#                 emvs.append(total_emv)
#                 costs.append(total_cost)
#                 ns.append(len(chosen))
#             else:
#                 chosen, total_cost, total_emv_low, _df2 = (
#                     solve_robust_knapsack(
#                         df, B, p_delta=p_delta, solver=solver
#                     )
#                 )
#                 emvs.append(total_emv_low)
#                 costs.append(total_cost)
#                 ns.append(len(chosen))

#         # Plot
#         plt.figure()
#         plt.plot(budgets, emvs, marker="o")
#         plt.xlabel("Budget (MUSD)")
#         plt.ylabel("Portfolio objective (MUSD)")
#         plt.title(f"Efficient frontier: Budget vs objective ({mode})")
#         plt.tight_layout()
#         out_fig = f"fig_frontier_{mode}.png"
#         plt.savefig(out_fig, dpi=300)
#         # plt.show()

#         # Save table
#         pd.DataFrame(
#             {
#                 "budget_musd": budgets,
#                 "objective_musd": emvs,
#                 "cost_musd": costs,
#                 "n_blocks": ns,
#             }
#         ).to_csv(f"frontier_{mode}.csv", index=False, encoding="utf-8")

#         print("Saved:", out_fig, f"frontier_{mode}.csv")

#     logger.success("Результаты экспортированы в файл")


if __name__ == "__main__":
    app()
