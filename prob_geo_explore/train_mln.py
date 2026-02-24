"""Обучение модели Markov Logic Network."""

from loguru import logger
import typer
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


def slug(x):
    s = str(x).strip().replace(" ", "").replace("/", "_").replace("-", "_")
    if not s.lower().startswith("b"):
        s = "b" + s
    return s.lower()


def generate_mln_features(ranked_path: Path, out_path: Path):
    """Формирует признаки для обучения MLN-модели."""
    df = pd.read_csv(ranked_path)

    df["block_id"] = df["block_id"].apply(slug)

    def num(col, default=np.nan):
        return (
            pd.to_numeric(df[col], errors="coerce").fillna(default)
            if col in df.columns
            else pd.Series(default, index=df.index)
        )

    dist = num("dist_to_nearest_field_km", np.nan)
    wdepth = num("water_depth_p50", np.nan)
    wells = num("well_count", 0.0)
    dry = num("dry_well_count", 0.0)
    expl = num("exploration_like_count", 0.0)
    near = num("near_field", 0.0).astype(int)
    deep = num("deepwater_penalty", 0.0).astype(int)

    feat = pd.DataFrame({"block_id": df["block_id"]})
    feat["VeryClose"] = ((dist.notna()) & (dist < 5)).astype(int)
    feat["Close"] = ((dist.notna()) & (dist < 20)).astype(int)
    feat["Near"] = ((dist.notna()) & (dist < 50)).astype(int)

    feat["ManyWells"] = (wells >= 5).astype(int)
    feat["FewWells"] = (wells < 2).astype(int)
    feat["ManyDryWells"] = (dry >= 3).astype(int)

    feat["ActiveArea"] = (expl >= 3).astype(int)
    feat["ShallowWater"] = ((wdepth.notna()) & (wdepth < 200)).astype(int)
    feat["DeepWater"] = ((wdepth.notna()) & (wdepth > 500)).astype(int)

    feat["NearField"] = (near == 1).astype(int)
    feat["DeepwaterPenalty"] = (deep == 1).astype(int)

    # interactions (MLN-like conjunctions)
    feat["Close_Active"] = (feat["Close"] & feat["ActiveArea"]).astype(int)
    feat["Close_ManyWells"] = (feat["Close"] & feat["ManyWells"]).astype(int)
    feat["FrontierGood"] = (
        feat["FewWells"] & (1 - feat["ManyDryWells"])
    ).astype(int)

    if "commercial_success" in df.columns:
        feat["y_train"] = (
            pd.to_numeric(df["commercial_success"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    feat.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Columns:", list(feat.columns))


@app.command()
def main(
    ranked_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked.csv",
        "--ranked",
        "-r",
        exists=True,
        help="Путь до файла block_ranked.csv (block_id, p_success)",
    ),
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "mln_features.csv",
        "--out",
        "-o",
        help="Путь до файла, куда будут записаны фичи для модели",
    ),
    mln_weights_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "mln_weights.csv",
        "--out_mln_weights",
        "-om",
        help="Путь до файла, куда будут записаны обученные веса модели",
    ),
    block_ranked_mln_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_mln.csv",
        "--out_block_ranked_mln",
        "-ob",
        help="Путь до файла, куда будут записаны ранжированные блоки",
    ),
):
    logger.info("Обучение Markov Logic Network...")
    logger.info("Формирование атрибутов для обучения Markov Logic Network...")
    generate_mln_features(ranked_path, features_path)
    logger.info(f"Результаты получены и записаны в {features_path}")

    # Обучение MLN
    logger.info("Обучение MLN через логистическую регресию...")

    feat = pd.read_csv(features_path)
    base = pd.read_csv(ranked_path)[["block_id", "p_success"]]

    if "y_train" not in feat.columns:
        raise ValueError("Отсутствует y_train in {features_path}.")

    # Feature set (логические предикаты + взаимодействия)
    feature_cols = [
        c for c in feat.columns if c not in ["block_id", "y_train"]
    ]
    X = feat[feature_cols].to_numpy()
    y = feat["y_train"].to_numpy()

    # Class imbalance handled by class_weight
    clf = LogisticRegression(
        C=0.01,
        penalty="l2",
        max_iter=5000,
        # class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X, y)

    p_mln = clf.predict_proba(X)[:, 1]

    out = feat[["block_id"]].copy()
    out["p_success_mln"] = p_mln

    # Merge with ProbLog probabilities for comparison
    out = out.merge(base, on="block_id", how="left")

    # AUC on training label (это baseline sanity; main validity у тебя post-T0)
    try:
        auc = roc_auc_score(y, p_mln)
        logger.info(f"Train-label ROC-AUC (y_train): {auc:.4f}")
    except Exception:
        pass

    # Save weights for monograph (это очень ценная таблица!)
    w = pd.DataFrame(
        {"feature": feature_cols, "weight": clf.coef_[0]}
    ).sort_values("weight", ascending=False)

    w.to_csv(mln_weights_path, index=False, encoding="utf-8")
    out.to_csv(block_ranked_mln_path, index=False, encoding="utf-8")

    logger.info(w.head(12).to_string(index=False))
    logger.success(
        f"Выполнено обучение MLN (см. {mln_weights_path} и {block_ranked_mln_path})"
    )


if __name__ == "__main__":
    app()
