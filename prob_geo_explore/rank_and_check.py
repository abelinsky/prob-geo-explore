from pathlib import Path

from loguru import logger
import typer
import pandas as pd

from prob_geo_explore.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_features.csv",
        "--features",
        "-f",
        exists=True,
        help="Путь до файла с характеристиками блоков block_features.csv",
    ),
    block_probabilities_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_probabilities.csv",
        "--probs",
        "-p",
        exists=True,
        help="Путь до файла с информацией о вероятностях успехов блоков",
    ),
    out_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked.csv",
        "--out",
        "-o",
        help="Путь до файла, куда будет записан результат",
    ),
):
    logger.info("Ранжирование блоков...")

    features = pd.read_csv(features_path)
    probs = pd.read_csv(block_probabilities_path)

    df = features.merge(probs, on="block_id", how="left")
    df["p_success"] = df["p_success"].fillna(0.0)

    # Rank
    df = df.sort_values("p_success", ascending=False)

    # Quick lift check: what share of true commercial_success are in top-K?
    for k in [20, 50, 100]:
        topk = df.head(k)
        rate = (
            topk["commercial_success"].mean()
            if "commercial_success" in topk.columns
            else float("nan")
        )
        logger.info(f"Топ-{k}: уровень commercial_success = {rate:.3f}")

    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.success("Расчеты выполнены. Ранжирование:")
    logger.success(f"Результаты экспортированы в файл {out_path}")


if __name__ == "__main__":
    app()
