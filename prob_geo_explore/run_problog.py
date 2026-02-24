from pathlib import Path

from loguru import logger
import typer
import re
import pandas as pd
from problog.program import PrologFile
from problog import get_evaluatable

from prob_geo_explore.config import PROJ_ROOT, PROCESSED_DATA_DIR

app = typer.Typer()

BLOCK_ID_RE = re.compile(r"^block\(([^)]+)\)\.\s*$")


def extract_blocks_from_facts(path: str):
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = BLOCK_ID_RE.match(line.strip())
            if m:
                blocks.append(m.group(1))
    return sorted(set(blocks))


def parse_results_to_df(result_dict):
    rows = []
    for q, p in result_dict.items():
        s = str(q)
        m = re.match(r"commercial_success_pred\(([^)]+)\)", s)
        if not m:
            continue
        block_id = m.group(1)
        rows.append({"block_id": block_id, "p_success": float(p)})
    return pd.DataFrame(rows).sort_values("p_success", ascending=False)


@app.command()
def main(
    facts_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "facts_blocks.pl",
        "--facts",
        "-f",
        exists=True,
        help="Путь до файла с ABox (онтологическими фактами)",
    ),
    derived_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "derived_facts_noseis.pl",
        "--derived",
        "-d",
        exists=True,
        help="Путь до файла с геологическими правилами",
    ),
    model_path: Path = typer.Option(
        PROJ_ROOT / "ontology/model_problog.pl",
        "--model",
        "-m",
        exists=True,
        help="Путь до файла с моделью Problog",
    ),
    out_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_probabilities.csv",
        "--out",
        "-o",
        help="Путь до файла, куда будет записан результат",
    ),
    use_previous: bool = typer.Option(
        True,
        "--use_previous",
        "-u",
        help="Использовать ранее полученный результат (файл `out_path`)",
    ),
):
    if use_previous and out_path.exists():
        logger.success(
            f"Использую ранее выведенный результат Problog: {out_path}..."
        )
        return

    logger.info("Запуск problog...")

    blocks = extract_blocks_from_facts(str(facts_path))
    if not blocks:
        raise RuntimeError("No block(...) facts found in facts_blocks.pl")

    # Конкатенация файлов
    # (Problog может загружать только один файл)
    combined_path = "_combined_problog.pl"
    with open(combined_path, "w", encoding="utf-8") as w:
        for path in [facts_path, derived_path, model_path]:
            with open(path, "r", encoding="utf-8") as r:
                w.write(f"\n% ===== {path} =====\n")
                w.write(r.read())
                w.write("\n")

        # Add GROUNDED queries for each block
        w.write("\n% ===== grounded queries =====\n")
        for b in blocks:
            w.write(f"query(commercial_success_pred({b})).\n")

    logger.info(f"Подготовлен Problog-файл: {combined_path}...")
    logger.info("Вывод Problog...")

    model = PrologFile(combined_path)
    result = get_evaluatable().create_from(model).evaluate()

    logger.info("Расчет в problog выполнен...")
    df = parse_results_to_df(result)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.success("Вычислены вероятности `commercial_success_pred(B)`.")
    logger.success(f"Записан файл {out_path} с {len(df)} строками")
    logger.success(df.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
