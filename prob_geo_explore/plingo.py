"""Вывод на основе plingo."""

import re
from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger
import typer
from prob_geo_explore.config import PROJ_ROOT, PROCESSED_DATA_DIR
import subprocess
from tqdm import tqdm
import tempfile

app = typer.Typer()


# how hard to approximate
N_MODELS = 200  # try 1000..5000
TIME_LIMIT = 10  # seconds per query
USE_BALANCED = False
BALANCED_N = 500  # up to 2*BALANCED_N models


def slug_block(x):
    s = str(x).strip().replace(" ", "").replace("/", "_").replace("-", "_")
    if not s.lower().startswith("b"):
        s = "b" + s
    return s.lower()


def generate_asp_evidence(block_features_path: Path, out_path: Path):
    df = pd.read_csv(block_features_path)
    df["block_id"] = df["block_id"].apply(slug_block)

    def num(col, default=np.nan):
        if col not in df.columns:
            return pd.Series(default, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    dist = num("dist_to_nearest_field_km", np.nan)
    wdepth = num("water_depth_p50", np.nan)
    wells = num("well_count", 0.0)
    dry = num("dry_well_count", 0.0)
    expl = num("exploration_like_count", 0.0)

    near = num("near_field", 0.0).astype(int)
    deep = num("deepwater_penalty", 0.0).astype(int)

    lines = []
    for i, r in df.iterrows():
        b = r["block_id"]
        lines.append(f"block({b}).")

        # proximity
        if not np.isnan(dist.iloc[i]):
            if dist.iloc[i] < 5:
                lines.append(f"very_close_to_field({b}).")
            if dist.iloc[i] < 20:
                lines.append(f"close_to_field({b}).")
            if dist.iloc[i] < 50:
                lines.append(f"near_to_field({b}).")

        # wells
        if wells.iloc[i] >= 5:
            lines.append(f"many_wells({b}).")
        if wells.iloc[i] < 2:
            lines.append(f"few_wells({b}).")
        if dry.iloc[i] >= 3:
            lines.append(f"many_dry_wells({b}).")

        # exploration activity
        if expl.iloc[i] >= 3:
            lines.append(f"active_area({b}).")

        # water depth
        if not np.isnan(wdepth.iloc[i]):
            if wdepth.iloc[i] < 200:
                lines.append(f"shallow_water({b}).")
            if wdepth.iloc[i] > 500:
                lines.append(f"deep_water({b}).")

        if near.iloc[i] == 1:
            lines.append(f"near_field({b}).")
        if deep.iloc[i] == 1:
            lines.append(f"deepwater_penalty({b}).")

        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(
        "Записан: ",
        out_path,
        "ичсло фактов:",
        sum(1 for ln in lines if ln.strip()),
    )


def extract_blocks(evidence_path: str):
    blocks = []
    pat = re.compile(r"^\s*block\(([^)]+)\)\s*\.\s*$", re.IGNORECASE)
    with open(evidence_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                blocks.append(m.group(1).strip().lower())
    if not blocks:
        raise RuntimeError("No block(...) facts found in evidence.lp")
    return sorted(set(blocks))


def run_one_query(
    block_id: str, model_path: Path, evidence_path: Path
) -> float:
    q = f"success({block_id})"

    cmd = [
        "plingo",
        str(model_path),
        str(evidence_path),
        "--frontend=lpmln",
        f"--query={q}",
        f"--time-limit={TIME_LIMIT}",
        "-q2",
    ]

    if USE_BALANCED:
        cmd += ["--balanced", str(BALANCED_N)]
    else:
        cmd += ["-n", str(N_MODELS)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # keep stderr for debugging
        raise RuntimeError(
            f"plingo failed for {block_id}:\n{res.stdout}\n{res.stderr}"
        )

    txt = res.stdout + "\n" + res.stderr

    # remove known warnings/errors from parsing stream
    txt2 = "\n".join(
        [
            ln
            for ln in txt.splitlines()
            if "ERROR" not in ln and "Balanced approximation" not in ln
        ]
    )

    floats = re.findall(r"([0-9]*\.[0-9]+)", txt2)
    if not floats:
        raise RuntimeError(
            f"Could not parse probability for {block_id}. Output:\n{txt[:2000]}"
        )
    return float(floats[-1])


# MODEL = "model_lpmln_noseis.lp"
# EVIDENCE_ALL = "evidence.lp"

TIME_LIMIT = 15
N_MODELS = 2000  # approximation via model enumeration

block_pat = re.compile(r"^block\(([^)]+)\)\.\s*$", re.IGNORECASE)
pred_pat = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)\.\s*$")


def load_evidence_by_block(path: str):
    by_block = {}
    blocks = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            m = block_pat.match(ln)
            if m:
                b = m.group(1).strip().lower()
                blocks.append(b)
                by_block.setdefault(b, []).append(f"block({b}).")
                continue
            m2 = pred_pat.match(ln)
            if m2:
                pred = m2.group(1)
                arg = m2.group(2).strip().lower()
                # только если это факт про конкретный block-id
                if arg in by_block:
                    by_block[arg].append(ln)
                else:
                    # факты могут идти ДО block(...). — тогда временно пропустим
                    # (или можно сделать двухпроходный парсинг)
                    pass
    # второй проход, чтобы поймать факты, которые шли до block(...)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or block_pat.match(ln):
                continue
            m2 = pred_pat.match(ln)
            if m2:
                arg = m2.group(2).strip().lower()
                if arg in by_block:
                    by_block[arg].append(ln)
    return sorted(set(blocks)), by_block


def run_plingo_for_block(
    model_path: Path, block_id: str, facts: list[str]
) -> float:
    # локальный evidence
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as tmp:
        tmp.write("\n".join(facts) + "\n")
        tmp_path = tmp.name

    cmd = [
        "plingo",
        str(model_path),
        tmp_path,
        "--frontend=lpmln",
        f"--query=success({block_id})",
        "-n",
        str(N_MODELS),
        "-q2",
        f"--time-limit={TIME_LIMIT}",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)

    # cleanup temp
    Path(tmp_path).unlink(missing_ok=True)

    txt = (res.stdout or "") + "\n" + (res.stderr or "")

    # Пытаемся распарсить вероятность ВСЕГДА (даже если returncode != 0)
    floats = re.findall(
        r"success\(" + re.escape(block_id) + r"\)\s*:\s*([0-9]*\.[0-9]+)", txt
    )
    if floats:
        return float(floats[-1])

    # Если не нашли именно success(b): p, то fallback на последнее float (на всякий)
    floats2 = re.findall(r"([0-9]*\.[0-9]+)", txt)
    if floats2:
        # только если код возврата "штатный" для clingo
        if res.returncode in (0, 10, 20, 30):
            return float(floats2[-1])

    # Если ничего не распарсилось — тогда это реально ошибка
    raise RuntimeError(
        f"plingo produced no probability for {block_id} (code {res.returncode}).\n{txt[:2000]}"
    )

    # if res.returncode != 0:
    #     raise RuntimeError(
    #         f"plingo failed for {block_id}:\n{res.stdout}\n{res.stderr}"
    #     )

    # txt = res.stdout + "\n" + res.stderr

    # # ВЫТАЩИМ ПОСЛЕДНЕЕ float-число из вывода (обычно это и есть вероятность)
    # floats = re.findall(r"([0-9]*\.[0-9]+)", txt)
    # if not floats:
    #     raise RuntimeError(
    #         f"Could not parse probability for {block_id}. Output:\n{txt[:2000]}"
    #     )
    # return float(floats[-1])


@app.command()
def main(
    block_features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_features.csv",
        "--features",
        "-f",
        exists=True,
        help="Путь до файла block_features.csv",
    ),
    model_lpmln_path: Path = typer.Option(
        PROJ_ROOT / "ontology/model_lpmln.lp",
        "--model_lpmln",
        "-ml",
        exists=True,
        help="Путь до файла model_lpmln.lp",
    ),
    out_evidence_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "evidence.lp",
        "--out_evidence",
        "-oe",
        help="Путь до файла, куда будет записан evidence.lp",
    ),
    out_block_ranked_plingo: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_plingo.csv",
        "--out_plingo",
        "-op",
        help="Путь до файла, куда будет записан block_ranked_plingo.csv",
    ),
):
    logger.info("Подготовка модели plingo")
    generate_asp_evidence(block_features_path, out_evidence_path)
    logger.info(f"Сгенерирован файл с данными plingo {out_evidence_path}")
    logger.info("Вывод plingo...")

    # Запуск plingo
    blocks, by_block = load_evidence_by_block(str(out_evidence_path))
    logger.info(f"Всего блоков: {len(blocks)}")

    rows = []
    for i, b in tqdm(enumerate(blocks, 1), desc="Анализ блоков"):
        try:
            p = run_plingo_for_block(model_lpmln_path, b, by_block[b])
        except Exception as e:
            logger.error(f"[{i}/{len(blocks)}] {b}: ERROR {e}")
            p = float("nan")
        rows.append((b, p))
        if i % 50 == 0:
            logger.info(f"Выполнено {i}/{len(blocks)}")

        # break

    df = pd.DataFrame(rows, columns=["block_id", "p_success_plingo"])
    df = df.dropna().sort_values("p_success_plingo", ascending=False)
    df.to_csv(out_block_ranked_plingo, index=False, encoding="utf-8")
    logger.info(f"Число записей: {len(df)}")

    # blocks = extract_blocks(str(out_evidence_path))
    # logger.info(f"Число блоков: {len(blocks)}")

    # rows = []

    # for i, b in tqdm(enumerate(blocks, 1), desc="Анализ блоков"):

    #     # for i, b in enumerate(blocks, 1):
    #     try:
    #         p = run_one_query(b, model_lpmln_path, out_evidence_path)
    #     except Exception as e:
    #         print(f"[{i}/{len(blocks)}] {b}: ERROR {e}")
    #         p = float("nan")
    #     rows.append((b, p))
    #     if i % 50 == 0:
    #         print(f"Done {i}/{len(blocks)}")

    #     break

    # df = pd.DataFrame(rows, columns=["block_id", "p_success_plingo"])
    # df = df.dropna().sort_values("p_success_plingo", ascending=False)
    # df.to_csv(out_block_ranked_plingo, index=False, encoding="utf-8")
    # logger.info(f"Запись {out_block_ranked_plingo}")
    # logger.info(f"Число записей: {len(df)}")

    logger.success("Вывод plingo выполнен успешно!")


if __name__ == "__main__":
    app()
