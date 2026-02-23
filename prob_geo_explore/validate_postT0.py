#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-sample validation (post-T0 discoveries) for block ranking.

Assumptions:
- You built block_ranked.csv using ETL with --t0 2010-01-01 (i.e., features/evidence limited to pre-2010).
- You have NPD/SODIR exports:
    discovery.csv
    field.csv (optional, for "commercial" definition)
    wellbore_all_long.csv (for mapping discovery well -> coordinates -> block via spatial join)
    blkArea.zip (block polygons)

Goal:
- Create a "future truth" label y_post2010(block)=1 if the block has a discovery AFTER T0
  (optionally, only those linked to a Field -> "commercial").
- Compare y_post2010 with your predicted p_success in block_ranked.csv.

Outputs:
- block_ranked_postT0_eval.csv (merged table)
- Prints Top-K hit rates and AUC (if sklearn available)
- Saves a couple of validation plots (matplotlib, no colors specified)
"""

from loguru import logger
import typer

import re
from pathlib import Path
import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt

from prob_geo_explore.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

WGS84 = 4326


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def pick_col(df: pd.DataFrame, candidates, required=True):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # contains fallback
    for cand in candidates:
        key = _norm(cand)
        for nk, orig in norm_map.items():
            if key and key in nk:
                return orig
    if required:
        raise KeyError(
            f"Missing columns. Tried {candidates}. Available: {list(df.columns)[:40]}..."
        )
    return None


def slug_block_name(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().replace(" ", "")
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"__+", "_", s)
    if not s:
        return ""
    if not s.lower().startswith("b"):
        s = "b" + s
    return s.lower()


def read_geo(path: str) -> gpd.GeoDataFrame:
    p = Path(path)
    if p.suffix.lower() == ".zip":
        gdf = gpd.read_file(f"zip://{p.as_posix()}")
    else:
        gdf = gpd.read_file(p.as_posix())
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=WGS84)
    if gdf.crs.to_epsg() != WGS84:
        gdf = gdf.to_crs(epsg=WGS84)
    return gdf


def load_blocks(blocks_path: str) -> gpd.GeoDataFrame:
    blocks = read_geo(blocks_path)
    # find block name col
    name_col = None
    for cand in [
        "blkName",
        "blockName",
        "block",
        "name",
        "BLK_NAME",
        "BLOCK",
        "BLOCKNAME",
        "prlAreaPolyBlockName",
    ]:
        c = pick_col(blocks, [cand], required=False)
        if c:
            name_col = c
            break
    if name_col is None:
        # fallback first object col
        obj_cols = [
            c
            for c in blocks.columns
            if blocks[c].dtype == "object" and c.lower() != "geometry"
        ]
        if not obj_cols:
            raise KeyError("Cannot find block name column in blocks layer.")
        name_col = obj_cols[0]

    blocks = blocks.rename(columns={name_col: "block_name_raw"}).copy()
    blocks["block_id"] = blocks["block_name_raw"].apply(slug_block_name)
    blocks = blocks[blocks["block_id"].astype(bool)].copy()
    return blocks[["block_id", "geometry"]]


def load_wells(wellbore_csv: str) -> gpd.GeoDataFrame:
    wells = pd.read_csv(wellbore_csv, low_memory=False)
    lon_col = pick_col(
        wells, ["wlbEwDecDeg", "longitude", "lon", "x"], required=True
    )
    lat_col = pick_col(
        wells, ["wlbNsDecDeg", "latitude", "lat", "y"], required=True
    )
    npdid_col = pick_col(
        wells,
        ["wlbNpdidWellbore", "npdidWellbore", "NPDIDWELLBORE"],
        required=True,
    )

    geometry = [Point(xy) for xy in zip(wells[lon_col], wells[lat_col])]
    gdf = gpd.GeoDataFrame(wells, geometry=geometry, crs=f"EPSG:{WGS84}")
    gdf["_npdid_wellbore"] = pd.to_numeric(gdf[npdid_col], errors="coerce")
    gdf = gdf.dropna(subset=["_npdid_wellbore"]).copy()
    gdf["_npdid_wellbore"] = gdf["_npdid_wellbore"].astype(int)
    return gdf[["_npdid_wellbore", "geometry"]]


def spatial_join_points_to_blocks(
    points_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    joined = gpd.sjoin(points_gdf, blocks_gdf, how="left", predicate="within")
    miss = joined["block_id"].isna()
    if miss.any():
        j2 = gpd.sjoin(
            points_gdf[miss], blocks_gdf, how="left", predicate="intersects"
        )
        joined.loc[miss, "block_id"] = j2["block_id"].values
    return pd.DataFrame(
        joined.drop(columns=["geometry", "index_right"], errors="ignore")
    )


def build_postT0_truth(
    discovery_csv: str,
    wells_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    t0_year: int,
    commercial_only: bool,
    field_csv: str | None,
) -> pd.DataFrame:
    disc = pd.read_csv(discovery_csv, low_memory=False)

    year_col = pick_col(
        disc, ["dscDiscoveryYear", "discoveryYear", "year"], required=True
    )
    wlb_id_col = pick_col(
        disc,
        ["wlbNpdidWellbore", "wlbNpdidWellboreReclass", "npdidWellbore"],
        required=True,
    )

    disc["_year"] = pd.to_numeric(disc[year_col], errors="coerce")
    disc["_disc_well_id"] = pd.to_numeric(disc[wlb_id_col], errors="coerce")
    disc = disc.dropna(subset=["_year", "_disc_well_id"]).copy()
    disc["_year"] = disc["_year"].astype(int)
    disc["_disc_well_id"] = disc["_disc_well_id"].astype(int)

    # Filter to post-T0
    disc_post = disc[disc["_year"] > t0_year].copy()

    if commercial_only:
        if not field_csv:
            raise ValueError("commercial_only=True requires --field field.csv")
        fld_id_col = pick_col(
            disc_post,
            ["fldNpdidField", "fldNpdidField", "fldNpdidField"],
            required=True,
        )
        disc_post["_fld_id"] = pd.to_numeric(
            disc_post[fld_id_col], errors="coerce"
        )
        disc_post = disc_post[disc_post["_fld_id"].notna()].copy()

    # Map discovery well -> point geometry
    disc_pts = disc_post.merge(
        wells_gdf,
        left_on="_disc_well_id",
        right_on="_npdid_wellbore",
        how="left",
    )
    disc_pts = disc_pts.dropna(subset=["geometry"]).copy()
    disc_pts_gdf = gpd.GeoDataFrame(
        disc_pts, geometry="geometry", crs=f"EPSG:{WGS84}"
    )

    # Assign to blocks
    disc_in_blocks = spatial_join_points_to_blocks(disc_pts_gdf, blocks_gdf)
    disc_in_blocks = disc_in_blocks.dropna(subset=["block_id"]).copy()

    # Block-level truth
    truth = (
        disc_in_blocks.groupby("block_id")
        .size()
        .reset_index(name="n_discoveries_postT0")
    )
    truth["y_postT0"] = 1
    return truth


def auc_if_possible(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


@app.command()
def main(
    ranked_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked.csv",
        "--ranked",
        "-r",
        exists=True,
        help="Путь до файла block_ranked.csv (block_id, p_success)",
    ),
    discovery_path: Path = typer.Option(
        RAW_DATA_DIR / "discovery.csv",
        "--discovery",
        "-d",
        exists=True,
        help="Путь до файла discovery.csv",
    ),
    wellbore_path: Path = typer.Option(
        RAW_DATA_DIR / "wellbore_all_long.csv",
        "--wellbore",
        "-w",
        exists=True,
        help="Путь до файла wellbore_all_long.csv",
    ),
    blocks_path: Path = typer.Option(
        RAW_DATA_DIR / "blkArea (1).zip",
        "--blocks",
        "-b",
        exists=True,
        help="Путь до файла blkArea (1).zip",
    ),
    t0: str = typer.Option(
        "2010-01-01",
        "--cutoff",
        "-t0",
        help="Временная граница для отсекания данных (YYYY-MM-DD)",
    ),
    commercial_only: bool = typer.Option(
        False,
        "--commercial_only",
        "-co",
        help="Использовать только открытия, ассоциированные с месторождением",
    ),
    field: str = typer.Option(
        RAW_DATA_DIR / "field.csv",
        "--field",
        "-f",
        help="field.csv, только при установленном commercial_only",
    ),
    out: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_ranked_postT0_eval.csv",
        "--out",
        "-o",
        help="Путь до файла, куда будет записан результат",
    ),
):
    logger.info("Валидация модели...")

    t0_year = pd.to_datetime(t0).year

    ranked = pd.read_csv(ranked_path)
    if "block_id" not in ranked.columns or "p_success" not in ranked.columns:
        raise ValueError(
            "ranked file must contain block_id and p_success columns"
        )

    blocks = load_blocks(blocks_path)
    wells = load_wells(wellbore_path)

    truth = build_postT0_truth(
        discovery_csv=discovery_path,
        wells_gdf=wells,
        blocks_gdf=blocks,
        t0_year=t0_year,
        commercial_only=commercial_only,
        field_csv=field,
    )

    merged = ranked.merge(
        truth[["block_id", "y_postT0", "n_discoveries_postT0"]],
        on="block_id",
        how="left",
    )
    merged["y_postT0"] = merged["y_postT0"].fillna(0).astype(int)
    merged["n_discoveries_postT0"] = (
        merged["n_discoveries_postT0"].fillna(0).astype(int)
    )

    # Metrics
    y = merged["y_postT0"].to_numpy()
    p = merged["p_success"].to_numpy()

    base_rate = float(y.mean())
    auc = auc_if_possible(y, p)

    print(f"T0 year: {t0_year}")
    print(f"Blocks: {len(merged)}")
    print(f"Post-T0 discovery blocks rate (base): {base_rate:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC: (install scikit-learn to compute)")

    for k in [20, 50, 100, 200, 300]:
        if k <= len(merged):
            topk_rate = float(
                merged.sort_values("p_success", ascending=False)
                .head(k)["y_postT0"]
                .mean()
            )
            lift = (topk_rate / base_rate) if base_rate > 0 else np.nan
            print(f"Top-{k}: postT0 hit rate={topk_rate:.4f}, lift={lift:.2f}")

    # Save merged
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"OK: wrote {out}")

    # Plots (publication-friendly)
    merged_sorted = merged.sort_values(
        "p_success", ascending=False
    ).reset_index(drop=True)

    # 1) Lift curve on post-T0 truth
    merged_sorted["cum_hits"] = merged_sorted["y_postT0"].cumsum()
    merged_sorted["cum_rate"] = merged_sorted["cum_hits"] / (
        merged_sorted.index + 1
    )

    plt.figure()
    plt.plot(merged_sorted.index + 1, merged_sorted["cum_rate"])
    plt.xlabel("Топ-K блоков (прогнозная вероятность)")
    plt.ylabel("Кумулятивный уровень post-T0 обнаружения")
    plt.title("Out-of-sample Lift Curve (post-T0 discoveries)")
    plt.tight_layout()
    plt.savefig("fig_postT0_lift_curve.png", dpi=300)
    plt.show()

    # 2) Calibration-style plot (binning)
    bins = 10
    merged_sorted["bin"] = pd.qcut(
        merged_sorted["p_success"], q=bins, duplicates="drop"
    )
    calib = (
        merged_sorted.groupby("bin")
        .agg(
            p_mean=("p_success", "mean"),
            y_rate=("y_postT0", "mean"),
            n=("y_postT0", "size"),
        )
        .reset_index()
    )

    plt.figure()
    plt.plot(calib["p_mean"], calib["y_rate"], marker="o")
    plt.xlabel("Средняя прогнозируемая вероятность")
    plt.ylabel("Набладаемая post-T0 уровнь")
    plt.title("Reliability Diagram (post-T0)")
    plt.tight_layout()
    plt.savefig("fig_postT0_reliability.png", dpi=300)
    plt.show()

    print(
        "Saved figures: fig_postT0_lift_curve.png, fig_postT0_reliability.png"
    )


if __name__ == "__main__":
    app()
