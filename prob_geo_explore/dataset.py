"""
ETL: Открытые данные Норвежского шельфа (SODIR/NPD) -> Таблица с фичами по блокам.

Результат:
  - block_features.csv  (одна строка на блок)
Опционально:
  - facts_blocks.pl     (Prolog/ProbLog-факты из фичей)

Исходные файлы (минимум):
  - wellbore_all_long.csv
  - discovery.csv
  - field.csv
  - blkArea.zip  (или директория с shapefile)
Опциональные исходные файлы:
  - strat_litho_wellbore.csv
  - strat_litho_wellbore_core.csv
  - geochem file (csv/tsv)
  - сейммические полигоны (GeoJSON/Shapefile) с завершенными исследованиями

Использование:
  python etl_npd_blocks.py \
    --wellbore data/wellbore_all_long.csv \
    --discovery data/discovery.csv \
    --field data/field.csv \
    --blocks data/blkArea.zip \
    --seismic data/seismic_finished.geojson \
    --strat data/strat_litho_wellbore.csv \
    --geochem data/geochem.csv \
    --out block_features.csv \
    --facts facts_blocks.pl

"""

import sys
from typing import Iterable, Optional
from pathlib import Path

import re

from loguru import logger
import typer

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

from prob_geo_explore.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Equal-area projection for Europe (good for area ratios)
AREA_EPSG = 3035  # ETRS89 / LAEA Europe
WGS84_EPSG = 4326

# -----------------------------
# Helpers
# -----------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def pick_col(
    df: pd.DataFrame, candidates: Iterable[str], required: bool = True
) -> Optional[str]:
    """Pick first existing column from candidates (case-insensitive by normalized name)."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # fuzzy: contains
    for cand in candidates:
        key = _norm(cand)
        for nk, orig in norm_map.items():
            if key and key in nk:
                return orig
    if required:
        raise KeyError(
            f"None of the candidate columns exist: {list(candidates)}. Available: {list(df.columns)[:30]}..."
        )
    return None


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # try separator sniff for csv/tsv
    if p.suffix.lower() in [".tsv", ".tab"]:
        return pd.read_csv(p, sep="\t", low_memory=False)
    if p.suffix.lower() in [".csv"]:
        return pd.read_csv(p, low_memory=False)
    # fallback
    return pd.read_csv(p, low_memory=False)


def read_geo(path: str) -> gpd.GeoDataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Support zip shapefile: "blkArea.zip"
    if p.suffix.lower() == ".zip":
        # geopandas/fiona can read a zipped shapefile directly
        # If zip contains multiple layers, fiona will pick first; usually fine for blkArea
        gdf = gpd.read_file(f"zip://{p.as_posix()}")
    else:
        gdf = gpd.read_file(p.as_posix())
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=WGS84_EPSG)
    return gdf


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=WGS84_EPSG)
    if gdf.crs.to_epsg() != WGS84_EPSG:
        return gdf.to_crs(epsg=WGS84_EPSG)
    return gdf


def to_area_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=WGS84_EPSG)
    if gdf.crs.to_epsg() != AREA_EPSG:
        return gdf.to_crs(epsg=AREA_EPSG)
    return gdf


def slug_block_name(x: str) -> str:
    # "25/6" -> "b25_6"
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = s.replace(" ", "")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = re.sub(r"__+", "_", s)
    if not s:
        return ""
    if not s.lower().startswith("b"):
        s = "b" + s
    return s.lower()


def slug(x):
    s = str(x).strip().replace(" ", "").replace("/", "_").replace("-", "_")
    if not s.lower().startswith("b"):
        s = "b" + s
    return s.lower()


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Vectorized haversine distance in km between (lon1,lat1) and (lon2,lat2)."""
    R = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# -----------------------------
# Main ETL steps
# -----------------------------


def load_blocks(blocks_path: str) -> gpd.GeoDataFrame:
    blocks = read_geo(blocks_path)
    blocks = ensure_wgs84(blocks)
    # Try to find a block name column
    # Common candidates: "blkName", "block", "BlockName", "BLK_NAME", etc.
    name_col = None
    for cand in [
        "blkName",
        "blockName",
        "block",
        "name",
        "BLK_NAME",
        "BLOCK",
        "BLOCKNAME",
        "blk_name",
    ]:
        if any(_norm(cand) == _norm(c) for c in blocks.columns):
            name_col = pick_col(blocks, [cand], required=False)
            break
    if name_col is None:
        # take first string-ish column if present
        str_cols = [
            c
            for c in blocks.columns
            if blocks[c].dtype == "object" and c.lower() not in ["geometry"]
        ]
        if str_cols:
            name_col = str_cols[0]
        else:
            raise KeyError(
                "Could not find a block name column in blocks layer."
            )
    blocks = blocks.rename(columns={name_col: "block_name_raw"})
    blocks["block_id"] = blocks["block_name_raw"].apply(slug_block_name)
    blocks = blocks[blocks["block_id"].astype(bool)].copy()

    # Region/quadrant may exist as attribute; if absent we keep blank
    region_col = None
    for cand in [
        "mainArea",
        "region",
        "blkMainArea",
        "blkMainAreaName",
        "area",
        "AREA",
    ]:
        region_col = pick_col(blocks, [cand], required=False)
        if region_col:
            break
    if region_col and region_col != "block_name_raw":
        blocks = blocks.rename(columns={region_col: "region"})
    else:
        blocks["region"] = ""

    # Compute block centroid in WGS84 for distances
    blocks_wgs = blocks.copy()
    blocks_wgs["centroid"] = blocks_wgs.geometry.centroid
    blocks_wgs["centroid_lon"] = blocks_wgs["centroid"].x
    blocks_wgs["centroid_lat"] = blocks_wgs["centroid"].y

    # Area in km^2
    blocks_area = to_area_crs(blocks)
    blocks_wgs["block_area_km2"] = (blocks_area.geometry.area / 1e6).values

    return blocks_wgs[
        [
            "block_id",
            "block_name_raw",
            "region",
            "geometry",
            "centroid_lon",
            "centroid_lat",
            "block_area_km2",
        ]
    ]


def load_wellbores(
    wellbore_csv: str, t0: Optional[str] = None
) -> gpd.GeoDataFrame:
    wells = read_table(wellbore_csv)

    lon_col = pick_col(
        wells, ["wlbEwDecDeg", "longitude", "lon", "x"], required=True
    )
    lat_col = pick_col(
        wells, ["wlbNsDecDeg", "latitude", "lat", "y"], required=True
    )
    content_col = pick_col(
        wells, ["wlbContent", "content", "result"], required=False
    )
    type_col = pick_col(wells, ["wlbWellType", "wellType"], required=False)
    purpose_col = pick_col(wells, ["wlbPurpose", "purpose"], required=False)
    water_col = pick_col(
        wells, ["wlbWaterDepth", "waterDepth"], required=False
    )
    completion_col = pick_col(
        wells, ["wlbCompletionDate", "completionDate"], required=False
    )
    npdid_well_col = pick_col(
        wells,
        ["wlbNpdidWellbore", "npdidWellbore", "NPDIDWELLBORE"],
        required=False,
    )
    wlb_name_col = pick_col(
        wells, ["wlbWellboreName", "wellboreName", "wlbName"], required=False
    )

    # Parse time filter if provided
    if t0 and completion_col:
        wells["_completion_dt"] = safe_to_datetime(wells[completion_col])
        t0_dt = pd.to_datetime(t0)
        wells = wells[
            (wells["_completion_dt"].isna())
            | (wells["_completion_dt"] <= t0_dt)
        ].copy()

    # Build GeoDataFrame
    geometry = [Point(xy) for xy in zip(wells[lon_col], wells[lat_col])]
    wells_gdf = gpd.GeoDataFrame(
        wells, geometry=geometry, crs=f"EPSG:{WGS84_EPSG}"
    )

    # Normalize a few helper columns
    wells_gdf["well_content"] = (
        wells_gdf[content_col].astype(str) if content_col else ""
    )
    wells_gdf["well_type"] = (
        wells_gdf[type_col].astype(str) if type_col else ""
    )
    wells_gdf["well_purpose"] = (
        wells_gdf[purpose_col].astype(str) if purpose_col else ""
    )
    wells_gdf["water_depth"] = (
        pd.to_numeric(wells_gdf[water_col], errors="coerce")
        if water_col
        else np.nan
    )
    wells_gdf["npdid_wellbore"] = (
        wells_gdf[npdid_well_col] if npdid_well_col else np.nan
    )
    wells_gdf["wellbore_name"] = (
        wells_gdf[wlb_name_col] if wlb_name_col else ""
    )

    # Determine dry wells (conservative)
    # NPD uses values like "OIL", "GAS", "DRY", etc., depending on table version.
    content = wells_gdf["well_content"].str.lower()
    wells_gdf["is_dry"] = content.str.contains("dry", na=False)

    # Determine exploration/appraisal vs development (optional)
    wt = (wells_gdf["well_type"] + " " + wells_gdf["well_purpose"]).str.lower()
    wells_gdf["is_exploration_like"] = (
        wt.str.contains("expl", na=False)
        | wt.str.contains("wildcat", na=False)
        | wt.str.contains("apprais", na=False)
    )

    return wells_gdf


def spatial_join_wells_to_blocks(
    wells_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    # Ensure same CRS
    wells = ensure_wgs84(wells_gdf)
    blocks = ensure_wgs84(blocks_gdf)

    # Use within; if some points lie on edges, consider intersects
    joined = gpd.sjoin(
        wells, blocks[["block_id", "geometry"]], how="left", predicate="within"
    )
    # fallback for missing joins: try intersects for those with no block_id
    missing = joined["block_id"].isna()
    if missing.any():
        j2 = gpd.sjoin(
            wells[missing],
            blocks[["block_id", "geometry"]],
            how="left",
            predicate="intersects",
        )
        joined.loc[missing, "block_id"] = j2["block_id"].values

    # Convert to plain DataFrame
    df = pd.DataFrame(
        joined.drop(columns=["geometry", "index_right"], errors="ignore")
    )
    return df


def aggregate_wells_by_block(wells_blocks: pd.DataFrame) -> pd.DataFrame:
    # If no block_id for some wells, drop them
    wb = wells_blocks.dropna(subset=["block_id"]).copy()
    wb["block_id"] = wb["block_id"].astype(str)

    agg = (
        wb.groupby("block_id")
        .agg(
            well_count=("block_id", "size"),
            dry_well_count=("is_dry", "sum"),
            water_depth_avg=("water_depth", "mean"),
            water_depth_p50=("water_depth", "median"),
            exploration_like_count=("is_exploration_like", "sum"),
        )
        .reset_index()
    )
    return agg


def load_discovery(
    discovery_csv: str, t0: Optional[str] = None
) -> pd.DataFrame:
    d = read_table(discovery_csv)
    # common columns from your earlier message:
    # dscName, dscHcType, wlbNpdidWellbore, fldNpdidField, dscDiscoveryYear, dscNpdidDiscovery
    dsc_year_col = pick_col(
        d, ["dscDiscoveryYear", "discoveryYear", "year"], required=False
    )
    if t0 and dsc_year_col:
        t0_year = pd.to_datetime(t0).year
        d = d[
            pd.to_numeric(d[dsc_year_col], errors="coerce") <= t0_year
        ].copy()
    return d


def load_field(field_csv: str, t0: Optional[str] = None) -> pd.DataFrame:
    f = read_table(field_csv)
    # Filter by completion year if present (optional)
    # Field table may have wlbCompletionDate (from discovery well) or none
    comp_col = pick_col(
        f,
        ["wlbCompletionDate", "fldDateUpdated", "completionDate"],
        required=False,
    )
    if t0 and comp_col:
        dt = safe_to_datetime(f[comp_col])
        n0 = f.size
        f = f[(dt.isna()) | (dt <= pd.to_datetime(t0))].copy()
        logger.info(
            f"Месторождения отфильтрованы по дате ({t0}), {n0} ед. -> {f.size} ед."
        )
    return f


def derive_commercial_success_label(
    blocks: gpd.GeoDataFrame,
    wells_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    field_df: pd.DataFrame,
    t0: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create block-level label commercial_success (0/1) using open NPD tables.
    Conservative rule:
      commercial_success(block)=1 if there exists a Discovery located in this block AND that discovery is linked to a Field.
    We locate discovery via discovery wellbore (wlbNpdidWellbore) joined to wells_df.
    """
    d = discovery_df.copy()
    # identify link columns
    disc_well_id_col = pick_col(
        d,
        ["wlbNpdidWellbore", "wlbNpdidWellbore", "wlbNpdidWellboreReclass"],
        required=False,
    )
    fld_id_col = pick_col(
        d, ["fldNpdidField", "fldNpdidField", "fieldId"], required=False
    )

    if disc_well_id_col is None or fld_id_col is None:
        # fallback: if discovery already has block name, we could use it, but usually it doesn't.
        # If missing, we cannot label robustly.
        print(
            "WARN: discovery table missing wlbNpdidWellbore and/or fldNpdidField. commercial_success will be NaN.",
            file=sys.stderr,
        )
        out = blocks[["block_id"]].copy()
        out["commercial_success"] = np.nan
        out["has_discovery_linked_to_field"] = np.nan
        return out

    # Prepare wells mapping: npdid_wellbore -> block_id
    wells_map = wells_df.dropna(subset=["npdid_wellbore", "block_id"]).copy()
    wells_map["npdid_wellbore"] = pd.to_numeric(
        wells_map["npdid_wellbore"], errors="coerce"
    )
    wells_map = wells_map.dropna(subset=["npdid_wellbore"])
    wells_map = wells_map[["npdid_wellbore", "block_id"]].drop_duplicates()

    d["_disc_well_id"] = pd.to_numeric(d[disc_well_id_col], errors="coerce")
    d["_fld_id"] = pd.to_numeric(d[fld_id_col], errors="coerce")

    d = d.merge(
        wells_map,
        left_on="_disc_well_id",
        right_on="npdid_wellbore",
        how="left",
    )
    # A discovery is "commercial-linked" if it has a non-null field id
    d["disc_linked_to_field"] = d["_fld_id"].notna()

    label = d.groupby("block_id")["disc_linked_to_field"].max().reset_index()
    label = label.rename(
        columns={"disc_linked_to_field": "has_discovery_linked_to_field"}
    )
    label["commercial_success"] = label[
        "has_discovery_linked_to_field"
    ].astype(int)

    # merge onto all blocks
    out = blocks[["block_id"]].merge(label, on="block_id", how="left")
    out["has_discovery_linked_to_field"] = out[
        "has_discovery_linked_to_field"
    ].fillna(False)
    out["commercial_success"] = out["commercial_success"].fillna(0).astype(int)
    return out


def compute_near_field_feature(
    blocks: gpd.GeoDataFrame,
    field_df: pd.DataFrame,
    wells_gdf: gpd.GeoDataFrame,
    near_km: float = 30.0,
) -> pd.DataFrame:
    """
    Compute near_field(block)=1 if block centroid is within near_km of any field location.
    Since field table may not have coords directly, we approximate field location by joining its listed wellbore (if present)
    to well coordinates, otherwise we skip.
    """
    f = field_df.copy()

    # Try to find a wellbore id to locate field
    f_well_id_col = pick_col(
        f,
        ["wlbNpdidWellbore", "wlbNpdidWellbore", "wlbNpdidWellboreReclass"],
        required=False,
    )
    if f_well_id_col is None:
        # Some field tables only have wlbName; try name join
        f_well_name_col = pick_col(
            f, ["wlbName", "wlbWellboreName"], required=False
        )
        if f_well_name_col is None:
            print(
                "WARN: field table has no wellbore link. near_field will be NaN.",
                file=sys.stderr,
            )
            out = blocks[["block_id"]].copy()
            out["near_field"] = np.nan
            out["dist_to_nearest_field_km"] = np.nan
            return out
        # name join
        wells_names = wells_gdf.dropna(subset=["wellbore_name"]).copy()
        wells_names["wellbore_name"] = wells_names["wellbore_name"].astype(str)
        f["_join_name"] = f[f_well_name_col].astype(str)
        f = f.merge(
            wells_names[["wellbore_name", "geometry"]],
            left_on="_join_name",
            right_on="wellbore_name",
            how="left",
        )
    else:
        wells_ids = wells_gdf.dropna(subset=["npdid_wellbore"]).copy()
        wells_ids["npdid_wellbore"] = pd.to_numeric(
            wells_ids["npdid_wellbore"], errors="coerce"
        )
        f["_join_id"] = pd.to_numeric(f[f_well_id_col], errors="coerce")
        f = f.merge(
            wells_ids[["npdid_wellbore", "geometry"]],
            left_on="_join_id",
            right_on="npdid_wellbore",
            how="left",
        )

    f_geo = gpd.GeoDataFrame(f, geometry="geometry", crs=f"EPSG:{WGS84_EPSG}")
    f_geo = f_geo.dropna(subset=["geometry"]).copy()
    if f_geo.empty:
        print(
            "WARN: could not derive field geometries from field table. near_field will be NaN.",
            file=sys.stderr,
        )
        out = blocks[["block_id"]].copy()
        out["near_field"] = np.nan
        out["dist_to_nearest_field_km"] = np.nan
        return out

    # Compute min distance in km between block centroid and all field points (vectorized)
    # For speed on large data: do in chunks
    bl = blocks[["block_id", "centroid_lon", "centroid_lat"]].copy()
    fld_lon = f_geo.geometry.x.values
    fld_lat = f_geo.geometry.y.values

    min_dist = []
    for _, row in bl.iterrows():
        dists = haversine_km(
            row["centroid_lon"], row["centroid_lat"], fld_lon, fld_lat
        )
        min_dist.append(float(np.nanmin(dists)) if len(dists) else np.nan)
    bl["dist_to_nearest_field_km"] = min_dist
    bl["near_field"] = (bl["dist_to_nearest_field_km"] <= near_km).astype(int)
    return bl[["block_id", "near_field", "dist_to_nearest_field_km"]]


def compute_seismic_coverage_ratio(
    blocks: gpd.GeoDataFrame, seismic_path: Optional[str]
) -> pd.DataFrame:
    """
    Compute seismicCoverageRatio(block)=area(intersection(block, seismic))/area(block).
    Requires a polygon layer of seismic coverage.
    """
    out = blocks[["block_id"]].copy()
    if not seismic_path:
        out["seismic_ratio"] = np.nan
        return out

    seismic = read_geo(seismic_path)
    seismic = ensure_wgs84(seismic)

    # project both to equal-area
    b_area = to_area_crs(
        gpd.GeoDataFrame(
            blocks[["block_id", "geometry"]],
            geometry="geometry",
            crs=blocks.crs,
        )
    )
    s_area = to_area_crs(seismic[["geometry"]].copy())

    # prepare spatial index and intersect
    ratios = []
    for idx, b in b_area.iterrows():
        geom_b = b.geometry
        if geom_b is None or geom_b.is_empty:
            ratios.append(np.nan)
            continue
        # candidate seismic polygons via bbox
        cand = s_area[s_area.intersects(geom_b)]
        if cand.empty:
            ratios.append(0.0)
            continue
        inter_area = cand.geometry.intersection(geom_b).area.sum()
        b_area_val = geom_b.area
        ratios.append(
            float(inter_area / b_area_val) if b_area_val > 0 else np.nan
        )

    out["seismic_ratio"] = ratios
    return out


def compute_strat_lith_scores(
    wells_blocks: pd.DataFrame, strat_path: Optional[str]
) -> pd.DataFrame:
    """
    Compute reservoir_score and seal_score at block level from strat/lith table.
    This function is robust to unknown column names: it searches for wellbore id/name and a lithology string column.
    Heuristic approach:
      reservoir lithologies: sandstone, limestone, dolomite, carbonate
      seal lithologies: shale, clay, mudstone, evaporite, salt
    Scores:
      fraction of rows matching the lithology group within the block (0..1).
    """
    out = pd.DataFrame(
        {"block_id": wells_blocks["block_id"].dropna().unique()}
    )
    out["reservoir_score"] = np.nan
    out["seal_score"] = np.nan
    if not strat_path:
        return out

    strat = read_table(strat_path)

    # attempt to find join key: wellbore id or name
    # In NPD strat files, often have wlbNpdidWellbore or wlbWellboreName or similar.
    strat_wid = pick_col(
        strat,
        ["wlbNpdidWellbore", "npdidWellbore", "wellboreId"],
        required=False,
    )
    strat_wname = pick_col(
        strat, ["wlbWellboreName", "wlbName", "wellboreName"], required=False
    )

    # lithology column candidates
    lith_col = pick_col(
        strat,
        [
            "lithology",
            "Lithology",
            "lith",
            "stratLith",
            "slbLithology",
            "litho",
        ],
        required=False,
    )
    if lith_col is None:
        # pick first object column with "lith" in name
        for c in strat.columns:
            if "lith" in c.lower():
                lith_col = c
                break
    if lith_col is None or (strat_wid is None and strat_wname is None):
        print(
            "WARN: strat file missing join key and/or lithology column; skipping reservoir/seal scores.",
            file=sys.stderr,
        )
        return out

    # map wells -> block via well id or name
    wb = wells_blocks.dropna(subset=["block_id"]).copy()
    wb["block_id"] = wb["block_id"].astype(str)

    # join using best available key
    if strat_wid is not None and "npdid_wellbore" in wb.columns:
        wb2 = wb.dropna(subset=["npdid_wellbore"]).copy()
        wb2["_join_id"] = pd.to_numeric(wb2["npdid_wellbore"], errors="coerce")
        strat["_join_id"] = pd.to_numeric(strat[strat_wid], errors="coerce")
        joined = strat.merge(
            wb2[["_join_id", "block_id"]].drop_duplicates(),
            on="_join_id",
            how="left",
        )
    else:
        # name join
        if "wellbore_name" in wb.columns and strat_wname is not None:
            wb2 = wb.dropna(subset=["wellbore_name"]).copy()
            wb2["_join_name"] = wb2["wellbore_name"].astype(str)
            strat["_join_name"] = strat[strat_wname].astype(str)
            joined = strat.merge(
                wb2[["_join_name", "block_id"]].drop_duplicates(),
                on="_join_name",
                how="left",
            )
        else:
            print(
                "WARN: cannot join strat to wells; skipping reservoir/seal scores.",
                file=sys.stderr,
            )
            return out

    joined = joined.dropna(subset=["block_id"]).copy()
    if joined.empty:
        print(
            "WARN: strat join resulted in empty table; skipping reservoir/seal scores.",
            file=sys.stderr,
        )
        return out

    lith = joined[lith_col].astype(str).str.lower()

    reservoir_terms = [
        "sandstone",
        "sand",
        "limestone",
        "dolomite",
        "carbonate",
    ]
    seal_terms = ["shale", "clay", "mudstone", "evaporite", "salt", "halite"]

    joined["is_reservoir_lith"] = lith.apply(
        lambda s: any(t in s for t in reservoir_terms)
    )
    joined["is_seal_lith"] = lith.apply(
        lambda s: any(t in s for t in seal_terms)
    )

    agg = (
        joined.groupby("block_id")
        .agg(
            reservoir_score=("is_reservoir_lith", "mean"),
            seal_score=("is_seal_lith", "mean"),
        )
        .reset_index()
    )
    return agg


def compute_geochem_flags(
    blocks: gpd.GeoDataFrame, geochem_path: Optional[str]
) -> pd.DataFrame:
    """
    Compute source_rock_flag and maturity_flag at block level from a geochem table.
    This is deliberately flexible because geochem exports differ. We search for columns:
      - block name/id OR coordinates OR wellbore id to link (we support two modes: by block name, or by wellbore id via external join not implemented here)
    For the demo, easiest is when geochem has block name column.
    Thresholds:
      Source rock: TOC >= 1.5 and HI >= 200
      Maturity: Ro >= 0.6 or Tmax >= 435
    """
    out = blocks[["block_id"]].copy()
    out["source_rock_flag"] = np.nan
    out["maturity_flag"] = np.nan

    if not geochem_path:
        return out

    g = read_table(geochem_path)

    # try to find block name column
    blk_col = pick_col(
        g,
        ["block", "blk", "blkName", "blockName", "prlAreaPolyBlockName"],
        required=False,
    )

    # try find numeric columns
    toc_col = pick_col(g, ["toc", "TOC"], required=False)
    hi_col = pick_col(g, ["hi", "HI", "hydrogenindex"], required=False)
    ro_col = pick_col(g, ["ro", "Ro", "vitrinitereflectance"], required=False)
    tmax_col = pick_col(g, ["tmax", "Tmax"], required=False)

    if blk_col is None:
        print(
            "WARN: geochem file has no obvious block column; skipping geochem flags (you can extend join logic).",
            file=sys.stderr,
        )
        return out

    g["_block_id"] = g[blk_col].astype(str).apply(slug_block_name)
    # numeric conversions
    if toc_col:
        g["_toc"] = pd.to_numeric(g[toc_col], errors="coerce")
    else:
        g["_toc"] = np.nan
    if hi_col:
        g["_hi"] = pd.to_numeric(g[hi_col], errors="coerce")
    else:
        g["_hi"] = np.nan
    if ro_col:
        g["_ro"] = pd.to_numeric(g[ro_col], errors="coerce")
    else:
        g["_ro"] = np.nan
    if tmax_col:
        g["_tmax"] = pd.to_numeric(g[tmax_col], errors="coerce")
    else:
        g["_tmax"] = np.nan

    g["source_rock_ok"] = (g["_toc"] >= 1.5) & (g["_hi"] >= 200)
    g["maturity_ok"] = (g["_ro"] >= 0.6) | (g["_tmax"] >= 435)

    agg = (
        g.groupby("_block_id")
        .agg(
            source_rock_flag=("source_rock_ok", "max"),
            maturity_flag=("maturity_ok", "max"),
        )
        .reset_index()
        .rename(columns={"_block_id": "block_id"})
    )

    out = out.merge(agg, on="block_id", how="left")
    out["source_rock_flag"] = out["source_rock_flag"].fillna(False).astype(int)
    out["maturity_flag"] = out["maturity_flag"].fillna(False).astype(int)
    return out


def export_facts(features: pd.DataFrame, facts_path: str) -> None:
    """
    Export Prolog/ProbLog-friendly facts for each block.
    Guarantees presence of key predicates for EVERY block by using defaults:
      seismic_ratio(B,0.0)
      reservoir_score(B,0.0)
      seal_score(B,0.0)
      near_field(B,0)
      deepwater_penalty(B,0)
    """

    def as_int(x, default=0) -> int:
        try:
            if pd.isna(x):
                return int(default)
            return int(x)
        except Exception:
            return int(default)

    def as_float(x, default=0.0) -> float:
        try:
            if pd.isna(x):
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    lines = []
    for _, r in features.iterrows():
        bid = str(r["block_id"])

        # required anchor
        lines.append(f"block({bid}).")

        # counts (defaults: 0)
        lines.append(f"well_count({bid},{as_int(r.get('well_count', 0), 0)}).")
        lines.append(
            f"dry_well_count({bid},{as_int(r.get('dry_well_count', 0), 0)})."
        )

        # numeric evidence scores (defaults: 0.0)
        lines.append(
            f"seismic_ratio({bid},{as_float(r.get('seismic_ratio', 0.0), 0.0):.4f})."
        )
        lines.append(
            f"reservoir_score({bid},{as_float(r.get('reservoir_score', 0.0), 0.0):.4f})."
        )
        lines.append(
            f"seal_score({bid},{as_float(r.get('seal_score', 0.0), 0.0):.4f})."
        )

        # binary flags (defaults: 0)
        lines.append(f"near_field({bid},{as_int(r.get('near_field', 0), 0)}).")
        lines.append(
            f"deepwater_penalty({bid},{as_int(r.get('deepwater_penalty', 0), 0)})."
        )

        # label (defaults: 0)
        if "commercial_success" in features.columns:
            lines.append(
                f"commercial_success({bid},{as_int(r.get('commercial_success', 0), 0)})."
            )

        lines.append("")  # blank line between blocks

    Path(facts_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_features(
    wellbore_csv: str,
    discovery_csv: str,
    field_csv: str,
    blocks_path: str,
    seismic_path: Optional[str],
    strat_path: Optional[str],
    geochem_path: Optional[str],
    t0: Optional[str],
    near_km: float,
) -> pd.DataFrame:
    blocks = load_blocks(blocks_path)
    wells_gdf = load_wellbores(wellbore_csv, t0=t0)
    wells_blocks = spatial_join_wells_to_blocks(wells_gdf, blocks)

    wells_agg = aggregate_wells_by_block(wells_blocks)

    discovery = load_discovery(discovery_csv, t0=t0)
    field = load_field(field_csv, t0=t0)

    label = derive_commercial_success_label(
        blocks, wells_blocks, discovery, field, t0=t0
    )
    near_field = compute_near_field_feature(
        blocks, field, wells_gdf, near_km=near_km
    )
    seismic = compute_seismic_coverage_ratio(blocks, seismic_path)
    strat_scores = compute_strat_lith_scores(wells_blocks, strat_path)
    geochem_flags = compute_geochem_flags(blocks, geochem_path)

    # Merge everything to blocks
    features = blocks[
        [
            "block_id",
            "block_name_raw",
            "region",
            "block_area_km2",
            "centroid_lon",
            "centroid_lat",
        ]
    ].copy()
    for df in [
        wells_agg,
        seismic,
        strat_scores,
        geochem_flags,
        near_field,
        label,
    ]:
        features = features.merge(df, on="block_id", how="left")

    # Fill missing numeric where appropriate
    for c in ["well_count", "dry_well_count", "exploration_like_count"]:
        if c in features.columns:
            features[c] = features[c].fillna(0).astype(int)

    # Derived boolean flags using the v1.0 demo thresholds
    if "seismic_ratio" in features.columns:
        features["seismic_good"] = (features["seismic_ratio"] >= 0.6).astype(
            int
        )
        features["seismic_poor"] = (features["seismic_ratio"] < 0.4).astype(
            int
        )
    else:
        features["seismic_good"] = np.nan
        features["seismic_poor"] = np.nan

    features["poorly_explored"] = (features["well_count"] < 2).astype(int)
    features["many_dry_wells"] = (features["dry_well_count"] >= 3).astype(int)

    # Optional economic flag (deepwater penalty)
    if "water_depth_avg" in features.columns:
        features["deepwater_penalty"] = (
            features["water_depth_avg"] > 400
        ).astype(int)
    else:
        features["deepwater_penalty"] = np.nan

    # MAX_BLOCKS = 25
    # features = features.sample(MAX_BLOCKS, random_state=42)

    return features


def write_derived_facts(block_features_path: str, out_derived_facts: str):
    df = pd.read_csv(block_features_path)
    df["block_id"] = df["block_id"].apply(slug)

    # safe numeric getters
    def num(col, default=0.0):
        return (
            pd.to_numeric(df[col], errors="coerce").fillna(default)
            if col in df.columns
            else pd.Series(default, index=df.index)
        )

    dist = num("dist_to_nearest_field_km", np.nan)
    wdepth = num("water_depth_p50", np.nan)
    wells = num("well_count", 0)
    dry = num("dry_well_count", 0)
    expl = num("exploration_like_count", 0)

    near = num("near_field", 0).astype(int)
    deep = num("deepwater_penalty", 0).astype(int)

    lines = []
    lines.append("% ===== derived facts (no seismic) =====")

    for i, r in df.iterrows():
        b = r["block_id"]

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

        # simple booleans
        if near.iloc[i] == 1:
            lines.append(f"near_field({b}).")
        if deep.iloc[i] == 1:
            lines.append(f"deepwater_penalty({b}).")

    with open(out_derived_facts, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(
        f"Записаны derived_facts {out_derived_facts}, всего записей: {len(lines)}",
    )


@app.command()
def main(
    ld: Path = typer.Option(
        RAW_DATA_DIR,
        "--local_dir",
        "-ld",
        help="Локальная директория для сохранения исходных данных",
    ),
    wellbore: Path = typer.Option(
        RAW_DATA_DIR / "wellbore_all_long.csv",
        "--wellbore",
        "-wb",
        exists=True,
        help="Путь до wellbore_all_long.csv",
    ),
    discovery: Path = typer.Option(
        RAW_DATA_DIR / "discovery.csv",
        "--discovery",
        "-d",
        exists=True,
        help="Путь до discovery.csv",
    ),
    field: Path = typer.Option(
        RAW_DATA_DIR / "field.csv",
        "--field",
        "-f",
        exists=True,
        help="Путь до field.csv",
    ),
    blocks: Path = typer.Option(
        RAW_DATA_DIR / "blkArea (1).zip",
        "--blocks",
        "-b",
        help="Путь до blkArea (1).zip или shapefile",
    ),
    seismic: Path | None = typer.Option(
        None,
        "--seismic",
        "-s",
        help="GeoJSON/Shapefile (seismic coverage polygons)",
    ),
    strat: Path | None = typer.Option(
        RAW_DATA_DIR / "strat_litho_wellbore.csv",
        "--strat",
        "-st",
        help="Стратиграфия по скважинам",
    ),
    geochem: Path | None = typer.Option(
        # RAW_DATA_DIR / "NOD_NOCS_Complete_01-2026.txt",
        None,
        "--geochem",
        "-gc",
        help="Геохимический состав",
    ),
    t0: str = typer.Option(
        "2010-01-01",
        "--cutoff",
        "-t0",
        help="Временная граница для отсекания данных",
    ),
    near_km: float = typer.Option(
        30.0,
        "--near_km",
        "-n",
        help="Радиус близости к месторождению (км)",
    ),
    out: Path = typer.Option(
        PROCESSED_DATA_DIR / "block_features.csv",
        "--out",
        "-o",
        help="Путь до файла, куда будет записан результат",
    ),
    facts: Path = typer.Option(
        PROCESSED_DATA_DIR / "facts_blocks.pl",
        "--facts",
        "-f",
        help="Путь до файла, куда будут записаны факты Prolog/ProbLog",
    ),
    derived_facts_noseis: Path = typer.Option(
        PROCESSED_DATA_DIR / "derived_facts_noseis.pl",
        "--derived_facts_noseis",
        "-df",
        help="Путь до файла, куда будут записаны derived_facts_noseis",
    ),
):
    logger.info("Скачивание данных: https://factpages.sodir.no...")
    logger.info(f"Поиск данных данных в {PROCESSED_DATA_DIR}...")

    features = build_features(
        wellbore_csv=wellbore,
        discovery_csv=discovery,
        field_csv=field,
        blocks_path=blocks,
        seismic_path=seismic,
        strat_path=strat,
        geochem_path=geochem,
        t0=t0,
        near_km=near_km,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(
        f"Записан файл {out_path} с {len(features)} блоками и {features.shape[1]} колонками."
    )

    if facts:
        export_facts(features, facts)
        logger.success(f"Факты записаны в {facts}")

    cols = [
        "well_count",
        "dry_well_count",
        "seismic_ratio",
        "near_field",
        "commercial_success",
    ]
    present = [c for c in cols if c in features.columns]
    if present:
        logger.info("\nПроверка (средние значения):")
        logger.info(features[present].mean(numeric_only=True))

    write_derived_facts(str(out_path), str(derived_facts_noseis))

    logger.success(f"Обработка данных завершена {RAW_DATA_DIR}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
