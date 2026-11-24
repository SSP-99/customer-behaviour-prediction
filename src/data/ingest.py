"""Raw data ingestion for UCI Online Retail II (local Excel, ucimlrepo, or UCI static ZIP)."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def _get_uci_zip_url() -> str:
    """Construct the official UCI static archive URL for Online Retail II (id=502)."""
    parts = [
        "https:/",
        "archive.ics.uci.edu",
        "static",
        "public",
        "502",
        "online+retail+ii.zip",
    ]
    return "/".join(parts)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names between Excel sheets and ucimlrepo schemas.

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with standardized column names matching CANONICAL_COLUMNS.
    """
    rename_map = {
        "InvoiceNo": "Invoice",
        "UnitPrice": "Price",
        "CustomerID": "Customer ID",
    }
    df = df.rename(columns=rename_map)
    return df[CANONICAL_COLUMNS].copy()


def _download_and_extract_uci_excel(destination: Path) -> None:
    """Download the official Online Retail II ZIP from UCI and extract the .xlsx workbook.

    Args:
        destination: Local target path (e.g., data/raw/online_retail_II.xlsx).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _get_uci_zip_url()
    logger.info("Downloading Online Retail II ZIP (~45 MB) from %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xlsx_members = [m for m in zf.namelist() if m.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError(f"No .xlsx file found inside UCI archive: {zf.namelist()}")
        with zf.open(next(iter(xlsx_members))) as src_file, destination.open("wb") as out_file:
            out_file.write(src_file.read())

    logger.info(
        "Extracted Excel workbook to %s (%.2f MB)",
        destination,
        destination.stat().st_size / 1e6,
    )


def load_raw_transactions(config: AppConfig) -> pd.DataFrame:
    """Load raw Online Retail II transactions from local Excel, cache, or UCI archive.

    Args:
        config: Application configuration dataclass.

    Returns:
        Combined raw DataFrame with canonical column names.
    """
    excel_path = Path(config.data.raw_excel_path)
    cached_raw_parquet = excel_path.parent / "online_retail_II_raw.parquet"

    if cached_raw_parquet.exists():
        logger.info("Loading cached raw Parquet from: %s", cached_raw_parquet)
        df_raw = pd.read_parquet(cached_raw_parquet)
        logger.info("Loaded %d total raw rows from cached Parquet.", len(df_raw))
        return _standardize_columns(df_raw)

    if not excel_path.exists():
        logger.info(
            "Local Excel not found at %s. Trying ucimlrepo id=%d...",
            excel_path,
            config.data.uci_repo_id,
        )
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=config.data.uci_repo_id)
            if dataset.data is not None and dataset.data.features is not None:
                df_raw = dataset.data.features.copy()
                df_raw = _standardize_columns(df_raw)
                cached_raw_parquet.parent.mkdir(parents=True, exist_ok=True)
                df_raw.to_parquet(cached_raw_parquet, index=False)
                return df_raw
        except Exception as exc:
            logger.warning(
                "ucimlrepo unavailable for id=%d (%s); downloading official UCI ZIP.",
                config.data.uci_repo_id,
                exc,
            )
            _download_and_extract_uci_excel(excel_path)

    logger.info("Reading raw transactions from local Excel: %s", excel_path)
    sheets: list[pd.DataFrame] = []
    for sheet_name in config.data.raw_sheets:
        logger.info("Reading sheet '%s' (this takes ~15-20s for 500k+ rows)...", sheet_name)
        sheet_df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
        sheets.append(_standardize_columns(sheet_df))

    df_raw = pd.concat(sheets, ignore_index=True)
    for col in ("Invoice", "StockCode", "Description", "Country"):
        df_raw[col] = df_raw[col].astype(str)

    cached_raw_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(cached_raw_parquet, index=False)
    logger.info("Loaded %d total raw rows and cached to %s", len(df_raw), cached_raw_parquet)
    return df_raw
