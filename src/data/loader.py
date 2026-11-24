"""Raw data loader for UCI Online Retail II with snake_case standardization."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

SNAKE_CASE_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


def _get_uci_zip_url() -> str:
    """Return official UCI static archive URL for Online Retail II (id=502)."""
    return "/".join(
        [
            "https:/",
            "archive.ics.uci.edu",
            "static",
            "public",
            "502",
            "online+retail+ii.zip",
        ]
    )


def standardize_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case, enforce dtypes, and sort chronologically.

    Args:
        df: Raw input DataFrame from ucimlrepo, Excel, or cached Parquet.

    Returns:
        Chronologically sorted DataFrame with standardized snake_case columns.
    """
    rename_map = {
        "Invoice": "invoice",
        "InvoiceNo": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "price",
        "UnitPrice": "price",
        "Customer ID": "customer_id",
        "CustomerID": "customer_id",
        "Country": "country",
    }
    standardized = df.rename(columns=rename_map).copy()
    standardized = standardized[SNAKE_CASE_COLUMNS].copy()

    standardized["invoice"] = standardized["invoice"].astype(str).str.strip()
    standardized["stock_code"] = standardized["stock_code"].astype(str).str.strip()
    standardized["description"] = standardized["description"].where(
        standardized["description"].notna(), None
    )
    standardized["quantity"] = pd.to_numeric(standardized["quantity"], errors="coerce")
    standardized["invoice_date"] = pd.to_datetime(standardized["invoice_date"], errors="coerce")
    standardized["price"] = pd.to_numeric(standardized["price"], errors="coerce")
    standardized["customer_id"] = pd.to_numeric(standardized["customer_id"], errors="coerce")
    standardized["country"] = standardized["country"].fillna("Unknown").astype(str).str.strip()

    standardized = standardized.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)
    return standardized


def _download_uci_excel(destination: Path) -> None:
    """Download Online Retail II ZIP archive from UCI and extract the .xlsx file.

    Args:
        destination: Local target path for the extracted Excel workbook.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _get_uci_zip_url()
    logger.info("Downloading Online Retail II archive from %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xlsx_members = [m for m in zf.namelist() if m.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError(f"No .xlsx file found inside archive: {zf.namelist()}")
        first_member = next(iter(xlsx_members))
        with zf.open(first_member) as src_file, destination.open("wb") as out_file:
            out_file.write(src_file.read())

    logger.info("Extracted Excel workbook to %s", destination)


def _save_with_csv_fallback(df: pd.DataFrame, target_path: Path) -> Path:
    """Save DataFrame to Parquet, falling back to CSV if Parquet serialization fails.

    Args:
        df: DataFrame to persist.
        target_path: Desired Parquet file path.

    Returns:
        Actual Path written on disk.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target_path, index=False)
        logger.info("Saved %d rows to Parquet: %s", len(df), target_path)
        return target_path
    except Exception as exc:
        csv_path = target_path.with_suffix(".csv")
        logger.warning("Parquet save failed (%s); falling back to CSV: %s", exc, csv_path)
        df.to_csv(csv_path, index=False)
        return csv_path


def load_raw_data(config: AppConfig, force_reload: bool = False) -> pd.DataFrame:
    """Load raw Online Retail II data via ucimlrepo or local Excel workbook.

    Tries `ucimlrepo.fetch_ucirepo(id=502)` first, falling back to the local Excel
    workbook (`config.data.raw_excel_path`, concatenating both configured sheets).
    Standardizes column names to snake_case, enforces dtypes, sorts chronologically,
    and persists to `config.data.raw_combined_path` (with CSV fallback).

    Args:
        config: Application configuration dataclass.
        force_reload: If True, bypasses existing raw_combined cache and reloads from source.

    Returns:
        Standardized, chronologically sorted raw transactions DataFrame.
    """
    combined_path = Path(config.data.raw_combined_path)
    legacy_raw_cache = Path(config.data.raw_excel_path).parent / "online_retail_II_raw.parquet"

    if combined_path.exists() and not force_reload:
        logger.info("Loading existing standardized raw dataset from %s", combined_path)
        return pd.read_parquet(combined_path)

    df_raw: pd.DataFrame | None = None

    try:
        from ucimlrepo import fetch_ucirepo

        logger.info("Attempting to load dataset id=%d via ucimlrepo...", config.data.uci_repo_id)
        dataset = fetch_ucirepo(id=config.data.uci_repo_id)
        if dataset.data is not None and dataset.data.features is not None:
            df_raw = dataset.data.features.copy()
            logger.info("Loaded %d rows via ucimlrepo.", len(df_raw))
    except Exception as exc:
        logger.warning("ucimlrepo unavailable (%s); falling back to local Excel.", exc)

    if df_raw is None:
        if legacy_raw_cache.exists() and not force_reload:
            logger.info("Loading raw rows from cached archive %s", legacy_raw_cache)
            df_raw = pd.read_parquet(legacy_raw_cache)
        else:
            excel_path = Path(config.data.raw_excel_path)
            if not excel_path.exists():
                _download_uci_excel(excel_path)
            logger.info("Reading sheets %s from %s", config.data.raw_sheets, excel_path)
            sheets = [
                pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
                for sheet in config.data.raw_sheets
            ]
            df_raw = pd.concat(sheets, ignore_index=True)

    df_standardized = standardize_raw_schema(df_raw)
    _save_with_csv_fallback(df_standardized, combined_path)
    return df_standardized
