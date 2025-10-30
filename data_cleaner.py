
"""
data_cleaner.py — A dependency-light, production-friendly Excel/CSV data cleaning toolkit.

Features
--------
- Ingest CSV or Excel (single sheet or by name/index)
- Standardize headers (lowercase, snake_case, dedupe)
- Trim whitespace & normalize NA tokens ("n/a","nil","-","—","NA", etc.)
- Remove fully empty rows/columns
- Type inference & casting (numeric, booleans, dates with multiple formats)
- Date normalization to ISO (YYYY-MM-DD) or full datetime
- Duplicate detection & removal
- Missing value imputation: mean/median/mode/ffill/bfill/constant per column
- Outlier handling via IQR (remove or winsorize)
- Lightweight validation and data-quality scoring
- CLI with JSON-config support

Usage
-----
Python API:
    from data_cleaner import DataCleaner, CleanConfig

    cfg = CleanConfig(
        na_tokens=["", "na", "n/a", "-", "—", "nil", "null"],
        date_cols=["date", "order_date"],
        date_iso=True,
        impute={"revenue": "median", "segment": "mode"},
        drop_duplicates=True,
        outliers={"method": "iqr", "action": "winsorize", "cols": ["revenue"], "k": 1.5},
    )

    cleaner = DataCleaner(cfg)
    df = cleaner.read("raw.xlsx", sheet="Sheet1")
    cleaned, report = cleaner.clean(df)
    cleaner.write(cleaned, "cleaned.xlsx")

CLI:
    python data_cleaner.py --input raw.xlsx --output cleaned.xlsx --sheet Sheet1 \
        --config sample_clean_config.json

Config file schema (JSON):
{
  "na_tokens": ["", "na", "n/a", "-", "—", "nil", "null"],
  "strip_whitespace": true,
  "dedupe_headers": true,
  "date_cols": ["date"],
  "date_iso": true,
  "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"],
  "cast_numeric": ["revenue","quantity"],
  "cast_bool": ["is_active"],
  "impute": {"revenue": "median", "segment": "mode", "age": {"strategy":"constant","value":0}},
  "drop_duplicates": true,
  "remove_empty_rows": true,
  "remove_empty_cols": true,
  "outliers": {"method":"iqr","action":"winsorize","cols":["revenue"], "k":1.5},
  "validation": {
    "no_empty_headers": true,
    "required_cols": ["date","revenue"],
    "unique_cols": ["id"],
    "non_negative": ["revenue","quantity"]
  }
}
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import math
from datetime import datetime
import argparse


# ---------------------- Config & Report dataclasses ----------------------

@dataclass
class CleanConfig:
    na_tokens: Optional[List[str]] = None
    strip_whitespace: bool = True
    dedupe_headers: bool = True
    date_cols: Optional[List[str]] = None
    date_iso: bool = True                   # normalize to YYYY-MM-DD if True; else keep datetime
    date_formats: Optional[List[str]] = None
    cast_numeric: Optional[List[str]] = None
    cast_bool: Optional[List[str]] = None
    impute: Optional[Dict[str, Any]] = None # {"col":"median"|...} or {"col":{"strategy":"constant","value":0}}
    drop_duplicates: bool = True
    remove_empty_rows: bool = True
    remove_empty_cols: bool = True
    outliers: Optional[Dict[str, Any]] = None # {"method":"iqr","action":"winsorize"|"remove","cols":[...], "k":1.5}
    validation: Optional[Dict[str, Any]] = None

@dataclass
class QualityReport:
    rows_before: int
    cols_before: int
    rows_after: int
    cols_after: int
    missing_by_col: Dict[str, int]
    dtypes: Dict[str, str]
    duplicates_removed: int
    outliers_modified: Dict[str, int]
    warnings: List[str]

# ---------------------- Utility helpers ----------------------

_SNAKE_RE = re.compile(r'[^0-9a-zA-Z]+')

def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = _SNAKE_RE.sub('_', s)
    s = s.strip('_')
    return s

def dedupe_list(items: List[str]) -> List[str]:
    seen = {}
    result = []
    for x in items:
        base = x
        i = seen.get(base, 0)
        if i == 0 and base not in seen:
            seen[base] = 1
            result.append(base)
        else:
            # find next suffix
            while True:
                i += 1
                candidate = f"{base}_{i}"
                if candidate not in seen:
                    seen[base] = i
                    seen[candidate] = 1
                    result.append(candidate)
                    break
    return result

def is_date_series(s: pd.Series) -> bool:
    # Heuristic: try parsing a sample
    try:
        pd.to_datetime(s.dropna().astype(str).head(10), errors='raise', infer_datetime_format=True)
        return True
    except Exception:
        return False

def safe_to_datetime(series: pd.Series, formats: Optional[List[str]]=None) -> pd.Series:
    if formats:
        parsed = None
        for fmt in formats:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                if parsed.notna().any():
                    series = series.where(parsed.isna(), parsed)
            except Exception:
                continue
        # Fallback try
        fallback = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        return fallback
    else:
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)

def winsorize_iqr(col: pd.Series, k: float=1.5) -> Tuple[pd.Series, int]:
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    before = col.copy()
    clipped = col.clip(lower, upper)
    modified = int((before != clipped).sum())
    return clipped, modified

def remove_outliers_iqr(col: pd.Series, k: float=1.5) -> Tuple[pd.Series, int]:
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (col < lower) | (col > upper)
    removed = int(mask.sum())
    return col.mask(mask, np.nan), removed

# ---------------------- Core class ----------------------

class DataCleaner:
    def __init__(self, config: Optional[CleanConfig]=None):
        self.config = config or CleanConfig(
            na_tokens=["", "na", "n/a", "-", "—", "nil", "null"],
            strip_whitespace=True,
            dedupe_headers=True,
            date_cols=None,
            date_iso=True,
            date_formats=None,
            cast_numeric=None,
            cast_bool=None,
            impute=None,
            drop_duplicates=True,
            remove_empty_rows=True,
            remove_empty_cols=True,
            outliers=None,
            validation={
                "no_empty_headers": True
            }
        )

    # ---------- IO ----------
    def read(self, path: Union[str, Path], sheet: Optional[Union[str, int]]=None) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
            if sheet is not None:
                return pd.read_excel(path, sheet_name=sheet)
            else:
                return pd.read_excel(path)  # default first sheet
        else:
            # CSV (or TSV if delimiter detected)
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_csv(path, sep="\t")

    def write(self, df: pd.DataFrame, path: Union[str, Path]):
        path = Path(path)
        if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="cleaned")
        else:
            df.to_csv(path, index=False)

    # ---------- Cleaning pipeline ----------
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        rows_before, cols_before = df.shape
        warnings: List[str] = []
        outlier_mods: Dict[str, int] = {}

        # 1) Standardize headers
        headers = list(df.columns)
        new_headers = [to_snake(str(h)) for h in headers]
        if self.config.dedupe_headers:
            new_headers = dedupe_list(new_headers)
        df.columns = new_headers
        if self.config.validation and self.config.validation.get("no_empty_headers", False):
            if any(h == "" for h in df.columns):
                warnings.append("Empty column headers detected; replaced with snake_case placeholders.")

        # 2) Strip whitespace
        if self.config.strip_whitespace:
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # 3) Harmonize NA tokens
        na_tokens = set([t.lower() for t in (self.config.na_tokens or [])])
        if na_tokens:
            def _normalize_na(x):
                if isinstance(x, str) and x.strip().lower() in na_tokens:
                    return np.nan
                return x
            df = df.applymap(_normalize_na)

        # 4) Remove empty rows/cols
        if self.config.remove_empty_rows:
            df = df.dropna(how="all")
        if self.config.remove_empty_cols:
            df = df.dropna(axis=1, how="all")

        # 5) Type casting
        # Numeric
        if self.config.cast_numeric:
            for c in self.config.cast_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

        # Bool
        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f"}
        if self.config.cast_bool:
            for c in self.config.cast_bool:
                if c in df.columns:
                    def _to_bool(val):
                        if isinstance(val, bool): return val
                        if pd.isna(val): return np.nan
                        s = str(val).strip().lower()
                        if s in truthy: return True
                        if s in falsy: return False
                        return np.nan
                    df[c] = df[c].map(_to_bool)

        # Dates
        if self.config.date_cols:
            for c in self.config.date_cols:
                if c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                    if self.config.date_formats:
                        # Attempt specific formats if many NAs
                        if parsed.isna().mean() > 0.3:
                            for fmt in self.config.date_formats:
                                parsed2 = pd.to_datetime(df[c], format=fmt, errors="coerce")
                                parsed = parsed.where(parsed.notna(), parsed2)
                    if self.config.date_iso:
                        df[c] = parsed.dt.date.astype("string")
                    else:
                        df[c] = parsed

        # 6) Drop duplicates
        duplicates_removed = 0
        if self.config.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            duplicates_removed = before - len(df)

        # 7) Imputation
        if self.config.impute:
            for col, rule in self.config.impute.items():
                if col not in df.columns:
                    continue
                series = df[col]
                if isinstance(rule, str):
                    if rule == "mean" and pd.api.types.is_numeric_dtype(series):
                        df[col] = series.fillna(series.mean())
                    elif rule == "median" and pd.api.types.is_numeric_dtype(series):
                        df[col] = series.fillna(series.median())
                    elif rule == "mode":
                        mode_val = series.mode(dropna=True)
                        fillv = mode_val.iloc[0] if not mode_val.empty else None
                        df[col] = series.fillna(fillv)
                    elif rule == "ffill":
                        df[col] = series.fillna(method="ffill")
                    elif rule == "bfill":
                        df[col] = series.fillna(method="bfill")
                elif isinstance(rule, dict):
                    strat = rule.get("strategy")
                    if strat == "constant":
                        df[col] = series.fillna(rule.get("value"))
                    elif strat == "mean" and pd.api.types.is_numeric_dtype(series):
                        df[col] = series.fillna(series.mean())
                    elif strat == "median" and pd.api.types.is_numeric_dtype(series):
                        df[col] = series.fillna(series.median())
                    elif strat == "mode":
                        mode_val = series.mode(dropna=True)
                        fillv = mode_val.iloc[0] if not mode_val.empty else None
                        df[col] = series.fillna(fillv)

        # 8) Outliers
        if self.config.outliers:
            method = self.config.outliers.get("method","iqr")
            action = self.config.outliers.get("action","winsorize") # or "remove"
            cols = self.config.outliers.get("cols", [])
            k = float(self.config.outliers.get("k", 1.5))
            for c in cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    if action == "winsorize":
                        df[c], modified = winsorize_iqr(df[c].astype(float), k=k)
                        outlier_mods[c] = modified
                    elif action == "remove":
                        df[c], removed = remove_outliers_iqr(df[c].astype(float), k=k)
                        outlier_mods[c] = removed

        # 9) Validation
        if self.config.validation:
            v = self.config.validation
            if v.get("required_cols"):
                missing = [c for c in v["required_cols"] if c not in df.columns]
                if missing:
                    warnings.append(f"Missing required columns: {missing}")
            if v.get("unique_cols"):
                for c in v["unique_cols"]:
                    if c in df.columns:
                        dups = df[c].duplicated(keep=False).sum()
                        if dups > 0:
                            warnings.append(f"Column '{c}' violates uniqueness: {dups} duplicate values.")
            if v.get("non_negative"):
                for c in v["non_negative"]:
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                        negs = (df[c] < 0).sum()
                        if negs > 0:
                            warnings.append(f"Column '{c}' has {negs} negative values but expects non-negative.")

        # Report
        rows_after, cols_after = df.shape
        missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
        dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
        report = QualityReport(
            rows_before=rows_before,
            cols_before=cols_before,
            rows_after=rows_after,
            cols_after=cols_after,
            missing_by_col=missing_by_col,
            dtypes=dtypes,
            duplicates_removed=duplicates_removed,
            outliers_modified=outlier_mods,
            warnings=warnings
        )
        return df, report

# ---------------------- CLI ----------------------

def load_config(path: Optional[str]) -> CleanConfig:
    if not path:
        return CleanConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return CleanConfig(**raw)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Data Cleaner: Clean CSV/Excel with a JSON config.")
    parser.add_argument("--input", required=True, help="Path to CSV or Excel file to clean")
    parser.add_argument("--output", required=True, help="Where to write cleaned data (csv/xlsx)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index for input")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cleaner = DataCleaner(cfg)
    # Parse sheet as int if possible
    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    df = cleaner.read(args.input, sheet=sheet)
    cleaned, report = cleaner.clean(df)
    cleaner.write(cleaned, args.output)

    # Print a compact report to stdout
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main()
