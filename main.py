# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import polars as pl
from io import BytesIO, StringIO
import hashlib
import json
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path

app = FastAPI(title="Polars Data Cleaning API (lazy)")

# Default config schema (simple)
class CleanConfig(BaseModel):
    na_tokens: Optional[List[str]] = ["", "na", "n/a", "-", "—", "nil", "null"]
    strip_whitespace: bool = True
    dedupe_headers: bool = True
    cast_numeric: Optional[List[str]] = None
    cast_bool: Optional[List[str]] = None
    date_cols: Optional[List[str]] = None
    date_formats: Optional[List[str]] = None
    drop_duplicates: bool = True

CACHE_DIR = Path(tempfile.gettempdir()) / "polars_clean_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# helpers
def to_snake(s: str) -> str:
    import re
    s = (s or "").strip().lower()
    s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
    s = s.strip('_')
    return s or "col"

def dedupe_headers(headers: List[str]) -> List[str]:
    seen = {}
    out = []
    for h in headers:
        base = h
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            i = seen[base] + 1
            # find unused suffix
            while f"{base}_{i}" in seen:
                i += 1
            seen[base] = i
            seen[f"{base}_{i}"] = 1
            out.append(f"{base}_{i}")
    return out

def compute_cache_key(file_bytes: bytes, config_json: str) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(config_json.encode("utf-8"))
    return h.hexdigest()

def load_csv_to_lazyframe(file_bytes: bytes, infer_schema_length: int = 1000) -> pl.LazyFrame:
    # polars can read from BytesIO - create DataFrame then lazy
    return pl.read_csv(BytesIO(file_bytes), infer_schema_length=infer_schema_length).lazy()

def string_cols(ldf: pl.LazyFrame) -> List[str]:
    # collect minimal schema
    schema = ldf.schema
    return [c for c, t in schema.items() if pl.datatypes.Utf8 == t or "Utf8" in str(t)]

@app.post("/clean")
async def clean_endpoint(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None)  # JSON string
):
    """
    Accepts multipart upload: file (csv). Optional `config` form field is JSON string of CleanConfig.
    Returns cleaned CSV (attachment) and a JSON report in headers/body.
    """
    raw = await file.read()
    cfg_obj: Dict[str, Any] = {}
    try:
        if config:
            cfg_obj = json.loads(config)
    except Exception as e:
        return JSONResponse({"error": "Invalid JSON in `config`"}, status_code=400)

    config_json = json.dumps(cfg_obj, sort_keys=True)
    cache_key = compute_cache_key(raw, config_json)
    cached_csv = CACHE_DIR / f"{cache_key}.csv"
    cached_report = CACHE_DIR / f"{cache_key}.report.json"

    # If cached, return cached file + report
    if cached_csv.exists() and cached_report.exists():
        report = json.loads(cached_report.read_text(encoding="utf-8"))
        return StreamingResponse(
            cached_csv.open("rb"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="cleaned_{file.filename}"',
                "X-Clean-Report": json.dumps(report)
            }
        )

    # Build config with defaults
    default_cfg = CleanConfig().dict()
    # merge provided keys
    merged = {**default_cfg, **cfg_obj}
    cfg = CleanConfig(**merged)

    # Load into Polars LazyFrame
    try:
        ldf = load_csv_to_lazyframe(raw)
    except Exception as e:
        return JSONResponse({"error": f"Failed to parse CSV: {str(e)}"}, status_code=400)

    # ---- 1) Standardize & dedupe headers ----
    # Get current columns (materialize small schema)
    cols = list(ldf.columns)
    new_cols = [to_snake(c) for c in cols]
    if cfg.dedupe_headers:
        new_cols = dedupe_headers(new_cols)
    # rename mapping
    if new_cols != cols:
        ldf = ldf.rename({old: new for old, new in zip(cols, new_cols)})
    # ---- 2) Trim whitespace on string cols ----
    if cfg.strip_whitespace:
        # apply str.strip to all utf8 columns lazily
        for c in string_cols(ldf):
            ldf = ldf.with_columns(pl.col(c).str.strip().alias(c))

    # ---- 3) Normalize NA tokens ----
    na_tokens = [t.lower() for t in (cfg.na_tokens or [])]
    if na_tokens:
        # For each string col: replace tokens with null
        for c in string_cols(ldf):
            # lower then check membership, keep original if not a token
            ldf = ldf.with_columns(
                pl.when(pl.col(c).is_null())
                  .then(pl.col(c))
                  .otherwise(
                      pl.when(pl.col(c).str.to_lowercase().is_in(na_tokens))
                        .then(pl.lit(None))
                        .otherwise(pl.col(c))
                  ).alias(c)
            )

    # ---- 4) Type coercion: numeric, bool, date (best-effort) ----
    if cfg.cast_numeric:
        for c in cfg.cast_numeric:
            if c in ldf.columns:
                # try cast to Float64 (coerce invalid -> null)
                ldf = ldf.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))

    if cfg.cast_bool:
        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f"}
        for c in cfg.cast_bool:
            if c in ldf.columns:
                # map known values to boolean, else null
                ldf = ldf.with_columns(
                    pl.when(pl.col(c).is_null()).then(pl.lit(None))
                    .otherwise(
                        pl.when(pl.col(c).str.to_lowercase().is_in(list(truthy))).then(pl.lit(True))
                          .when(pl.col(c).str.to_lowercase().is_in(list(falsy))).then(pl.lit(False))
                          .otherwise(pl.lit(None))
                    ).alias(c)
                )

    # Basic date coercion - attempt to parse date_cols if given
    if cfg.date_cols:
        for c in cfg.date_cols:
            if c in ldf.columns:
                if cfg.date_formats:
                    # try formats in order, fallback to parse
                    parsed = None
                    for fmt in cfg.date_formats:
                        # polars supports strptime for expressions
                        # create column parsed_fmt; keep first successful parse
                        try:
                            parsed_expr = pl.col(c).str.strptime(pl.Date, fmt)
                            ldf = ldf.with_columns(parsed_expr.alias(f"{c}__parsed_tmp"))
                            # replace original where parsed not null
                            ldf = ldf.with_columns(
                                pl.when(pl.col(f"{c}__parsed_tmp").is_not_null())
                                  .then(pl.col(f"{c}__parsed_tmp"))
                                  .otherwise(pl.col(c))
                                  .alias(c)
                            ).drop(f"{c}__parsed_tmp")
                        except Exception:
                            # ignore format if fails
                            continue
                else:
                    # best-effort parse using strptime with common formats (polars fallback limited)
                    # We'll try to cast to Date without specific fmt (some values will become null)
                    ldf = ldf.with_columns(pl.col(c).str.strptime(pl.Date, "%Y-%m-%d").alias(c).keep_name())
                    # if above produced all nulls, skip — collecting to check is expensive; keep as-is for now

    # ---- 5) Drop duplicates ----
    duplicates_removed = 0
    if cfg.drop_duplicates:
        # compute count before/after by materializing a small sample? We'll just do unique via lazy
        # Polars: .unique() returns unique rows; to get counts we must collect. For lightweight reporting, we'll collect at end.
        ldf = ldf.unique()

    # ---- 6) Collect & build report ----
    df_out = ldf.collect()
    rows_after = df_out.shape[0]
    cols_after = df_out.shape[1]

    # Build report fields similar to your data_cleaner.py
    rows_before = None
    try:
        # attempt to estimate rows from original raw CSV quickly by counting newlines
        rows_before = raw.count(b"\n")
    except Exception:
        rows_before = -1

    missing_by_col = {c: int(df_out[c].null_count()) for c in df_out.columns}
    dtypes = {c: str(t) for c, t in df_out.dtypes.items()}

    # duplicates_removed: we can approximate using rows_before (if available)
    if rows_before > 0:
        duplicates_removed = max(0, rows_before - rows_after)
    else:
        duplicates_removed = 0

    report = {
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "cols_after": int(cols_after),
        "missing_by_col": missing_by_col,
        "dtypes": dtypes,
        "duplicates_removed": duplicates_removed,
        "cached": False
    }

    # Write cleaned CSV to cache file and report file
    csv_bytes = df_out.to_csv().encode("utf-8")
    cached_csv.write_bytes(csv_bytes)
    cached_report.write_text(json.dumps(report), encoding="utf-8")

    # Return the cleaned CSV as attachment and include report header
    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="cleaned_{file.filename}"',
            "X-Clean-Report": json.dumps(report)
        }
    )
