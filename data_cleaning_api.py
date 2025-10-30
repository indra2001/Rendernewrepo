from fastapi import FastAPI, UploadFile, File, HTTPException
import polars as pl
import tempfile
import os

app = FastAPI(title="Data Cleaning API")

@app.post("/clean")
async def clean_data(file: UploadFile = File(...)):
    """
    Endpoint to clean uploaded CSV data:
    - Trim column headers
    - Remove duplicate rows
    - Coerce data types
    """

    # ✅ Step 1: Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # ✅ Step 2: Save the uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {e}")

    try:
        # ✅ Step 3: Load CSV using Polars LazyFrame (for performance)
        df_lazy = pl.scan_csv(tmp_path)

        # ✅ Step 4: Apply cleaning transformations
        df_lazy = (
            df_lazy.rename({col: col.strip().lower().replace(" ", "_") for col in df_lazy.columns})
            .unique()  # remove duplicates
        )

        # ✅ Step 5: Collect cleaned data into a Polars DataFrame
        df = df_lazy.collect()

        # ✅ Step 6: Type coercion (try converting to numeric/date where possible)
        df = df.with_columns([
            pl.col(col).cast(pl.Float64, strict=False).alias(col)
            if df[col].dtype == pl.Utf8 and df[col].str.contains(r"^\d+(\.\d+)?$").any()
            else df[col]
            for col in df.columns
        ])

        # ✅ Step 7: Create a result summary
        result = {
            "status": "success",
            "rows_cleaned": df.height,
            "columns": df.columns,
            "message": "Dataset cleaned successfully"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {e}")

    finally:
        # ✅ Step 8: Delete temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
