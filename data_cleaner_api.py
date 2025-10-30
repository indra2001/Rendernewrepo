# data_cleaner_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io
import os

app = FastAPI(title="data_cleaner_api")

# Try to import your cleaning function from data_cleaner.py
try:
    from data_cleaner import clean_dataframe  # <- change if different
except Exception as e:
    clean_dataframe = None
    print("Warning: could not import clean_dataframe from data_cleaner.py:", e)

@app.get("/")
def root():
    return {"status": "ok", "service": "data_cleaner_api"}

@app.post("/clean")
async def clean_csv(file: UploadFile = File(...)):
    if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if clean_dataframe is None:
        # fallback: simple trim headers and drop duplicates
        df.columns = [c.strip() for c in df.columns.astype(str)]
        df = df.drop_duplicates()
    else:
        # call user-defined cleaning logic
        df = clean_dataframe(df)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=cleaned_{file.filename}"
    })

# Optional: JSON contract endpoint for programmatic use
@app.post("/clean/json")
async def clean_json(payload: dict):
    try:
        df = pd.DataFrame(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad JSON payload: {e}")
    if clean_dataframe is not None:
        df = clean_dataframe(df)
    else:
        df.columns = [c.strip() for c in df.columns.astype(str)]
        df = df.drop_duplicates()
    return JSONResponse(content={"rows": df.to_dict(orient="records")})
