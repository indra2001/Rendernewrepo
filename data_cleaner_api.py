from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import polars as pl
import tempfile
import uuid
from data_cleaner import DataCleaner

app = FastAPI(title="Data Cleaning API", version="1.0")

# Temporary in-memory storage
TEMP_STORE = {}

@app.get("/")
def root():
    return {"message": "Data Cleaning API is running successfully!"}


@app.post("/clean")
async def clean_file(file: UploadFile = File(...)):
    """
    Endpoint to clean uploaded CSV data using Polars LazyFrame.
    Returns metadata and stores cleaned result temporarily in memory.
    """
    try:
        # Create a temporary file to save the upload
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_input.write(await file.read())
        temp_input.close()

        # Clean data
        cleaner = DataCleaner(temp_input.name)
        cleaned_df = cleaner.clean()

        # Store in memory
        file_id = str(uuid.uuid4())
        TEMP_STORE[file_id] = cleaned_df

        # Return success response
        return JSONResponse(
            content={
                "status": "success",
                "file_id": file_id,
                "rows": cleaned_df.shape[0],
                "columns": cleaned_df.shape[1],
                "message": "File cleaned and stored temporarily."
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/result/{file_id}")
def get_cleaned_result(file_id: str):
    """
    Retrieve cleaned dataset from temporary memory using file_id.
    """
    if file_id not in TEMP_STORE:
        return JSONResponse(status_code=404, content={"error": "File ID not found"})

    df = TEMP_STORE[file_id]
    return JSONResponse(
        content={
            "columns": df.columns,
            "preview": df.head(5).to_dicts()
        }
    )
