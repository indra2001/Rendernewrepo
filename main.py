from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def root():
    return {"msg": "hello from Render!"}
