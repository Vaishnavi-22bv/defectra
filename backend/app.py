from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Defectra API", version="0.1.0")


@app.get("/")
def read_root() -> dict:
    return {"message": "Defectra backend is running"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    return JSONResponse(
        {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "Upload endpoint ready for defect detection integration",
        }
    )
