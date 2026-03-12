from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from routes.segmentation import segmentation_route
from routes.inpainting import hair_try__in_route
#from pydantic import BaseModel #Para la validacion de tipos de datos

app = FastAPI()
base_dir = Path(__file__).resolve().parent
ui_dir = base_dir / "ui"

app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="assets")


@app.get("/", include_in_schema=False)
async def root():
	return FileResponse(ui_dir / "index.html")


#class Image(BaseModel):
#    image: str
    
app.include_router(segmentation_route)
app.include_router(hair_try__in_route)
