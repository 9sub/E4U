from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/segmentation/image/")
def send_segmentation_image():
    result_image_path = "/workspace/AI/CV/teeth_segmentation/output/teethnum.png"

    return FileResponse(result_image_path, media_type="image/png", filename="segmentation_result.png")