from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from PIL import Image
from io import BytesIO

import numpy as np
import torch
import sys
import os

# thêm path real-esrgan local
sys.path.append("real-esrgan")

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

DEVICE = torch.device("cpu")

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=32,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=DEVICE,
)

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    output, _ = upsampler.enhance(img_np, outscale=2)

    out_img = Image.fromarray(output)

    buffer = BytesIO()
    out_img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")