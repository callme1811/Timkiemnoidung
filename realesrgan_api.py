from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse

from PIL import Image
from io import BytesIO

import numpy as np
import torch
import sys
import traceback


# Dùng repo/package local nếu có
sys.path.append("real-esrgan")

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


app = FastAPI()

DEVICE = torch.device("cpu")


def load_upsampler():
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

    return upsampler


upsampler = load_upsampler()


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "RealESRGAN API",
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Giảm kích thước ảnh đầu vào để tránh Render Free bị crash RAM
        max_side = 900
        img.thumbnail((max_side, max_side))

        img_np = np.array(img)

        # outscale=2 nhẹ hơn outscale=4
        output, _ = upsampler.enhance(img_np, outscale=2)

        out_img = Image.fromarray(output)

        buffer = BytesIO()
        out_img.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )