from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from io import BytesIO

import numpy as np
import torch
import sys
import traceback
from pathlib import Path


# Nếu có folder real-esrgan local thì cho Python nhận
BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR / "real-esrgan"))

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


app = FastAPI(title="RealESRGAN API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = BASE_DIR / "weights" / "RealESRGAN_x4plus.pth"


def load_upsampler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

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
        model_path=str(MODEL_PATH),
        model=model,
        tile=8,
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
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
    }


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "File ảnh rỗng."},
            )

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Giảm ảnh đầu vào để tránh timeout/RAM yếu
        max_side = 400
        img.thumbnail((max_side, max_side))

        img_np = np.array(img)

        # outscale=1 nhẹ nhất, ổn định hơn trên CPU/Free
        output, _ = upsampler.enhance(img_np, outscale=1)

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