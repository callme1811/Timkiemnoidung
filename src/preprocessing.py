import re
from pathlib import Path

import cv2


def split_text(text: str, chunk_size: int = 1000, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []

    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{para}" if current_chunk else para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = max(end - overlap, start + 1)
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def opencv_fast_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    blur = cv2.GaussianBlur(l_clahe, (5, 5), 0)
    l_sharpen = cv2.addWeighted(l_clahe, 1.3, blur, -0.3, 0)

    enhanced = cv2.merge((l_sharpen, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def opencv_balanced_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    l_smooth = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=15, sigmaSpace=15)

    blur = cv2.GaussianBlur(l_smooth, (0, 0), 0.6)
    l_sharpen = cv2.addWeighted(l_smooth, 1.4, blur, -0.4, 0)

    enhanced = cv2.merge((l_sharpen, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def opencv_high_quality_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    l_smooth = cv2.bilateralFilter(l_clahe, d=7, sigmaColor=25, sigmaSpace=25)

    blur = cv2.GaussianBlur(l_smooth, (0, 0), 1.0)
    l_sharpen = cv2.addWeighted(l_smooth, 1.5, blur, -0.5, 0)

    enhanced = cv2.merge((l_sharpen, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def enhance_image_opencv(image_path, mode: str = "balanced"):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))

    if img is None:
        return str(image_path), False, "Không đọc được ảnh bằng OpenCV."

    try:
        if mode == "fast":
            processed = opencv_fast_enhance(img)
        elif mode == "high_quality":
            processed = opencv_high_quality_enhance(img)
        else:
            processed = opencv_balanced_enhance(img)

        output_path = image_path.parent / f"opencv_{mode}_{image_path.name}"
        if output_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            output_path = output_path.with_suffix(".png")

        saved = cv2.imwrite(str(output_path), processed)
        if not saved:
            return str(image_path), False, "Không lưu được ảnh đã xử lý."

        return str(output_path), True, f"Đã xử lý ảnh bằng OpenCV. Chế độ: {mode}."
    except Exception as e:
        return str(image_path), False, f"Lỗi xử lý ảnh bằng OpenCV: {e}"
