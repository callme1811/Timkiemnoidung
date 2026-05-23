import re
import time

import google.generativeai as genai
from PIL import Image


def ask_gemini_vision(question: str, image_paths: list[str], model_name: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    images = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except Exception:
            pass

    if not images:
        raise Exception("Không mở được các ảnh yêu cầu để Gemini Vision phân tích.")

    prompt = f"""
Bạn là AI hỗ trợ phân tích ảnh/tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên ảnh được cung cấp.
- Nếu ảnh là ECG/điện tim, chỉ mô tả những gì quan sát được.
- Không tự ý đưa ra chẩn đoán y khoa chắc chắn.
- Không khẳng định bệnh lý chắc chắn khi không đủ dữ kiện.
- Nếu ảnh mờ hoặc thiếu thông tin, nói rõ hạn chế của ảnh.
- Trả lời trực quan bằng tiếng Việt ngắn gọn.

CÂU HỎI:
{question}
"""

    response = model.generate_content([prompt, *images])
    answer = getattr(response, "text", "") or ""

    if not answer.strip():
        raise Exception("Gemini Vision không trả về nội dung trả lời nào.")

    return answer.strip()


def clean_answer(text: str) -> str:
    text = re.sub(r"\[SOURCE\s*\d+\]", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()


def build_prompt(question: str, context_text: str) -> str:
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Nếu thiếu thông tin để làm rõ câu hỏi, nói rõ tài liệu không cung cấp.
- Không hiển thị SOURCE.
- Trả lời ngắn gọn, tối đa 500 từ.

FORMAT:
- Dùng markdown.
- Có tiêu đề nhỏ.
- Có bullet point.
- Có phần "Kết luận ngắn".

Nếu tài liệu không có thông tin để trả lời:
"Tôi không tìm thấy thông tin này trong tài liệu."

QUESTION:
{question}

CONTEXT:
{context_text}
"""


def ask_gemini(
    question: str,
    context_text: str,
    model_name: str,
    temp: float,
    max_tokens: int,
    api_key: str,
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = build_prompt(question, context_text)

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                },
                stream=False,
            )

            answer = getattr(response, "text", "") or ""
            answer = clean_answer(answer)

            if answer.strip():
                return answer

            raise Exception("Gemini không trả về nội dung.")

        except Exception as e:
            error_text = str(e)
            if "503" in error_text or "overloaded" in error_text.lower():
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)
                continue
            if "429" in error_text:
                raise Exception("Gemini đã hết quota hoặc bị giới hạn tốc độ (Rate Limit 429).")
            if "403" in error_text:
                raise Exception("API key hoặc project Gemini chưa có quyền truy cập model này.")
            raise e

    raise Exception("Mô hình Gemini phản hồi quá tải sau nhiều lần thử lại.")
