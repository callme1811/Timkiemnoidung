import os
import re
import time
import hashlib
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image
import numpy as np

APP_TITLE = "DocAnalyzer AI"

BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 700
TOP_K = 3

USE_REALESRGAN = os.getenv("USE_REALESRGAN", "false").lower() == "true"
REALESRGAN_DEVICE = os.getenv("REALESRGAN_DEVICE", "cpu")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API key không tìm thấy trong secrets. Vui lòng thêm GEMINI_API_KEY vào Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

from realesrgan import RealESRGAN
@st.cache_resource(show_spinner=False)
def load_realesrgan_model():
    import sys
    import torchvision.transforms.functional as functional

    sys.modules["torchvision.transforms.functional_tensor"] = functional

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

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
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=REALESRGAN_DEVICE,
    )

    return upsampler


def upscale_ecg_image(image_path):
    if not USE_REALESRGAN:
        return str(image_path), False, "RealESRGAN đang tắt."

    try:
        sr_model = load_realesrgan_model()
        img = Image.open(image_path).convert("RGB")
        sr_img, _ = sr_model.enhance(img)

        out_path = Path(image_path).parent / f"upscaled_{Path(image_path).name}"
        sr_img.save(out_path)

        return str(out_path), True, "Đã upscale bằng RealESRGAN."

    except Exception as e:
        return str(image_path), False, f"Không chạy được RealESRGAN, dùng ảnh gốc. Lỗi: {e}"


def ask_gemini_vision(question, image_paths):
    model = genai.GenerativeModel(MODEL_NAME)

    images = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("RGB")
            images.append(img)
        except Exception:
            continue

    if not images:
        raise Exception("Không mở được ảnh để Gemini Vision phân tích.")

    prompt = f"""
Bạn là AI hỗ trợ phân tích ảnh/tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên ảnh được cung cấp.
- Nếu ảnh là ECG/điện tim, hãy mô tả những gì có thể quan sát được.
- Không chẩn đoán y khoa chắc chắn.
- Nếu ảnh mờ hoặc thiếu thông tin, nói rõ hạn chế.
- Trả lời bằng tiếng Việt.
- Dùng markdown ngắn gọn.

CÂU HỎI:
{question}
"""

    response = model.generate_content([prompt, *images])
    answer = getattr(response, "text", "") or ""

    if not answer.strip():
        raise Exception("Gemini Vision không trả về nội dung.")

    return answer.strip()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""


def clean_answer(text):
    text = re.sub(r"\[SOURCE\s*\d+\]", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()


def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


def save_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    save_path = UPLOADS_DIR / f"{file_hash[:10]}_{safe_name}"

    if not save_path.exists():
        save_path.write_bytes(file_bytes)

    return save_path


def split_text(text, chunk_size=1000, overlap=120):
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
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))

    nodes = []
    counter = 0

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks, start=1):
            counter += 1

            nodes.append({
                "node_id": str(counter).zfill(4),
                "title": f"Page {page_index}",
                "path": f"{pdf_path.name} > Page {page_index} > Chunk {chunk_index}",
                "source_file": pdf_path.name,
                "page": page_index,
                "chunk": chunk_index,
                "text": chunk,
            })

    return nodes


def extract_txt_nodes(file_path):
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    chunks = split_text(text)
    nodes = []

    for i, chunk in enumerate(chunks, start=1):
        nodes.append({
            "node_id": str(i).zfill(4),
            "title": file_path.name,
            "path": f"{file_path.name} > Chunk {i}",
            "source_file": file_path.name,
            "page": None,
            "chunk": i,
            "text": chunk,
        })

    return nodes


def extract_markdown_nodes(md_text):
    lines = md_text.splitlines()
    nodes = []
    in_code_block = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

        if match:
            nodes.append({
                "title": match.group(2).strip(),
                "level": len(match.group(1)),
                "line_num": i,
            })

    return nodes, lines


def add_text_to_nodes(nodes, lines):
    for i, node in enumerate(nodes):
        start = node["line_num"]

        end = (
            nodes[i + 1]["line_num"] - 1
            if i + 1 < len(nodes)
            else len(lines)
        )

        node["start_line"] = start
        node["end_line"] = end
        node["text"] = "\n".join(lines[start - 1:end]).strip()

    return nodes


def build_tree(flat_nodes):
    root = []
    stack = []
    counter = 0

    for node in flat_nodes:
        counter += 1

        tree_node = {
            "title": node["title"],
            "node_id": str(counter).zfill(4),
            "line_num": node["line_num"],
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "text": node["text"],
            "nodes": [],
        }

        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1]["node"]["nodes"].append(tree_node)
        else:
            root.append(tree_node)

        stack.append({
            "level": node["level"],
            "node": tree_node,
        })

    return root


def flatten_tree(nodes, parent_path="", source_file=""):
    result = []

    for node in nodes:
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title

        text_chunks = split_text(node.get("text", ""))

        for chunk_index, chunk in enumerate(text_chunks, start=1):
            result.append({
                "node_id": node.get("node_id", ""),
                "title": title,
                "path": f"{source_file} > {path} > Chunk {chunk_index}",
                "source_file": source_file,
                "page": None,
                "chunk": chunk_index,
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
                "text": chunk,
            })

        children = node.get("nodes", [])

        if children:
            result.extend(flatten_tree(children, path, source_file))

    return result


def parse_markdown(file_path):
    md_text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    flat_nodes, lines = extract_markdown_nodes(md_text)

    if not flat_nodes:
        return extract_txt_nodes(file_path)

    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)

    return flatten_tree(
        nodes=tree,
        source_file=file_path.name,
    )


def parse_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(file_path)

    if suffix in [".md", ".markdown"]:
        return parse_markdown(file_path)

    if suffix == ".txt":
        return extract_txt_nodes(file_path)

    return []


@st.cache_data(show_spinner=False)
def parse_uploaded_files_cached(file_paths):
    all_nodes = []

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        nodes = parse_document(file_path)
        all_nodes.extend(nodes)

    return all_nodes


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_score(question, text, title=""):
    q_words = re.findall(r"\w+", normalize_text(question))
    content = normalize_text(f"{title} {text}")

    if not q_words:
        return 0

    score = 0

    for word in q_words:
        if word in content:
            score += 2

    question_norm = normalize_text(question)

    if question_norm and question_norm in content:
        score += 5

    return score


def select_relevant_nodes(question, nodes, top_k=TOP_K):
    if not nodes:
        return []

    ranked = sorted(
        nodes,
        key=lambda n: keyword_score(
            question,
            n.get("text", ""),
            n.get("title", ""),
        ),
        reverse=True,
    )

    selected = [
        n for n in ranked
        if keyword_score(
            question,
            n.get("text", ""),
            n.get("title", ""),
        ) > 0
    ]

    if not selected:
        selected = ranked

    return selected[:top_k]


def build_context(selected_nodes):
    context_parts = []

    for n in selected_nodes:
        context_parts.append(n.get("text", ""))

    return "\n\n".join(context_parts)


def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Giải thích dễ hiểu cho người mới.
- Nếu thiếu thông tin, nói rõ tài liệu không cung cấp.
- Không hiển thị SOURCE.
- Không nhắc [SOURCE 1], [SOURCE 2], [SOURCE 3].
- Trả lời ngắn gọn, tối đa 500 từ.

FORMAT:
- Dùng markdown.
- Có tiêu đề nhỏ.
- Có bullet point.
- Có phần "Ví dụ minh họa".
- Có phần "Kết luận ngắn".

Nếu tài liệu không có thông tin để trả lời:
"Tôi không tìm thấy thông tin này trong tài liệu."

QUESTION:
{question}

CONTEXT:
{context_text}
"""


def ask_gemini(question, context_text):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = build_prompt(question, context_text)

    retries = 3

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.25,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
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
                st.warning(f"Gemini đang quá tải. Đang thử lại sau {wait_time}s...")
                time.sleep(wait_time)
                continue

            if "429" in error_text:
                raise Exception("Gemini đã hết quota hoặc bị giới hạn tốc độ.")

            if "403" in error_text:
                raise Exception("API key hoặc project Gemini chưa có quyền truy cập model này.")

            raise e

    raise Exception("Gemini quá tải sau nhiều lần thử.")


st.markdown(
    """
<style>
.block-container{
    max-width:1200px;
    padding-top:25px;
}

.stButton button{
    width:100%;
    height:50px;
    border-radius:14px;
    font-size:16px;
    font-weight:700;
}

.answer-box{
    padding:24px;
    border-radius:18px;
    background:#111827;
    border:1px solid #374151;
    margin-top:10px;
    line-height:1.7;
}

.history-box{
    padding:10px;
    border-radius:12px;
    margin-bottom:8px;
    background:#111827;
    border:1px solid #374151;
    font-size:14px;
}
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("📚 Lịch sử")

    if USE_REALESRGAN:
        st.success(f"RealESRGAN: Đang bật ({REALESRGAN_DEVICE})")
    else:
        st.info("RealESRGAN: Đang tắt")

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.session_state.question_history = []
        st.session_state.last_context = ""
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")

    history_list = st.session_state.get("question_history", [])

    if len(history_list) > 0:
        for i, q in enumerate(history_list[::-1], start=1):
            st.markdown(
                f"""
<div class="history-box">
<b>{i}.</b> {q}
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
<div class="history-box">
Chưa có câu hỏi nào
</div>
""",
            unsafe_allow_html=True,
        )


st.title("📄 DocAnalyzer AI")
st.caption("Chat với PDF, Markdown, TXT và ảnh bằng Gemini 2.5")

uploaded_files = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Hỏi nội dung tài liệu...")


if question:
    if not uploaded_files:
        st.warning("Vui lòng upload tài liệu.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    if question not in st.session_state.question_history:
        st.session_state.question_history.append(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("📖 Đang xử lý tài liệu..."):
        saved_paths = []
        image_paths_for_vision = []

        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            suffix = Path(file_path).suffix.lower()

            if suffix in [".png", ".jpg", ".jpeg"]:
                original_path = str(file_path)
                processed_path, ok, message = upscale_ecg_image(file_path)

                st.subheader("🖼️ Kết quả RealESRGAN")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Ảnh gốc")
                    st.image(original_path, use_column_width=True)

                with col2:
                    st.markdown("### Ảnh sau upscale")
                    st.image(processed_path, use_column_width=True)

                if ok:
                    st.success(message)
                else:
                    st.warning(message)

                image_paths_for_vision.append(processed_path)
                saved_paths.append(processed_path)
            else:
                saved_paths.append(str(file_path))

        all_nodes = parse_uploaded_files_cached(tuple(saved_paths))

    if not all_nodes:
        if image_paths_for_vision:
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🤖 Gemini Vision đang phân tích ảnh..."):
                        vision_answer = ask_gemini_vision(question, image_paths_for_vision)

                    st.markdown(
                        f"""
<div class="answer-box">

{vision_answer}

</div>
""",
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": vision_answer,
                    })

                except Exception as e:
                    st.error(f"Lỗi Gemini Vision: {e}")

            st.stop()

        st.error("Không đọc được nội dung tài liệu.")
        st.stop()

    selected_nodes = select_relevant_nodes(
        question=question,
        nodes=all_nodes,
        top_k=TOP_K,
    )

    context_text = build_context(selected_nodes)
    st.session_state.last_context = context_text

    with st.chat_message("assistant"):
        try:
            with st.spinner("🤖 Gemini đang trả lời..."):
                answer = ask_gemini(question, context_text)

            st.markdown(
                f"""
<div class="answer-box">

{answer}

</div>
""",
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as e:
            st.error(f"Lỗi Gemini: {e}")
    with st.expander("📚 Xem context đã dùng"):
        st.code(context_text)