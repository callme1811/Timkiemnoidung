import re
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image


# =========================
# CONFIG
# =========================

APP_TITLE = "DocAnalyzer AI"
BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()

MAX_OUTPUT_TOKENS = 700
TOP_K = 3


# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY trong Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""


# =========================
# OPENCV IMAGE HELPERS
# =========================

def get_mime_type(path):
    suffix = Path(path).suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    return "application/octet-stream"


def enhance_image_opencv(image_path, mode="balanced"):
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

        output_path = image_path.parent / f"opencv_{mode}_{image_path.stem}.png"
        saved = cv2.imwrite(str(output_path), processed)

        if not saved:
            return str(image_path), False, "Không lưu được ảnh đã xử lý."

        return str(output_path), True, f"Đã xử lý ảnh bằng OpenCV. Chế độ: {mode}."

    except Exception as e:
        return str(image_path), False, f"Lỗi xử lý ảnh bằng OpenCV: {e}"


def opencv_fast_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    enhanced = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])

    return cv2.filter2D(enhanced, -1, kernel)


def opencv_balanced_enhance(img):
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=6,
        hColor=6,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    enhanced = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)


def opencv_high_quality_enhance(img):
    height, width = img.shape[:2]

    scale = 1.5
    resized = cv2.resize(
        img,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        resized,
        None,
        h=8,
        hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    enhanced = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.7, gaussian, -0.7, 0)


# =========================
# GEMINI
# =========================

def ask_gemini_vision(question, image_paths):
    model = genai.GenerativeModel(MODEL_NAME)
    images = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except Exception:
            pass

    if not images:
        raise Exception("Không mở được ảnh để Gemini Vision phân tích.")

    prompt = f"""
Bạn là AI hỗ trợ phân tích ảnh/tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên ảnh được cung cấp.
- Nếu ảnh là ECG/điện tim, chỉ mô tả những gì quan sát được.
- Không chẩn đoán y khoa chắc chắn.
- Không khẳng định bệnh lý chắc chắn.
- Nếu ảnh mờ hoặc thiếu thông tin, nói rõ hạn chế.
- Trả lời bằng tiếng Việt, ngắn gọn.

CÂU HỎI:
{question}
"""

    response = model.generate_content([prompt, *images])
    answer = getattr(response, "text", "") or ""

    if not answer.strip():
        raise Exception("Gemini Vision không trả về nội dung.")

    return answer.strip()


def clean_answer(text):
    text = re.sub(r"\[SOURCE\s*\d+\]", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()


def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Nếu thiếu thông tin, nói rõ tài liệu không cung cấp.
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


# =========================
# FILE PARSING
# =========================

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

                current_chunk = ""
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
    text = file_path.read_text(encoding="utf-8", errors="ignore")
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
        end = nodes[i + 1]["line_num"] - 1 if i + 1 < len(nodes) else len(lines)

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

        stack.append({"level": node["level"], "node": tree_node})

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
    md_text = file_path.read_text(encoding="utf-8", errors="ignore")
    flat_nodes, lines = extract_markdown_nodes(md_text)

    if not flat_nodes:
        return extract_txt_nodes(file_path)

    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)
    return flatten_tree(nodes=tree, source_file=file_path.name)


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
        key=lambda n: keyword_score(question, n.get("text", ""), n.get("title", "")),
        reverse=True,
    )

    selected = [
        n for n in ranked
        if keyword_score(question, n.get("text", ""), n.get("title", "")) > 0
    ]

    if not selected:
        selected = ranked

    return selected[:top_k]


def build_context(selected_nodes):
    return "\n\n".join([n.get("text", "") for n in selected_nodes])


# =========================
# CSS
# =========================

st.markdown(
    """
<style>
.block-container{
    max-width:1200px;
    padding-top:32px;
}

.stButton button{
    width:100%;
    height:48px;
    border-radius:14px;
    font-size:15px;
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
    padding:12px;
    border-radius:12px;
    margin-bottom:8px;
    background:#111827;
    border:1px solid #374151;
    font-size:14px;
}

.small-muted{
    opacity:0.75;
    font-size:13px;
}

img{
    border-radius:10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("📚 Lịch sử")

    st.success("Xử lý ảnh: OpenCV local")
    st.caption("Không dùng Colab, không dùng ngrok, không dùng RealESRGAN API.")

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.session_state.question_history = []
        st.session_state.last_context = ""
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    history_list = st.session_state.get("question_history", [])

    if history_list:
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


# =========================
# MAIN UI
# =========================

st.title("📄 DocAnalyzer AI")
st.caption("Chat với PDF, Markdown, TXT và ảnh bằng Gemini + OpenCV local")

with st.expander("⚙️ Cài đặt xử lý ảnh", expanded=True):
    col_a, col_b = st.columns([2, 1])

    with col_a:
        process_mode_label = st.radio(
            "Chọn chế độ xử lý ảnh",
            ["Nhanh", "Cân bằng", "Rõ nét cao"],
            horizontal=True,
        )

    with col_b:
        enable_enhance = st.toggle(
            "Dùng OpenCV Enhance",
            value=True,
        )

    mode_map = {
        "Nhanh": "fast",
        "Cân bằng": "balanced",
        "Rõ nét cao": "high_quality",
    }

    process_mode = mode_map[process_mode_label]

    if enable_enhance:
        st.info(f"Đang bật xử lý ảnh bằng OpenCV. Chế độ: {process_mode_label}")
    else:
        st.warning("Đang tắt xử lý ảnh. Gemini sẽ dùng ảnh gốc.")

uploaded_files = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Hỏi nội dung tài liệu...")


# =========================
# MAIN FLOW
# =========================

if question:
    if not uploaded_files:
        st.warning("Vui lòng upload tài liệu.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})

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

                if enable_enhance:
                    processed_path, ok, message = enhance_image_opencv(
                        image_path=file_path,
                        mode=process_mode,
                    )

                    if not ok:
                        processed_path = original_path
                else:
                    processed_path = original_path
                    ok = True
                    message = "Đang dùng ảnh gốc."

                st.subheader("🖼️ Kết quả xử lý ảnh")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Ảnh gốc")
                    st.image(original_path, width=600)

                with col2:
                    if enable_enhance:
                        st.markdown("### Ảnh đã xử lý bằng OpenCV")
                    else:
                        st.markdown("### Ảnh đang dùng")

                    st.image(processed_path, width=600)

                    try:
                        with open(processed_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Tải ảnh đã xử lý",
                                data=f,
                                file_name=Path(processed_path).name,
                                mime=get_mime_type(processed_path),
                            )
                    except Exception:
                        st.warning("Không thể tạo nút tải ảnh.")

                if ok:
                    st.success(message)
                else:
                    st.warning(message)
                    st.info("OpenCV lỗi nên Gemini sẽ dùng ảnh gốc để phân tích.")

                image_paths_for_vision.append(processed_path)

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
