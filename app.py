import os
import re
import time
import hashlib
import subprocess
import platform
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


APP_TITLE = "DocAnalyzer AI"

BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_OUTPUT_TOKENS = 700
TOP_K = 3


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


def get_secret_or_env(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


GEMINI_API_KEY = get_secret_or_env("GEMINI_API_KEY", "").strip()
MODEL_NAME = get_secret_or_env("GEMINI_MODEL", "gemini-2.5-flash")


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


if not GEMINI_API_KEY:
    st.error("Chưa có GEMINI_API_KEY. Hãy thêm key trong Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


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


def safe_filename(filename):
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = re.sub(r"[^a-zA-Z0-9_.\-() ]", "_", filename)
    return filename


def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


def save_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    name = safe_filename(uploaded_file.name)
    save_path = UPLOADS_DIR / f"{file_hash[:10]}_{name}"

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
                    chunk = para[start:end].strip()

                    if chunk:
                        chunks.append(chunk)

                    if end >= len(para):
                        break

                    start = max(end - overlap, start + 1)
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_pdf_text(pdf_path):
    nodes = []
    counter = 0

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

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
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

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


def extract_markdown_headers(md_text):
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


def add_text_to_markdown_nodes(nodes, lines):
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
            "text": node.get("text", ""),
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
    try:
        md_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    flat_nodes, lines = extract_markdown_headers(md_text)

    if not flat_nodes:
        return extract_txt_nodes(file_path)

    flat_nodes = add_text_to_markdown_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)

    return flatten_tree(tree, source_file=file_path.name)


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

        if file_path.exists():
            all_nodes.extend(parse_document(file_path))

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

    for index, node in enumerate(selected_nodes, start=1):
        context_parts.append(
            f"[ĐOẠN {index}]\n"
            f"Nguồn: {node.get('path', '')}\n"
            f"Nội dung:\n{node.get('text', '')}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Nếu thiếu thông tin, nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
- Giải thích dễ hiểu cho người mới.
- Không nhắc tên SOURCE.
- Không hiển thị [SOURCE 1], [SOURCE 2], [SOURCE 3].
- Trả lời ngắn gọn, tối đa 500 từ.

FORMAT:
- Dùng markdown.
- Có tiêu đề nhỏ.
- Có bullet point.
- Có phần "Ví dụ minh họa".
- Có phần "Kết luận ngắn".

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

            if answer:
                return answer

            raise RuntimeError("Gemini không trả về nội dung.")

        except Exception as e:
            error_text = str(e)

            if "503" in error_text or "overloaded" in error_text.lower():
                wait_time = 2 * (attempt + 1)
                st.warning(f"Gemini đang quá tải. Đang thử lại sau {wait_time}s...")
                time.sleep(wait_time)
                continue

            if "429" in error_text:
                raise RuntimeError("Gemini đã hết quota hoặc bị giới hạn tốc độ.")

            if "403" in error_text:
                raise RuntimeError("API key hoặc project Gemini chưa có quyền truy cập model này.")

            raise RuntimeError(error_text)

    raise RuntimeError("Gemini quá tải sau nhiều lần thử.")


def get_default_realesrgan_path():
    system_name = platform.system().lower()

    # Nếu Windows, trả về .exe
    if "windows" in system_name:
        # Cập nhật tên folder Windows của bạn
        return str(BASE_DIR / "realesrgan-ncnn-vulkan-v0.2.0-windows/realesrgan-ncnn-vulkan-v0.2.0-windows/realesrgan-ncnn-vulkan.exe")

    # Các OS khác (Linux/Mac)
    candidates = [
        BASE_DIR / "realesrgan-ncnn-vulkan-v0.2.0-ubuntu" / "realesrgan-ncnn-vulkan",
        BASE_DIR / "realesrgan-ncnn-vulkan",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return str(candidates[0])


with st.sidebar:
    st.title("📚 Lịch sử")

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
            safe_q = q.replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"""
<div class="history-box">
<b>{i}.</b> {safe_q}
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
st.caption("Chat với PDF, Markdown và TXT bằng Gemini")

uploaded_files = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=["pdf", "md", "markdown", "txt"],
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

    with st.spinner("📖 Đang đọc tài liệu..."):
        saved_paths = []

        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            saved_paths.append(str(file_path))

        all_nodes = parse_uploaded_files_cached(tuple(saved_paths))

    if not all_nodes:
        st.error("Không đọc được nội dung tài liệu. Nếu PDF là dạng scan ảnh, cần OCR.")
        st.stop()

    selected_nodes = select_relevant_nodes(question, all_nodes, TOP_K)
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


st.markdown("---")
st.header("🖼️ Làm nét / Upscale ảnh bằng Real-ESRGAN")
st.caption(
    "Chức năng này dùng Real-ESRGAN NCNN Vulkan chạy local. "
    "Trên Streamlit Cloud có thể không chạy được nếu thiếu Vulkan/GPU."
)

with st.expander("⚙️ Cấu hình Real-ESRGAN", expanded=False):
    realesrgan_exe = st.text_input(
        "Đường dẫn file realesrgan-ncnn-vulkan",
        value=get_default_realesrgan_path(),
        help=(
            "Windows: đường dẫn tới realesrgan-ncnn-vulkan.exe. "
            "Linux/Streamlit Cloud: đường dẫn tới file realesrgan-ncnn-vulkan."
        ),
    )

    model_name = st.selectbox(
        "Model",
        options=[
            "realesrgan-x4plus",
            "realesrnet-x4plus",
            "realesrgan-x4plus-anime",
            "realesr-animevideov3",
            "realesr-general-x4v3",
        ],
        index=0,
    )

    output_format = st.selectbox(
        "Định dạng output",
        options=["png", "jpg", "webp"],
        index=0,
    )

    scale = st.selectbox(
        "Tỉ lệ upscale",
        options=[2, 3, 4],
        index=2,
    )

    tile_size = st.number_input(
        "Tile size",
        min_value=0,
        max_value=1024,
        value=0,
        step=32,
        help="0 = tự động.",
    )

    show_debug = st.checkbox("Hiện debug Real-ESRGAN", value=False)


upscale_file = st.file_uploader(
    "📤 Tải ảnh cần làm nét",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    key="realesrgan_image_uploader",
)

col_preview_1, col_preview_2 = st.columns(2)

if upscale_file is not None:
    image_bytes = upscale_file.getvalue()
    safe_image_name = safe_filename(upscale_file.name)

    with col_preview_1:
        st.subheader("Ảnh gốc")
        st.image(image_bytes, use_container_width=True)

    run_upscale = st.button("🚀 Làm nét ảnh bằng Real-ESRGAN")

    if run_upscale:
        exe_path = Path(realesrgan_exe.strip().strip('"'))

        if show_debug:
            st.write("BASE_DIR:", BASE_DIR)
            st.write("Real-ESRGAN path:", exe_path)
            st.write("Exists:", exe_path.exists())
            st.write("Is file:", exe_path.is_file())
            st.write("Platform:", platform.system())

        if not exe_path.exists():
            st.error("Không tìm thấy file Real-ESRGAN. Hãy kiểm tra lại đường dẫn hoặc đã push file lên GitHub chưa.")
            st.stop()

        if not exe_path.is_file():
            st.error("Đường dẫn Real-ESRGAN không phải là file executable.")
            st.stop()

        if platform.system().lower() != "windows":
            try:
                exe_path.chmod(0o755)
            except Exception:
                pass

        job_id = hashlib.md5(image_bytes + str(time.time()).encode()).hexdigest()[:10]
        job_dir = UPLOADS_DIR / f"realesrgan_job_{job_id}"
        job_dir.mkdir(exist_ok=True)

        input_path = job_dir / safe_image_name
        output_path = job_dir / f"upscaled_{Path(safe_image_name).stem}.{output_format}"

        input_path.write_bytes(image_bytes)

        cmd = [
            str(exe_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model_name,
            "-s", str(scale),
            "-f", output_format,
        ]

        if tile_size and tile_size > 0:
            cmd.extend(["-t", str(tile_size)])

        try:
            with st.spinner("Real-ESRGAN đang xử lý ảnh..."):
                completed = subprocess.run(
                    cmd,
                    cwd=str(exe_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            if completed.returncode != 0:
                st.error("Real-ESRGAN chạy lỗi.")
                st.code(completed.stderr or completed.stdout or "Không có log lỗi.")
                st.stop()

            if not output_path.exists():
                st.error("Không tìm thấy file output sau khi xử lý.")
                st.stop()

            output_bytes = output_path.read_bytes()

            with col_preview_2:
                st.subheader("Ảnh sau khi làm nét")
                st.image(output_bytes, use_container_width=True)

            mime_type = "image/jpeg" if output_format == "jpg" else f"image/{output_format}"

            st.success("Xử lý xong!")
            st.download_button(
                "⬇️ Tải ảnh đã làm nét",
                data=output_bytes,
                file_name=output_path.name,
                mime=mime_type,
            )

        except subprocess.TimeoutExpired:
            st.error("Real-ESRGAN xử lý quá lâu. Hãy thử ảnh nhỏ hơn hoặc giảm tile size.")
        except Exception as e:
            st.error(f"Lỗi khi chạy Real-ESRGAN: {e}")
else:
    st.info("Tải một ảnh lên để dùng chức năng Real-ESRGAN.")