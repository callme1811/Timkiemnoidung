import re
import time
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image


# =========================
# CONFIG & PATHS
# =========================

APP_TITLE = "DocAnalyzer AI"
BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


# =========================
# PAGE SETUP & STYLING
# =========================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

# Custom Premium CSS for Outfit Typography & Glassmorphism Chat Bubbles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom Glassmorphism Chat Bubbles styling */
    .chat-bubble-user {
        padding: 16px 22px;
        border-radius: 20px 20px 0px 20px;
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
        line-height: 1.6;
    }

    .chat-bubble-assistant {
        padding: 22px 26px;
        border-radius: 20px 20px 20px 0px;
        background: #111827;
        border: 1px solid #374151;
        color: #F3F4F6;
        margin-top: 10px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        line-height: 1.7;
    }

    .answer-header {
        font-weight: 700;
        font-size: 1.15rem;
        color: #60A5FA;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .history-box {
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 8px;
        background: #1F2937;
        border: 1px solid #374151;
        font-size: 14px;
        color: #D1D5DB;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .history-box:hover {
        border-color: #3B82F6;
        color: #F3F4F6;
        background: #111827;
    }

    img {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Custom premium buttons */
    .stButton button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        height: 44px !important;
    }

    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# SESSION STATE INITIALIZATION
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "processed_images" not in st.session_state:
    # Key format: f"{image_path}_{mode}" -> {"processed_path": str, "ok": bool, "message": str}
    st.session_state.processed_images = {}

if "pdf_extracted_images" not in st.session_state:
    # Persistent list of image paths extracted from scanned PDF pages
    st.session_state.pdf_extracted_images = []

if "parsed_nodes" not in st.session_state:
    # Flat list of extracted text chunks/nodes
    st.session_state.parsed_nodes = []

if "node_embeddings" not in st.session_state:
    # Key format: node_id -> embedding list
    st.session_state.node_embeddings = {}

if "files_hash" not in st.session_state:
    # Track uploaded files to reset states on document changes
    st.session_state.files_hash = ""


# =========================
# SIDEBAR CONFIGURATION
# =========================

with st.sidebar:
    st.title("📚 Cấu hình")

    # API Key Settings
    api_key_env = st.secrets.get("GEMINI_API_KEY", "").strip()
    api_key_input = st.text_input("🔑 Gemini API Key", value=api_key_env, type="password")

    if not api_key_input:
        st.error("Thiếu GEMINI_API_KEY. Vui lòng nhập vào ô trên hoặc thiết lập trong Streamlit Secrets.")
        st.stop()

    genai.configure(api_key=api_key_input)

    st.markdown("---")
    st.subheader("🤖 Cấu hình AI & Model")

    # Model Selection
    model_choice = st.selectbox(
        "Chọn mô hình Gemini",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0
    )

    # Sliders for LLM
    temperature = st.slider("Độ sáng tạo (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_output_tokens = st.slider("Giới hạn từ trả lời (Tokens)", min_value=100, max_value=2048, value=750, step=50)

    st.markdown("---")
    st.subheader("🔍 Cấu hình RAG & Search")

    # Search Mode Toggle
    search_mode = st.radio(
        "Thuật toán tìm kiếm tài liệu",
        ["Semantic Search (Gemini Embeddings)", "Keyword Search (Từ khóa Cục bộ)"],
        index=0
    )

    top_k_val = st.slider("Số lượng ngữ cảnh (Top K)", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.subheader("🧼 Dọn dẹp dữ liệu")
    if st.button("🗑️ Xóa toàn bộ lịch sử"):
        st.session_state.messages = []
        st.session_state.question_history = []
        st.session_state.last_context = ""
        st.session_state.processed_images = {}
        st.session_state.pdf_extracted_images = []
        st.session_state.parsed_nodes = []
        st.session_state.node_embeddings = {}
        st.session_state.files_hash = ""
        st.cache_data.clear()
        st.success("Đã làm sạch lịch sử thành công!")
        time.sleep(1)
        st.rerun()


# OpenCV pipeline removed to keep only Gemini Vision / search content.


# =========================
# FILE PARSING & CHUNKING
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


def extract_pdf_text_and_images(pdf_path, uploads_dir):
    """
    Extracts text page-by-page.
    If a page is scanned (no text), extracts images embedded in the page for Vision OCR.
    """
    reader = PdfReader(str(pdf_path))
    nodes = []
    counter = 0
    scanned_pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        
        # Detect potential scanned page
        if not page_text.strip() or len(page_text.strip()) < 50:
            scanned_pages.append(page_index)
            # Try to extract page images
            for img_idx, img_file in enumerate(page.images, start=1):
                try:
                    img_bytes = img_file.data
                    # Clean filename format
                    ext = Path(img_file.name).suffix or ".png"
                    img_name = f"{pdf_path.stem}_page{page_index}_img{img_idx}{ext}"
                    img_path = uploads_dir / img_name
                    img_path.write_bytes(img_bytes)

                    if str(img_path) not in st.session_state.pdf_extracted_images:
                        st.session_state.pdf_extracted_images.append(str(img_path))
                except Exception:
                    pass
        
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

    return nodes, scanned_pages


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
        # Handled in higher-level loop to handle image extraction
        return extract_pdf_text_and_images(file_path, UPLOADS_DIR)[0]

    if suffix in [".md", ".markdown"]:
        return parse_markdown(file_path)

    if suffix == ".txt":
        return extract_txt_nodes(file_path)

    return []


# =========================
# KEYWORD & SEMANTIC RAG RETRIEVAL
# =========================

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


def select_relevant_nodes(question, nodes, top_k):
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


# Semantic Embeddings RAG
def get_embeddings_for_nodes(nodes, api_key):
    """
    Fetch vector embeddings for nodes in batches using models/text-embedding-004
    Caches results inside st.session_state.node_embeddings.
    """
    genai.configure(api_key=api_key)
    uncached_nodes = [n for n in nodes if n["node_id"] not in st.session_state.node_embeddings]
    
    if uncached_nodes:
        try:
            batch_size = 50
            for i in range(0, len(uncached_nodes), batch_size):
                batch = uncached_nodes[i:i+batch_size]
                texts = [n["text"] for n in batch]
                
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                
                for idx, node in enumerate(batch):
                    st.session_state.node_embeddings[node["node_id"]] = result["embedding"][idx]
        except Exception as e:
            st.error(f"Lỗi tạo Vector Embeddings từ Gemini API: {e}. Vui lòng thử lại hoặc dùng chế độ Keyword Search.")
            return {}

    return st.session_state.node_embeddings


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    return np.dot(a, b) / norm


def select_relevant_nodes_semantic(question, nodes, top_k, api_key):
    """
    Retrieves nodes using semantic similarity. 
    Fallbacks to keyword search if errors occur or quota is exceeded.
    """
    if not nodes:
        return []

    embeddings = get_embeddings_for_nodes(nodes, api_key)
    if not embeddings:
        st.warning("⚠️ Lỗi trích xuất Vector, tự động fallback sang Tìm kiếm từ khóa cục bộ.")
        return select_relevant_nodes(question, nodes, top_k)

    try:
        genai.configure(api_key=api_key)
        q_result = genai.embed_content(
            model="models/text-embedding-004",
            content=question,
            task_type="retrieval_query"
        )
        q_emb = q_result["embedding"]

        scored_nodes = []
        for node in nodes:
            node_id = node["node_id"]
            if node_id in embeddings:
                score = cosine_similarity(q_emb, embeddings[node_id])
                scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_nodes[:top_k]]

    except Exception as e:
        st.warning(f"⚠️ Lỗi Tìm kiếm ngữ nghĩa ({e}), tự động fallback sang Tìm kiếm từ khóa cục bộ.")
        return select_relevant_nodes(question, nodes, top_k)


def build_context(selected_nodes):
    return "\n\n".join([n.get("text", "") for n in selected_nodes])


# =========================
# GEMINI API WRAPPERS
# =========================

def ask_gemini_vision(question, image_paths, model_name, api_key):
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


def ask_gemini(question, context_text, model_name, temp, max_tokens, api_key):
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
                st.warning(f"Mô hình Gemini đang quá tải. Đang thử lại tự động sau {wait_time}s...")
                time.sleep(wait_time)
                continue
            if "429" in error_text:
                raise Exception("Gemini đã hết quota hoặc bị giới hạn tốc độ (Rate Limit 429).")
            if "403" in error_text:
                raise Exception("API key hoặc project Gemini chưa có quyền truy cập model này.")
            raise e

    raise Exception("Mô hình Gemini phản hồi quá tải sau nhiều lần thử lại.")


# =========================
# MAIN APP INTERFACE
# =========================

st.title("📄 DocAnalyzer AI Pro")
st.caption("Trò chuyện thông minh cùng tài liệu và hình ảnh bằng Gemini RAG")


# 2. File Uploader
uploaded_files = st.file_uploader(
    "📂 Tải tài liệu hoặc hình ảnh lên",
    type=["pdf", "md", "markdown", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)


# =========================
# IMMEDIATE UPLOAD PROCESSING (Solves Streamlit Rerun Bug)
# =========================

if uploaded_files:
    saved_paths = []
    image_paths = []
    
    # Calculate simple hash of uploaded files to detect structural changes
    files_hash = hashlib.md5("".join([f.name for f in uploaded_files]).encode()).hexdigest()
    if st.session_state.files_hash != files_hash:
        # Reset intermediate states on structural file upload changes
        st.session_state.files_hash = files_hash
        st.session_state.parsed_nodes = []
        st.session_state.processed_images = {}
        st.session_state.pdf_extracted_images = []
        st.session_state.node_embeddings = {}

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        suffix = Path(file_path).suffix.lower()

        if suffix in [".png", ".jpg", ".jpeg"]:
            image_paths.append(str(file_path))
        else:
            saved_paths.append(str(file_path))

    # Parse and index text documents
    if saved_paths and not st.session_state.parsed_nodes:
        with st.spinner("📖 Đang phân tích cấu trúc tài liệu..."):
            all_nodes = []
            scanned_pages_info = {}
            
            for path_str in saved_paths:
                p_file = Path(path_str)
                if p_file.suffix.lower() == ".pdf":
                    nodes, scanned_pages = extract_pdf_text_and_images(p_file, UPLOADS_DIR)
                    all_nodes.extend(nodes)
                    if scanned_pages:
                        scanned_pages_info[p_file.name] = scanned_pages
                else:
                    nodes = parse_document(p_file)
                    all_nodes.extend(nodes)

            st.session_state.parsed_nodes = all_nodes

            # Show warnings for detected scanned pages
            for doc_name, pages in scanned_pages_info.items():
                st.warning(
                    f"⚠️ Tài liệu `{doc_name}` phát hiện các trang chụp/quét (không có text): Trang {', '.join(map(str, pages))}. "
                    "Hệ thống đã tự động trích xuất các hình ảnh trong trang để bạn gửi trực tiếp cho Gemini Vision!"
                )

    # 3. Preview Interface
    extracted_pdf_imgs = st.session_state.pdf_extracted_images

    if image_paths or extracted_pdf_imgs:
        with st.expander("🖼️ Khu vực xem thử hình ảnh (Không mất khi chat)", expanded=True):
            # Render standard uploaded images
            for img_path in image_paths:
                st.markdown(f"**📸 Hình ảnh:** `{Path(img_path).name}`")
                st.image(img_path, caption="Ảnh gốc đã upload", use_container_width=True)
                st.markdown("---")

            # Render extracted images from scanned PDFs
            if extracted_pdf_imgs:
                st.markdown("**📂 Hình ảnh tự động trích xuất từ PDF quét:**")
                grid_cols = st.columns(min(3, len(extracted_pdf_imgs)))
                for idx, ext_path in enumerate(extracted_pdf_imgs):
                    target_col = grid_cols[idx % len(grid_cols)]
                    with target_col:
                        st.image(ext_path, caption=f"PDF Extracted - {Path(ext_path).name}", use_container_width=True)


# =========================
# CHAT FLOW & RENDERING
# =========================

# Render Historical Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if "images" in msg and msg["images"]:
                cols = st.columns(min(4, len(msg["images"])))
                for i, img_path in enumerate(msg["images"]):
                    with cols[i % len(cols)]:
                        st.image(img_path, caption="Ảnh gửi kèm", use_container_width=True)
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bubble-assistant">'
                f'<div class="answer-header">🤖 Trợ lý AI</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )

# Input
question = st.chat_input("Hỏi nội dung tài liệu hoặc hình ảnh...")


# =========================
# QUERY EXECUTION
# =========================

if question:
    if not uploaded_files:
        st.warning("⚠️ Vui lòng tải tài liệu hoặc hình ảnh lên trước.")
        st.stop()

    # Collect active images (uploaded images + extracted PDF images)
    active_query_images = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            suffix = Path(file_path).suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg"]:
                active_query_images.append(str(file_path))

    if st.session_state.pdf_extracted_images:
        for ext_path in st.session_state.pdf_extracted_images:
            active_query_images.append(ext_path)

    # Record User Message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "images": active_query_images
    })

    if question not in st.session_state.question_history:
        st.session_state.question_history.append(question)

    st.rerun()  # Rerun immediately to display user bubble before invoking LLM


# Invoke LLM (Triggered if the last message belongs to user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]
    user_question = user_msg["content"]
    user_images = user_msg.get("images", [])

    with st.chat_message("assistant"):
        # 1. RAG Text Document Search
        if st.session_state.parsed_nodes:
            with st.spinner("🤖 Đang quét tài liệu tìm ngữ cảnh phù hợp..."):
                if search_mode == "Semantic Search (Gemini Embeddings)":
                    selected_nodes = select_relevant_nodes_semantic(
                        question=user_question,
                        nodes=st.session_state.parsed_nodes,
                        top_k=top_k_val,
                        api_key=api_key_input
                    )
                else:
                    selected_nodes = select_relevant_nodes(
                        question=user_question,
                        nodes=st.session_state.parsed_nodes,
                        top_k=top_k_val
                    )

                context_text = build_context(selected_nodes)
                st.session_state.last_context = context_text

            with st.spinner("🤖 Gemini đang suy luận và lập luận văn bản..."):
                try:
                    answer = ask_gemini(
                        question=user_question,
                        context_text=context_text,
                        model_name=model_choice,
                        temp=temperature,
                        max_tokens=max_output_tokens,
                        api_key=api_key_input
                    )
                except Exception as e:
                    answer = f"❌ Lỗi truy vấn Gemini: {e}"

        # 2. Vision Only OCR
        elif user_images:
            with st.spinner("🤖 Gemini Vision đang phân tích hình ảnh..."):
                try:
                    answer = ask_gemini_vision(
                        question=user_question,
                        image_paths=user_images,
                        model_name=model_choice,
                        api_key=api_key_input
                    )
                except Exception as e:
                    answer = f"❌ Lỗi phân tích ảnh bằng Gemini Vision: {e}"
        else:
            answer = "❌ Không tìm thấy văn bản tài liệu hay hình ảnh hợp lệ để phân tích."

        # Display Answer using premium bubbles
        st.markdown(
            f'<div class="chat-bubble-assistant">'
            f'<div class="answer-header">🤖 Trợ lý AI</div>'
            f'{answer}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Context inspection expander
        if st.session_state.parsed_nodes and st.session_state.last_context:
            with st.expander("📚 Xem context đã trích xuất"):
                st.code(st.session_state.last_context)
                
        st.rerun()  # Rerun to sync final state and reset chat input trigger


# =========================
# LỊCH SỬ SIDEBAR PANEL RENDERING
# =========================

with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Nhật ký câu hỏi")
    hist = st.session_state.get("question_history", [])
    if hist:
        for idx, q_text in enumerate(hist[::-1], start=1):
            st.markdown(f'<div class="history-box"><b>{idx}.</b> {q_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="history-box">Chưa có câu hỏi nào</div>', unsafe_allow_html=True)
