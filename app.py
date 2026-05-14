import re
import uuid
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "DocAnalyzer AI"

BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Dán API key Gemini của bạn vào đây
GEMINI_API_KEY = "AIzaSyCcRzC2uVjqKz2dVeUXcejQ1SmGIGYHeTM"

genai.configure(api_key=GEMINI_API_KEY)


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_history" not in st.session_state:
    st.session_state.question_history = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""


# =========================================================
# FILE HELPERS
# =========================================================
def save_uploaded_file(uploaded_file):
    file_id = str(uuid.uuid4())[:8]
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    save_path = UPLOADS_DIR / f"{file_id}_{safe_name}"
    save_path.write_bytes(uploaded_file.getbuffer())
    return save_path


# =========================================================
# TEXT CHUNKING
# =========================================================
def split_text(text, chunk_size=1200, overlap=150):
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
            current_chunk += "\n" + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = end - overlap
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# =========================================================
# PDF PARSER
# =========================================================
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


# =========================================================
# TXT PARSER
# =========================================================
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


# =========================================================
# MARKDOWN PARSER
# =========================================================
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
            "start_line": node["start_line"],
            "end_line": node["end_line"],
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
            result.extend(
                flatten_tree(
                    children,
                    path,
                    source_file,
                )
            )

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


# =========================================================
# DOCUMENT PARSER
# =========================================================
def parse_document(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(file_path)

    if suffix in [".md", ".markdown"]:
        return parse_markdown(file_path)

    if suffix == ".txt":
        return extract_txt_nodes(file_path)

    return []


def parse_uploaded_files(uploaded_files):
    all_nodes = []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        nodes = parse_document(file_path)
        all_nodes.extend(nodes)

    return all_nodes


# =========================================================
# RETRIEVAL
# =========================================================
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


def select_relevant_nodes(question, nodes, top_k=6):
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

    for index, n in enumerate(selected_nodes, start=1):
        context_parts.append(
            f"""
[SOURCE {index}]
File: {n.get("source_file", "")}
Vị trí: {n.get("path", "")}

Nội dung:
{n.get("text", "")}
"""
        )

    return "\n\n---\n\n".join(context_parts)


# =========================================================
# GEMINI
# =========================================================
def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Giải thích dễ hiểu cho người mới.
- Không trả lời quá ngắn.
- Nếu có nhiều ý, dùng bullet point.
- Luôn ưu tiên nội dung có trong tài liệu.
- Nếu thiếu thông tin, nói rõ phần nào tài liệu không cung cấp.

NẾU LÀ KHÁI NIỆM:
- Giải thích định nghĩa.
- Giải thích mục đích.
- Giải thích cách hoạt động.
- Nêu ưu điểm.
- Nêu hạn chế nếu tài liệu có.
- Sau khi trả lời xong, thêm ví dụ minh họa hoàn chỉnh.

NẾU LÀ AI / AGENT / SYSTEM:
- Giải thích workflow.
- Nêu các thành phần liên quan.
- Giải thích cách các thành phần giao tiếp với nhau.
- Sau khi trả lời xong, thêm ví dụ thực tế dễ hiểu.

NẾU LÀ BÁO CÁO / SỐ LIỆU:
- Tóm tắt ý chính.
- Nêu số liệu quan trọng.
- Phân tích xu hướng nếu có.
- Nêu insight quan trọng.

YÊU CẦU VÍ DỤ:
- Ví dụ phải viết sau phần giải thích chính.
- Ví dụ phải có đủ: bối cảnh, hành động, kết quả.
- Không được viết câu cụt như: "Hãy tưởng..."
- Không được bỏ dở câu trả lời giữa chừng.
- Nếu tài liệu không cung cấp ví dụ cụ thể, hãy tự tạo ví dụ minh họa đơn giản nhưng phải nói rõ đó là "ví dụ minh họa".

YÊU CẦU FORMAT:
- Dùng markdown.
- Có tiêu đề nhỏ.
- Có bullet point.
- Có phần "Ví dụ minh họa".
- Có phần "Kết luận ngắn".
- Nếu có thể, nhắc nguồn theo dạng [SOURCE 1], [SOURCE 2].

Nếu tài liệu không có thông tin để trả lời:
"Tôi không tìm thấy thông tin này trong tài liệu."

QUESTION:
{question}

CONTEXT:
{context_text}
"""


def extract_chunk_text(chunk):
    chunk_text = ""

    if hasattr(chunk, "parts") and chunk.parts:
        for part in chunk.parts:
            if hasattr(part, "text") and part.text:
                chunk_text += part.text

    if not chunk_text:
        try:
            chunk_text = chunk.text or ""
        except Exception:
            chunk_text = ""

    return chunk_text


def ask_gemini(question, context_text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = build_prompt(question, context_text)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.35,
            "max_output_tokens": 3000,
        },
        stream=True,
    )

    full_text = ""
    response_placeholder = st.empty()

    try:
        for chunk in response:
            chunk_text = extract_chunk_text(chunk)

            if chunk_text:
                full_text += chunk_text

                response_placeholder.markdown(
                    f"""
<div class="answer-box">

{full_text}

</div>
""",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Lỗi stream Gemini: {e}")

    return full_text


# =========================================================
# UI STYLE
# =========================================================
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

.source-chip{
    display:inline-block;
    padding:6px 10px;
    border-radius:999px;
    background:#1f2937;
    border:1px solid #374151;
    margin:4px 4px 4px 0;
    font-size:13px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("📚 Lịch sử")

    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages = []
        st.session_state.question_history = []
        st.session_state.last_context = ""
        st.rerun()

    st.markdown("---")

    if st.session_state.question_history:
        for i, q in enumerate(
            reversed(st.session_state.question_history),
            start=1,
        ):
            st.markdown(
                f"""
<div class="history-box">
{i}. {q}
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Chưa có câu hỏi nào")


# =========================================================
# HEADER
# =========================================================
st.title("📄 DocAnalyzer AI")
st.caption("Chat với PDF, Markdown và TXT bằng Gemini 2.5")


# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_files = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=["pdf", "md", "markdown", "txt"],
    accept_multiple_files=True,
)


# =========================================================
# SHOW CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# CHAT INPUT
# =========================================================
question = st.chat_input("Hỏi nội dung tài liệu...")


# =========================================================
# RUN
# =========================================================
if question:
    if not uploaded_files:
        st.warning("Vui lòng upload tài liệu.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    st.session_state.question_history.append(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("📖 Đang đọc tài liệu..."):
        all_nodes = parse_uploaded_files(uploaded_files)

    if not all_nodes:
        st.error("Không đọc được nội dung tài liệu. Nếu PDF là dạng scan ảnh, cần OCR.")
        st.stop()

    selected_nodes = select_relevant_nodes(
        question=question,
        nodes=all_nodes,
        top_k=6,
    )

    context_text = build_context(selected_nodes)
    st.session_state.last_context = context_text

    with st.chat_message("assistant"):
        try:
            answer = ask_gemini(question, context_text)

            if answer.strip():
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })
            else:
                st.error(
                    "Gemini không trả về nội dung. Hãy thử hỏi ngắn hơn hoặc giảm số tài liệu."
                )

        except Exception as e:
            error_text = str(e)

            if "429" in error_text:
                st.error(
                    "🚫 Gemini đã hết quota miễn phí. Hãy đổi API key hoặc chờ reset quota."
                )
            elif "finish_reason" in error_text:
                st.error(
                    "Gemini đã dừng phản hồi giữa chừng. Hãy hỏi ngắn hơn hoặc giảm số tài liệu."
                )
            elif "403" in error_text:
                st.error(
                    "🚫 API key hoặc project Gemini chưa có quyền truy cập. Hãy đổi API key/project khác."
                )
            else:
                st.error(f"Lỗi Gemini: {e}")

    with st.expander("📚 Xem context đã dùng"):
        st.code(context_text)