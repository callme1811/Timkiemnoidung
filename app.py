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

# =========================================================
# GEMINI API
# =========================================================
GEMINI_API_KEY = "AIzaSyCcRzC2uVjqKz2dVeUXcejQ1SmGIGYHeTM"

genai.configure(
    api_key=GEMINI_API_KEY
)

# =========================================================
# STREAMLIT
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

# =========================================================
# SAVE FILE
# =========================================================
def save_uploaded_file(uploaded_file):

    file_id = str(uuid.uuid4())[:8]

    save_path = (
        UPLOADS_DIR /
        f"{file_id}_{uploaded_file.name}"
    )

    save_path.write_bytes(
        uploaded_file.getbuffer()
    )

    return save_path

# =========================================================
# TEXT CHUNKING
# =========================================================
def split_text(
    text,
    chunk_size=1200
):

    text = text.strip()

    if not text:
        return []

    paragraphs = text.split("\n")

    chunks = []

    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if (
            len(current_chunk) + len(para)
            < chunk_size
        ):

            current_chunk += (
                "\n" + para
            )

        else:

            chunks.append(
                current_chunk.strip()
            )

            current_chunk = para

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    return chunks

# =========================================================
# PDF PARSER
# =========================================================
def extract_pdf_text(pdf_path):

    reader = PdfReader(
        str(pdf_path)
    )

    nodes = []

    counter = 0

    for page_index, page in enumerate(
        reader.pages,
        start=1
    ):

        text = (
            page.extract_text()
            or ""
        )

        chunks = split_text(text)

        for chunk in chunks:

            counter += 1

            nodes.append({
                "node_id": str(counter).zfill(4),
                "title": f"Page {page_index}",
                "path": f"{pdf_path.name} > Page {page_index}",
                "text": chunk,
            })

    return nodes

# =========================================================
# TXT PARSER
# =========================================================
def extract_txt_nodes(file_path):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = split_text(text)

    nodes = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        nodes.append({
            "node_id": str(i).zfill(4),
            "title": file_path.name,
            "path": file_path.name,
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

    for i, line in enumerate(
        lines,
        start=1
    ):

        stripped = line.strip()

        if stripped.startswith("```"):

            in_code_block = (
                not in_code_block
            )

            continue

        if in_code_block:
            continue

        match = re.match(
            r"^(#{1,6})\s+(.+)$",
            stripped
        )

        if match:

            nodes.append({
                "title": match.group(2).strip(),
                "level": len(match.group(1)),
                "line_num": i,
            })

    return nodes, lines

def add_text_to_nodes(
    nodes,
    lines
):

    for i, node in enumerate(nodes):

        start = node["line_num"]

        end = (
            nodes[i + 1]["line_num"] - 1
            if i + 1 < len(nodes)
            else len(lines)
        )

        text = "\n".join(
            lines[start - 1:end]
        ).strip()

        node["text"] = text

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
            "text": node["text"],
            "nodes": [],
        }

        while (
            stack
            and stack[-1]["level"]
            >= node["level"]
        ):
            stack.pop()

        if stack:

            stack[-1]["node"][
                "nodes"
            ].append(tree_node)

        else:
            root.append(tree_node)

        stack.append({
            "level": node["level"],
            "node": tree_node,
        })

    return root

def flatten_tree(
    nodes,
    parent_path=""
):

    result = []

    for node in nodes:

        title = node.get(
            "title",
            ""
        )

        path = (
            f"{parent_path} > {title}"
            if parent_path
            else title
        )

        text_chunks = split_text(
            node.get("text", "")
        )

        for chunk in text_chunks:

            result.append({
                "node_id": node.get(
                    "node_id",
                    ""
                ),
                "title": title,
                "path": path,
                "text": chunk,
            })

        children = node.get(
            "nodes",
            []
        )

        if children:

            result.extend(
                flatten_tree(
                    children,
                    path
                )
            )

    return result

def parse_markdown(file_path):

    md_text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    flat_nodes, lines = (
        extract_markdown_nodes(
            md_text
        )
    )

    if not flat_nodes:

        return extract_txt_nodes(
            file_path
        )

    flat_nodes = add_text_to_nodes(
        flat_nodes,
        lines
    )

    tree = build_tree(
        flat_nodes
    )

    return flatten_tree(tree)

# =========================================================
# DOCUMENT PARSER
# =========================================================
def parse_document(file_path):

    suffix = (
        file_path.suffix.lower()
    )

    if suffix == ".pdf":

        return extract_pdf_text(
            file_path
        )

    if suffix in [
        ".md",
        ".markdown"
    ]:

        return parse_markdown(
            file_path
        )

    if suffix == ".txt":

        return extract_txt_nodes(
            file_path
        )

    return []

# =========================================================
# RETRIEVAL
# =========================================================
def keyword_score(
    question,
    text,
    title=""
):

    q_words = re.findall(
        r"\w+",
        question.lower()
    )

    content = (
        f"{title} {text}"
    ).lower()

    score = 0

    for word in q_words:

        if word in content:
            score += 1

    return score

def select_relevant_nodes(
    question,
    nodes,
    top_k=5
):

    ranked = sorted(
        nodes,
        key=lambda n: keyword_score(
            question,
            n["text"],
            n["title"]
        ),
        reverse=True,
    )

    selected = [
        n for n in ranked
        if keyword_score(
            question,
            n["text"],
            n["title"]
        ) > 0
    ]

    if not selected:
        selected = ranked

    return selected[:top_k]

def build_context(
    selected_nodes
):

    return "\n\n---\n\n".join([

        f"""
[Nguồn: {n["path"]}]

{n["text"]}
"""

        for n in selected_nodes
    ])

# =========================================================
# GEMINI
# =========================================================
def ask_gemini(
    question,
    context_text
):

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    prompt = f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên CONTEXT.
- Không được bịa.
- Giải thích dễ hiểu.
- Không trả lời quá ngắn.
- Viết như đang hướng dẫn người mới.
- Nếu có nhiều ý hãy dùng bullet point.

NẾU LÀ KHÁI NIỆM:
- Giải thích định nghĩa
- Mục đích
- Cách hoạt động
- Ưu điểm
- Ví dụ minh họa

NẾU LÀ AI / AGENT / SYSTEM:
- Giải thích workflow
- Thành phần liên quan
- Cách giao tiếp
- Ví dụ thực tế

NẾU LÀ BÁO CÁO:
- Tóm tắt ý chính
- Insight quan trọng
- Xu hướng

BẮT BUỘC:
- Có ví dụ minh họa
- Có bullet point
- Có kết luận ngắn
- Dùng markdown đẹp

Nếu tài liệu không có thông tin:
"Tôi không tìm thấy thông tin này trong tài liệu."

QUESTION:
{question}

CONTEXT:
{context_text}
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 1500,
        },
        stream=True
    )

    full_text = ""

    response_placeholder = st.empty()

    for chunk in response:

        if chunk.text:

            full_text += chunk.text

            response_placeholder.markdown(
                f"""
<div class="answer-box">

{full_text}

</div>
""",
                unsafe_allow_html=True
            )

    return full_text

# =========================================================
# UI
# =========================================================
st.markdown(
    """
<style>

.block-container{
    max-width:1000px;
    padding-top:30px;
}

.stButton button{
    width:100%;
    height:52px;
    border-radius:14px;
    font-size:18px;
    font-weight:700;
}

.stTextInput input{
    border-radius:14px;
    height:50px;
    font-size:18px;
}

.stFileUploader{
    border-radius:14px;
}

.answer-box{
    padding:25px;
    border-radius:18px;
    background:#111827;
    border:1px solid #374151;
    margin-top:10px;
}

.user-box{
    padding:18px;
    border-radius:16px;
    background:#1f2937;
    margin-bottom:12px;
}

</style>
""",
    unsafe_allow_html=True
)

st.title(
    "📄 DocAnalyzer AI"
)

st.caption(
    "Chat với PDF, Markdown và TXT bằng Gemini 2.5"
)

# =========================================================
# MULTI FILE UPLOAD
# =========================================================
uploaded_files = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=[
        "pdf",
        "md",
        "markdown",
        "txt"
    ],
    accept_multiple_files=True
)

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# =========================================================
# USER INPUT
# =========================================================
question = st.chat_input(
    "Hỏi nội dung tài liệu..."
)

# =========================================================
# RUN
# =========================================================
if question:

    if not uploaded_files:

        st.warning(
            "Vui lòng upload tài liệu."
        )

        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    all_nodes = []

    with st.spinner(
        "📖 Đang đọc tài liệu..."
    ):

        for uploaded_file in uploaded_files:

            file_path = save_uploaded_file(
                uploaded_file
            )

            nodes = parse_document(
                file_path
            )

            all_nodes.extend(nodes)

    selected_nodes = (
        select_relevant_nodes(
            question=question,
            nodes=all_nodes,
            top_k=6,
        )
    )

    context_text = build_context(
        selected_nodes
    )

    with st.chat_message(
        "assistant"
    ):

        try:

            answer = ask_gemini(
                question,
                context_text
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            error_text = str(e)

            if "429" in error_text:

                st.error(
                    "🚫 Gemini đã hết quota miễn phí. Hãy đổi API key hoặc chờ reset quota."
                )

            else:

                st.error(
                    f"Lỗi Gemini: {e}"
                )

    with st.expander(
        "📚 Xem context đã dùng"
    ):

        st.code(context_text)