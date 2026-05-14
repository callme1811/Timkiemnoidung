import re
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


# ==================================================
# CONFIG
# ==================================================
APP_TITLE = "DocAnalyzer AI"

BASE_DIR = Path(__file__).parent.resolve()

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ==================================================
# GEMINI API KEY
# ==================================================
GEMINI_API_KEY = "AIzaSyCcRzC2uVjqKz2dVeUXcejQ1SmGIGYHeTM"

genai.configure(
    api_key=GEMINI_API_KEY
)

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


# ==================================================
# SAVE FILE
# ==================================================
def save_uploaded_file(uploaded_file):

    save_path = (
        UPLOADS_DIR / uploaded_file.name
    )

    save_path.write_bytes(
        uploaded_file.getbuffer()
    )

    return save_path


# ==================================================
# PDF PARSER
# ==================================================
def extract_pdf_text(pdf_path: Path):

    reader = PdfReader(str(pdf_path))

    pages = []

    for index, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "node_id": str(index).zfill(4),
                "title": f"Page {index}",
                "path": f"{pdf_path.name} > Page {index}",
                "text": text.strip(),
            })

    return pages


# ==================================================
# TXT PARSER
# ==================================================
def extract_txt_nodes(file_path: Path):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [{
        "node_id": "0001",
        "title": file_path.name,
        "path": file_path.name,
        "text": text.strip(),
    }]


# ==================================================
# MARKDOWN PARSER
# ==================================================
def extract_markdown_nodes(md_text: str):

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


def add_text_to_nodes(nodes, lines):

    for i, node in enumerate(nodes):

        start = node["line_num"]

        end = (
            nodes[i + 1]["line_num"] - 1
            if i + 1 < len(nodes)
            else len(lines)
        )

        node["text"] = "\n".join(
            lines[start - 1:end]
        ).strip()

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

        title = node.get("title", "")

        path = (
            f"{parent_path} > {title}"
            if parent_path
            else title
        )

        result.append({
            "node_id": node.get(
                "node_id",
                ""
            ),
            "title": title,
            "path": path,
            "text": node.get(
                "text",
                ""
            ),
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


def parse_markdown(
    file_path: Path
):

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


# ==================================================
# DOCUMENT PARSER
# ==================================================
def parse_document(
    file_path: Path
):

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


# ==================================================
# RETRIEVAL
# ==================================================
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
            n.get("text", ""),
            n.get("title", "")
        ),
        reverse=True,
    )

    selected = [
        n for n in ranked
        if keyword_score(
            question,
            n.get("text", ""),
            n.get("title", "")
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


# ==================================================
# GEMINI
# ==================================================
def ask_gemini(
    question,
    context_text
):

    model = genai.GenerativeModel(
        "gemini-2.0-flash"
    )

    prompt = f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Giải thích đầy đủ để người đọc dễ hiểu.
- Không trả lời quá ngắn.
- Viết theo kiểu giải thích cho người mới học.
- Nếu có nhiều ý hãy chia bullet point.
- Nếu là khái niệm:
    + giải thích định nghĩa
    + mục đích
    + cách hoạt động
    + lợi ích
    + nhược điểm nếu có
    + ví dụ thực tế
- Nếu là hệ thống/AI/Agent:
    + mô tả flow hoạt động
    + các thành phần liên quan
    + cách các thành phần giao tiếp với nhau
    + ví dụ minh họa thực tế
- Nếu là báo cáo/số liệu:
    + tóm tắt số liệu chính
    + phân tích xu hướng
    + nêu insight quan trọng
- Luôn thêm:
    + ví dụ minh họa
    + tình huống thực tế
    + cách áp dụng thực tế
- Nếu tài liệu không có thông tin:
  trả lời đúng:
  "Tôi không tìm thấy thông tin này trong tài liệu."

YÊU CẦU FORMAT:
- Dùng markdown đẹp.
- Có tiêu đề nhỏ.
- Có bullet point.
- Có ví dụ minh họa.
- Có kết luận ngắn.
- Không trả lời 1 câu cụt ngủn.
- Giải thích rõ ràng để người không chuyên vẫn hiểu.

QUESTION:
{question}

CONTEXT:
{context_text}
"""

    response = model.generate_content(
        prompt
    )

    return response.text


# ==================================================
# UI
# ==================================================
st.markdown(
    """
<style>

.block-container{
    max-width:1000px;
    padding-top:40px;
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

</style>
""",
    unsafe_allow_html=True,
)

st.title(
    "📄 DocAnalyzer AI"
)

st.caption(
    "Upload PDF, Markdown hoặc TXT rồi hỏi nội dung tài liệu."
)

uploaded_file = st.file_uploader(
    "📂 Tải tài liệu lên",
    type=[
        "pdf",
        "md",
        "markdown",
        "txt"
    ]
)

question = st.text_input(
    "💬 Nhập câu hỏi",
    placeholder=(
        "Ví dụ: Agent A2A là gì?"
    )
)

top_k = st.slider(
    "📚 Số đoạn dùng để trả lời",
    min_value=1,
    max_value=10,
    value=5
)

ask_button = st.button(
    "✨ Hỏi tài liệu"
)


# ==================================================
# RUN
# ==================================================
if ask_button:

    if uploaded_file is None:

        st.warning(
            "Vui lòng upload tài liệu."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "Vui lòng nhập câu hỏi."
        )

        st.stop()

    with st.spinner(
        "📖 Đang đọc tài liệu..."
    ):

        file_path = save_uploaded_file(
            uploaded_file
        )

        nodes = parse_document(
            file_path
        )

    if not nodes:

        st.error(
            "Không đọc được tài liệu."
        )

        st.stop()

    selected_nodes = (
        select_relevant_nodes(
            question=question,
            nodes=nodes,
            top_k=top_k,
        )
    )

    context_text = build_context(
        selected_nodes
    )

    with st.spinner(
        "🤖 Gemini đang phân tích..."
    ):

        try:

            answer = ask_gemini(
                question,
                context_text
            )

            st.markdown(
                "## 🤖 Câu trả lời"
            )

            st.markdown(
                f"""
<div class="answer-box">

{answer}

</div>
""",
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"Lỗi Gemini: {e}"
            )

    with st.expander(
        "📚 Xem đoạn tài liệu đã dùng"
    ):

        st.code(context_text)