import json
import re
from pathlib import Path

import streamlit as st
from google import genai


# =========================
# CONFIG
# =========================
APP_TITLE = "📄 DocAnalyzer AI"

BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

# =========================
# GEMINI KEY
# =========================
GEMINI_API_KEY = "AIzaSyDZyz_6O7OoAkrS7a8P1-d7YbDrsBAlT5c"


# =========================
# HELPERS
# =========================
def save_uploaded_file(uploaded_file) -> Path:
    save_path = UPLOADS_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    return save_path


# =========================
# MARKDOWN PARSER
# =========================
def extract_markdown_nodes(md_text: str):
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


def flatten_tree(nodes, parent_path=""):
    result = []

    for node in nodes:
        title = node.get("title", "")

        path = (
            f"{parent_path} > {title}"
            if parent_path
            else title
        )

        result.append({
            "node_id": node.get("node_id", ""),
            "title": title,
            "path": path,
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "text": node.get("text", ""),
        })

        children = node.get("nodes", [])

        if children:
            result.extend(
                flatten_tree(children, path)
            )

    return result


def parse_markdown(md_path: Path):
    md_text = md_path.read_text(
        encoding="utf-8"
    )

    flat_nodes, lines = extract_markdown_nodes(md_text)

    if not flat_nodes:
        flat_nodes = [{
            "title": md_path.stem,
            "level": 1,
            "line_num": 1,
        }]

    flat_nodes = add_text_to_nodes(
        flat_nodes,
        lines
    )

    tree = build_tree(flat_nodes)

    return flatten_tree(tree)


# =========================
# RETRIEVAL
# =========================
def keyword_score(question, text, title=""):
    q_words = re.findall(
        r"\w+",
        question.lower()
    )

    content = f"{title} {text}".lower()

    score = 0

    for word in q_words:
        if word in content:
            score += 1

    return score


def select_relevant_nodes(
    question,
    nodes,
    top_k=3
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

    return ranked[:top_k]


def build_context(selected_nodes):
    return "\n\n---\n\n".join([
        f"""
[Node {n['node_id']}]
{n['text']}
"""
        for n in selected_nodes
    ])


# =========================
# GEMINI
# =========================
def ask_gemini(question, context_text):
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
Bạn là AI phân tích tài liệu.

Chỉ trả lời dựa trên CONTEXT.
Không được bịa.

QUESTION:
{question}

CONTEXT:
{context_text}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return response.text


# =========================
# UI
# =========================
st.markdown(
    """
    <style>
    .main {
        padding-top: 20px;
    }

    .stTextInput input {
        border-radius: 12px;
    }

    .stButton button {
        width: 100%;
        border-radius: 12px;
        height: 45px;
        font-size: 16px;
    }

    .block-container {
        max-width: 1100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 DocAnalyzer AI")
st.caption(
    "Chat với tài liệu Markdown bằng Gemini"
)

uploaded_file = st.file_uploader(
    "Upload tài liệu Markdown",
    type=["md", "markdown", "txt"]
)

question = st.text_input(
    "💬 Hỏi tài liệu của bạn...",
    placeholder="Ví dụ: doanh thu năm 2025 thế nào?"
)

top_k = st.slider(
    "Số đoạn context",
    min_value=1,
    max_value=10,
    value=3
)

ask_button = st.button("✨ Phân tích")

# =========================
# PROCESS
# =========================
if ask_button:

    if uploaded_file is None:
        st.warning("Vui lòng upload file.")
        st.stop()

    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()

    save_path = save_uploaded_file(
        uploaded_file
    )

    with st.spinner(
        "Đang đọc tài liệu..."
    ):
        nodes = parse_markdown(save_path)

    selected_nodes = select_relevant_nodes(
        question,
        nodes,
        top_k
    )

    context_text = build_context(
        selected_nodes
    )

    with st.spinner(
        "Gemini đang phân tích..."
    ):
        try:
            answer = ask_gemini(
                question,
                context_text
            )

            st.markdown("## 🤖 Kết quả")
            st.write(answer)

        except Exception as e:
            st.error(
                f"Lỗi Gemini: {e}"
            )

    with st.expander(
        "📚 Context đã dùng"
    ):
        st.code(context_text)