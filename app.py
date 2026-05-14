import json
import re
from pathlib import Path

import streamlit as st
from google import genai


# =========================
# CẤU HÌNH APP
# =========================
APP_TITLE = "DocAnalyzer - Gemini Document Search"

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MD_PATH = BASE_DIR / "technova_ai_demo_data.md"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"

RESULTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")


# =========================
# HÀM HỖ TRỢ
# =========================
def normalize_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else BASE_DIR / p


def get_tree_path(md_path: Path) -> Path:
    safe_name = md_path.stem.replace(" ", "_")
    return RESULTS_DIR / f"{safe_name}_structure.json"


def save_uploaded_file(uploaded_file) -> Path:
    save_path = UPLOADS_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    return save_path


# =========================
# BUILD TREE TỪ MARKDOWN
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


def build_markdown_tree(md_path: Path):
    md_text = md_path.read_text(encoding="utf-8")
    flat_nodes, lines = extract_markdown_nodes(md_text)

    if not flat_nodes:
        flat_nodes = [{
            "title": md_path.stem,
            "level": 1,
            "line_num": 1,
        }]

    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)

    data = {
        "doc_name": md_path.stem,
        "source_file": str(md_path),
        "line_count": len(lines),
        "structure": tree,
    }

    tree_path = get_tree_path(md_path)
    tree_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return tree_path, data


def flatten_tree(nodes, parent_path=""):
    result = []

    for node in nodes:
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title

        result.append({
            "node_id": node.get("node_id", ""),
            "title": title,
            "path": path,
            "line_num": node.get("line_num"),
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "text": node.get("text", ""),
        })

        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path))

    return result


# =========================
# RETRIEVAL KEYWORD
# =========================
def keyword_score(question: str, node_text: str, node_title: str = ""):
    q_words = re.findall(r"\w+", question.lower())
    content = f"{node_title} {node_text}".lower()

    if not q_words:
        return 0

    score = 0
    for word in q_words:
        if word in content:
            score += 1

    return score


def select_relevant_nodes(question, nodes, top_k=3):
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
        if keyword_score(question, n.get("text", ""), n.get("title", "")) > 0
    ]

    if not selected:
        selected = ranked

    return selected[:top_k]


def build_context_text(selected_nodes):
    if not selected_nodes:
        return ""

    return "\n\n---\n\n".join([
        f"[Node {n['node_id']} | {n['path']} | lines {n['start_line']}-{n['end_line']}]\n\n{n['text']}"
        for n in selected_nodes
    ])


# =========================
# GEMINI ANSWER
# =========================
def ask_gemini(api_key: str, question: str, context_text: str):
    client = genai.Client(api_key=api_key)

    prompt = f"""
Bạn là trợ lý phân tích tài liệu.

Nhiệm vụ:
- Trả lời câu hỏi của người dùng dựa trên CONTEXT.
- Không bịa thông tin ngoài tài liệu.
- Nếu CONTEXT không có thông tin, hãy nói: "Không tìm thấy thông tin này trong tài liệu."
- Trả lời bằng tiếng Việt.
- Nên trích dẫn Node và dòng nếu có.

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
# STREAMLIT UI
# =========================
st.title("📄 DocAnalyzer")
st.caption("Upload Markdown → Build Tree → Search Nodes → Ask Gemini")

with st.sidebar:
    st.header("🔑 Gemini")

    gemini_key = st.text_input(
        "AIzaSyDZyz_6O7OoAkrS7a8P1-d7YbDrsBAlT5c",
        type="password",
        help="Nhập API key để Gemini trả lời dựa trên tài liệu.",
    )

    st.markdown("---")
    st.header("📂 Tài liệu")

    uploaded_file = st.file_uploader(
        "Upload file Markdown",
        type=["md", "markdown", "txt"],
    )

    if uploaded_file is not None:
        uploaded_path = save_uploaded_file(uploaded_file)
        st.success(f"Đã upload: {uploaded_file.name}")
        default_path_value = str(uploaded_path)
    else:
        default_path_value = str(DEFAULT_MD_PATH)

    md_path_input = st.text_input(
        "Đường dẫn file Markdown",
        value=default_path_value,
    )

    build_button = st.button("Build Tree")

    st.markdown("---")
    st.header("Hướng dẫn")
    st.markdown("""
1. Nhập Gemini API key.  
2. Upload file Markdown hoặc dùng file mặc định.  
3. Bấm **Build Tree**.  
4. Nhập câu hỏi.  
5. Gemini sẽ trả lời dựa trên node liên quan.
""")


md_path = normalize_path(md_path_input)
tree_path = get_tree_path(md_path)

if build_button:
    if not md_path.exists():
        st.error(f"Không tìm thấy file Markdown: {md_path}")
    else:
        with st.spinner("Đang build tree..."):
            tree_path, tree_data = build_markdown_tree(md_path)

        st.success(f"Build tree thành công: {tree_path}")

        with st.expander("Xem JSON tree"):
            st.code(
                json.dumps(tree_data, indent=2, ensure_ascii=False)[:5000],
                language="json",
            )


st.subheader("1. Load tree structure")

if tree_path.exists():
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    raw_nodes = tree.get("structure", [])
    nodes = flatten_tree(raw_nodes)

    st.success(f"Đã load tài liệu: {tree.get('doc_name', '')}")
    st.caption(f"Tổng số node: {len(nodes)}")

    with st.expander("Danh sách node"):
        for n in nodes:
            st.write(
                f"{n['node_id']} — {n['path']} | lines {n['start_line']}-{n['end_line']}"
            )

    st.subheader("2. Hỏi tài liệu")

    question = st.text_input(
        "Nhập câu hỏi",
        placeholder="Ví dụ: Báo cáo doanh thu năm 2025 thế nào?",
    )

    top_k = st.slider(
        "Số node retrieval",
        min_value=1,
        max_value=10,
        value=3,
    )

    if st.button("Tìm kiếm"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            selected_nodes = select_relevant_nodes(question, nodes, top_k)
            context_text = build_context_text(selected_nodes)

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown("### 📌 Node được chọn")

                for node in selected_nodes:
                    st.markdown(f"**{node['node_id']} — {node['path']}**")
                    st.caption(
                        f"lines {node['start_line']}-{node['end_line']}"
                    )
                    st.write(node["text"])

            with col2:
                st.markdown("### ✅ Kết quả Gemini")

                if not gemini_key:
                    st.warning("Chưa nhập Gemini API key.")
                else:
                    with st.spinner("Gemini đang trả lời..."):
                        try:
                            answer = ask_gemini(
                                gemini_key,
                                question,
                                context_text,
                            )
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Lỗi khi gọi Gemini: {e}")

            with st.expander("Context đã gửi cho Gemini"):
                st.code(context_text)

else:
    st.warning("Chưa thấy file tree JSON. Hãy upload file hoặc bấm Build Tree trước.")