import json
import re
from pathlib import Path

import requests
import streamlit as st


APP_TITLE = "PageIndex Local Document Search Demo"
DEFAULT_MODEL = "gemma2:2b"

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MD_PATH = BASE_DIR / "/workspaces/Timkiemnoidung/technova_ai_demo_data.md"
RESULTS_DIR = BASE_DIR / "results"


# =========================
# PATH HELPERS
# =========================
def normalize_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else BASE_DIR / p


def get_tree_path(md_path: Path) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / f"{md_path.stem}_structure.json"


# =========================
# MARKDOWN TREE BUILDER
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
                "nodes": []
            })

    return nodes, lines


def add_text_to_nodes(nodes, lines):
    for i, node in enumerate(nodes):
        start = node["line_num"]

        if i + 1 < len(nodes):
            end = nodes[i + 1]["line_num"] - 1
        else:
            end = len(lines)

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
            "nodes": []
        }

        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1]["node"]["nodes"].append(tree_node)
        else:
            root.append(tree_node)

        stack.append({
            "level": node["level"],
            "node": tree_node
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
            "nodes": []
        }]

    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)

    data = {
        "doc_name": md_path.stem,
        "line_count": len(lines),
        "structure": tree
    }

    tree_path = get_tree_path(md_path)
    tree_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return tree_path, data


# =========================
# TREE PROCESSING
# =========================
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
            "text": node.get("text", "")
        })

        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path))

    return result


# =========================
# OLLAMA
# =========================
def ollama_generate(prompt: str, model: str, temperature: float = 0.1):
    url = "http://127.0.0.1:11434/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
    }

    try:
        res = requests.post(url, json=payload, timeout=180)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"


# =========================
# RETRIEVAL
# =========================
def simple_keyword_score(question: str, node_text: str):
    q_words = set(re.findall(r"\w+", question.lower()))
    t_words = set(re.findall(r"\w+", node_text.lower()))

    if not q_words:
        return 0

    return len(q_words & t_words)


def select_relevant_nodes(question, nodes, model, top_k=3, use_ollama=True):
    if not use_ollama:
        ranked = sorted(
            nodes,
            key=lambda n: simple_keyword_score(question, n.get("text", "")),
            reverse=True
        )
        return ranked[:top_k], "keyword fallback"

    node_list = "\n".join([
        f"- {n['node_id']}: {n['path']} "
        f"(lines {n['start_line']}-{n['end_line']})"
        for n in nodes
    ])

    prompt = f"""
Bạn là hệ thống retrieval.

Chọn tối đa {top_k} node phù hợp nhất với câu hỏi.

QUY TẮC:
- Chỉ trả về node_id
- Ngăn cách bằng dấu phẩy
- Không giải thích

CÂU HỎI:
{question}

DANH SÁCH NODE:
{node_list}
"""

    response = ollama_generate(prompt, model=model, temperature=0.0)

    if response.startswith("[OLLAMA_ERROR]"):
        ranked = sorted(
            nodes,
            key=lambda n: simple_keyword_score(question, n.get("text", "")),
            reverse=True
        )
        return ranked[:top_k], response + "\n\nĐã dùng keyword fallback."

    ids = re.findall(r"\b\d{4}\b", response)
    ids = list(dict.fromkeys(ids))[:top_k]

    selected = [n for n in nodes if n["node_id"] in ids]

    if not selected:
        ranked = sorted(
            nodes,
            key=lambda n: simple_keyword_score(question, n.get("text", "")),
            reverse=True
        )
        selected = ranked[:top_k]

    return selected, response


def answer_question(question, selected_nodes, model, use_ollama=True):
    context_text = "\n\n---\n\n".join([
        f"[Node {n['node_id']} | {n['path']} | lines {n['start_line']}-{n['end_line']}]\n\n{n['text']}"
        for n in selected_nodes
    ])

    if not use_ollama:
        return "Đã tìm thấy các đoạn liên quan. Bật Ollama để sinh câu trả lời tự động.", context_text

    prompt = f"""
Bạn là trợ lý tìm kiếm tài liệu.

Chỉ trả lời dựa trên CONTEXT.

Nếu không có thông tin, trả lời:
"Không tìm thấy thông tin trong tài liệu."

CÂU HỎI:
{question}

CONTEXT:
{context_text}

Yêu cầu:
- Trả lời tiếng Việt
- Ngắn gọn
- Chính xác
"""

    answer = ollama_generate(prompt, model=model, temperature=0.1)

    return answer, context_text


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)

st.title(" PageIndex Local Document Search Demo")
st.caption("Markdown → Tree → Retrieval → Ollama/Gemma Answer")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Cấu hình")

    md_path_input = st.text_input(
        "Đường dẫn file Markdown",
        value=str(DEFAULT_MD_PATH)
    )

    model = st.text_input(
        "Ollama model",
        value=DEFAULT_MODEL
    )

    use_ollama = st.checkbox(
        "Dùng Ollama để trả lời",
        value=True
    )

    st.markdown("---")

    build_button = st.button("1. Build Tree")

    st.info(
        "Codespaces dùng Ollama local được. Streamlit Cloud thường không gọi được 127.0.0.1:11434."
    )


md_path = normalize_path(md_path_input)
tree_path = get_tree_path(md_path)


# =========================
# BUILD TREE
# =========================
if build_button:
    st.subheader("Kết quả build")

    if not md_path.exists():
        st.error(f"Không tìm thấy file markdown: {md_path}")
    else:
        try:
            with st.spinner("Đang build tree..."):
                tree_path, tree_data = build_markdown_tree(md_path)

            st.success(f"Build tree thành công: {tree_path}")
            st.code(json.dumps(tree_data, indent=2, ensure_ascii=False)[:3000])

        except Exception as e:
            st.error("Build tree thất bại.")
            st.exception(e)


# =========================
# LOAD TREE
# =========================
st.subheader("2. Load tree structure")

if tree_path.exists():
    st.success(f"Đã tìm thấy tree: {tree_path}")

    tree = json.loads(tree_path.read_text(encoding="utf-8"))

    raw_nodes = tree.get("structure", [])
    nodes = flatten_tree(raw_nodes)

    with st.expander("Danh sách node"):
        for n in nodes:
            st.write(
                f"{n['node_id']} — {n['path']} "
                f"(lines {n['start_line']}-{n['end_line']})"
            )

    st.subheader("3. Hỏi tài liệu")

    question = st.text_input(
        "Nhập câu hỏi",
        value="Quy trình xử lý sự cố production gồm những bước nào?"
    )

    top_k = st.slider(
        "Số node retrieval",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button("Tìm kiếm và trả lời"):
        with st.spinner("Đang tìm kiếm..."):
            selected_nodes, raw_selection = select_relevant_nodes(
                question=question,
                nodes=nodes,
                model=model,
                top_k=top_k,
                use_ollama=use_ollama
            )

            answer, context_text = answer_question(
                question=question,
                selected_nodes=selected_nodes,
                model=model,
                use_ollama=use_ollama
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Node được chọn")
            st.code(raw_selection)

            for node in selected_nodes:
                st.markdown(f"**{node['node_id']} — {node['path']}**")
                st.caption(f"lines {node['start_line']}-{node['end_line']}")

        with col2:
            st.markdown("### Câu trả lời")
            st.write(answer)

        with st.expander("Context đã dùng"):
            st.code(context_text)

else:
    st.warning(
        "Chưa thấy file tree JSON. Hãy bấm Build Tree trước."
    )