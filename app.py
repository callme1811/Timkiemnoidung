import json
import re
import subprocess
from pathlib import Path

import requests
import streamlit as st


APP_TITLE = "PageIndex Local Document Search Demo"
DEFAULT_MODEL = "gemma2:2b"

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MD_PATH = BASE_DIR / "technova_ai_demo_data.md"
RESULTS_DIR = BASE_DIR / "results"


# =========================
# OLLAMA
# =========================
def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1):
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
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"


# =========================
# FILE HELPERS
# =========================
def normalize_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()

    if p.is_absolute():
        return p

    return BASE_DIR / p


def read_markdown_lines(md_path: Path):
    with open(md_path, "r", encoding="utf-8") as f:
        return f.readlines()


def load_tree(tree_path: Path):
    with open(tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_tree_output_path(md_path: Path):
    stem = md_path.stem
    return RESULTS_DIR / f"{stem}_structure.json"


# =========================
# PAGEINDEX BUILD
# =========================
def build_pageindex_tree(md_path: Path, model: str):
    run_script = BASE_DIR / "run_pageindex.py"

    cmd = [
        "python3",
        str(run_script),
        "--md_path",
        str(md_path),
        "--model",
        f"ollama/{model}",
    ]

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False,
        cwd=str(BASE_DIR),
    )

    return process.returncode, process.stdout, process.stderr


# =========================
# TREE PROCESSING
# =========================
def flatten_nodes(nodes, parent_path=""):
    result = []

    for node in nodes:
        title = node.get("title", "")
        node_id = node.get("node_id", "")
        line_num = node.get("line_num", None)

        path = f"{parent_path} > {title}" if parent_path else title

        result.append({
            "node_id": node_id,
            "title": title,
            "line_num": line_num,
            "path": path,
        })

        if "nodes" in node:
            result.extend(flatten_nodes(node["nodes"], path))

    return result


def add_line_ranges(flat_nodes, total_lines):
    sorted_nodes = sorted(
        [n for n in flat_nodes if n.get("line_num") is not None],
        key=lambda x: x["line_num"],
    )

    for i, node in enumerate(sorted_nodes):
        start = node["line_num"]

        if i + 1 < len(sorted_nodes):
            end = sorted_nodes[i + 1]["line_num"] - 1
        else:
            end = total_lines

        node["start_line"] = start
        node["end_line"] = end

    return sorted_nodes


def extract_node_text(md_lines, node):
    start = max(node["start_line"] - 1, 0)
    end = min(node["end_line"], len(md_lines))
    return "".join(md_lines[start:end]).strip()


# =========================
# RETRIEVAL
# =========================
def select_relevant_nodes(question, nodes, model, top_k=3):
    node_list = "\n".join([
        f"- {n['node_id']}: {n['path']} "
        f"(lines {n['start_line']}-{n['end_line']})"
        for n in nodes
    ])

    prompt = f"""
Bạn là hệ thống retrieval PageIndex.

Hãy chọn tối đa {top_k} node phù hợp nhất.

QUY TẮC:
- Chỉ trả về node_id
- Ngăn cách bằng dấu phẩy
- Không giải thích

CÂU HỎI:
{question}

DANH SÁCH NODE:
{node_list}

Ví dụ:
0006, 0011
"""

    response = ollama_generate(prompt, model=model, temperature=0.0)

    ids = re.findall(r"\b\d{4}\b", response)
    ids = list(dict.fromkeys(ids))[:top_k]

    selected = [n for n in nodes if n["node_id"] in ids]

    return selected, response


# =========================
# ANSWER
# =========================
def answer_question(question, selected_nodes, md_lines, model):
    contexts = []

    for node in selected_nodes:
        text = extract_node_text(md_lines, node)

        contexts.append(
            f"[Node {node['node_id']} | "
            f"{node['path']} | "
            f"lines {node['start_line']}-{node['end_line']}]\n\n{text}"
        )

    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""
Bạn là trợ lý tìm kiếm tài liệu.

Chỉ trả lời dựa trên CONTEXT.

Nếu không có thông tin:
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

st.title("🔎 PageIndex Local Document Search Demo")

st.caption(
    "Markdown → PageIndex Tree → Ollama/Gemma Reasoning → Answer"
)


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

    st.markdown("---")

    build_button = st.button(
        "1. Build PageIndex Tree"
    )

    st.info(
        "Ollama phải chạy ở 127.0.0.1:11434"
    )


md_path = normalize_path(md_path_input)
tree_path = find_tree_output_path(md_path)


# =========================
# BUILD TREE
# =========================
if build_button:
    if not md_path.exists():
        st.error(f"Không tìm thấy file markdown: {md_path}")
    else:
        with st.spinner("Đang build tree..."):
            code, stdout, stderr = build_pageindex_tree(md_path, model)

        st.subheader("Kết quả build")

        st.code(stdout or "(không có output)")

        if stderr:
            st.code(stderr)

        if code == 0:
            st.success("Build tree thành công.")
        else:
            st.error("Build tree thất bại.")


# =========================
# LOAD TREE
# =========================
st.subheader("2. Load tree structure")

if tree_path.exists():
    st.success(f"Đã tìm thấy tree: {tree_path}")

    tree = load_tree(tree_path)
    md_lines = read_markdown_lines(md_path)

    raw_nodes = tree.get("structure", [])

    flat_nodes = flatten_nodes(raw_nodes)

    nodes = add_line_ranges(
        flat_nodes,
        len(md_lines)
    )

    with st.expander("Danh sách node"):
        for n in nodes:
            st.write(
                f"{n['node_id']} — "
                f"{n['path']} "
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
        with st.spinner("Đang reasoning..."):
            selected_nodes, raw_selection = select_relevant_nodes(
                question,
                nodes,
                model,
                top_k
            )

            answer, context_text = answer_question(
                question,
                selected_nodes,
                md_lines,
                model
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Node được chọn")
            st.code(raw_selection)

            for node in selected_nodes:
                st.markdown(
                    f"**{node['node_id']} — {node['path']}**"
                )
                st.caption(
                    f"lines {node['start_line']}-{node['end_line']}"
                )

        with col2:
            st.markdown("### Câu trả lời")
            st.write(answer)

        with st.expander("Context đã dùng"):
            st.code(context_text)

else:
    st.warning(
        "Chưa thấy file tree JSON. "
        "Hãy bấm Build PageIndex Tree trước."
    )