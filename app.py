import json
import re
import subprocess
from pathlib import Path

import requests
# pyrefly: ignore [missing-import]
import streamlit as st


APP_TITLE = "PageIndex Local Document Search Demo"
DEFAULT_MODEL = "gemma2:2b"


def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    """Call local Ollama server using OpenAI-compatible API."""
    url = "http://127.0.0.1:11434/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": temperature
    }

    try:
        res = requests.post(url, json=payload, timeout=180)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"


def read_markdown_lines(md_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        return f.readlines()


def load_tree(tree_path: str):
    with open(tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def find_tree_output_path(md_path: str):
    stem = Path(md_path).stem
    return Path("./results") / f"{stem}_structure.json"


def build_pageindex_tree(md_path: str, model: str):
    cmd = [
        "py",
        "run_pageindex.py",
        "--md_path",
        md_path,
        "--model",
        f"ollama/{model}",
    ]

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False,
    )

    return process.returncode, process.stdout, process.stderr


def select_relevant_nodes(question: str, nodes, model: str, top_k: int = 3):
    node_list = "\n".join([
        f"- {n['node_id']}: {n['path']} (lines {n['start_line']}-{n['end_line']})"
        for n in nodes
    ])

    prompt = f"""
Bạn là hệ thống retrieval theo kiểu PageIndex.

Nhiệm vụ:
Chọn tối đa {top_k} node phù hợp nhất để trả lời câu hỏi.

Quy tắc:
- Chỉ trả về node_id.
- Các node_id phân tách bằng dấu phẩy.
- Không giải thích.
- Không thêm văn bản khác.

Câu hỏi:
{question}

Danh sách node:
{node_list}

Ví dụ output:
0006, 0007, 0011
"""

    response = ollama_generate(prompt, model=model, temperature=0.0)

    ids = re.findall(r"\b\d{4}\b", response)
    ids = list(dict.fromkeys(ids))[:top_k]
    selected = [n for n in nodes if n["node_id"] in ids]

    if not selected:
        q = question.lower()
        selected = [
            n for n in nodes
            if any(word in n["path"].lower() for word in q.split() if len(word) > 3)
        ][:top_k]

    return selected, response


def answer_question(question: str, selected_nodes, md_lines, model: str):
    contexts = []

    for node in selected_nodes:
        text = extract_node_text(md_lines, node)
        contexts.append(
            f"[Node {node['node_id']} | {node['path']} | "
            f"lines {node['start_line']}-{node['end_line']}]\n{text}"
        )

    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""
Bạn là trợ lý tìm kiếm tài liệu nội bộ.

Chỉ trả lời dựa trên CONTEXT bên dưới.
Nếu không đủ thông tin, hãy nói:
"Không tìm thấy thông tin trong tài liệu."

CÂU HỎI:
{question}

CONTEXT:
{context_text}

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Ngắn gọn, rõ ràng.
- Có nhắc node hoặc section đã dùng.
- Không bịa thông tin ngoài context.
"""

    answer = ollama_generate(prompt, model=model, temperature=0.1)
    return answer, context_text


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title("🔎 PageIndex Local Document Search Demo")
st.caption("Markdown → PageIndex Tree → Ollama/Gemma Reasoning Retrieval → Answer")


with st.sidebar:
    st.header("Cấu hình")

    md_path = st.text_input(
        "Đường dẫn file Markdown",
        value=r"D:\Practice\Rag\TimKiemUngDung\Timkiemnoidung\technova_ai_demo_data.md",
    )

    model = st.text_input(
        "Ollama model",
        value=DEFAULT_MODEL,
        help="Ví dụ: gemma2:9b, qwen2.5:7b",
    )

    st.markdown("---")
    build_button = st.button("1. Build PageIndex Tree")

    st.markdown("### Lưu ý")
    st.write("Ollama cần đang chạy ở `127.0.0.1:11434`.")
    st.write("App phải đặt trong thư mục PageIndex, cùng cấp với `run_pageindex.py`.")


if build_button:
    if not Path(md_path).exists():
        st.error("Không tìm thấy file Markdown.")
    else:
        with st.spinner("Đang build tree bằng PageIndex..."):
            code, stdout, stderr = build_pageindex_tree(md_path, model)

        st.subheader("Kết quả build")
        st.code(stdout or "(không có stdout)", language="text")

        if stderr:
            st.code(stderr, language="text")

        if code == 0:
            st.success("Build tree thành công.")
        else:
            st.error("Build tree thất bại.")


tree_path = find_tree_output_path(md_path)

st.subheader("2. Load tree structure")

if tree_path.exists():
    st.success(f"Đã tìm thấy tree: {tree_path}")

    tree = load_tree(str(tree_path))
    md_lines = read_markdown_lines(md_path)

    raw_nodes = tree.get("structure", [])
    flat_nodes = flatten_nodes(raw_nodes)
    nodes = add_line_ranges(flat_nodes, len(md_lines))

    with st.expander("Xem danh sách node"):
        for n in nodes:
            st.write(
                f"**{n['node_id']}** — {n['path']} "
                f"— lines {n['start_line']}-{n['end_line']}"
            )

    st.subheader("3. Hỏi tài liệu")

    question = st.text_input(
        "Nhập câu hỏi",
        value="Quy trình xử lý sự cố production gồm những bước nào?",
    )

    top_k = st.slider("Số node truy xuất", min_value=1, max_value=5, value=3)

    if st.button("Tìm kiếm và trả lời"):
        with st.spinner("Đang reasoning trên tree..."):
            selected_nodes, raw_selection = select_relevant_nodes(
                question, nodes, model, top_k=top_k
            )
            answer, context_text = answer_question(
                question, selected_nodes, md_lines, model
            )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Node được chọn")
            st.code(raw_selection, language="text")

            if selected_nodes:
                for node in selected_nodes:
                    st.markdown(f"**{node['node_id']} — {node['path']}**")
                    st.caption(f"lines {node['start_line']}-{node['end_line']}")
            else:
                st.warning("Không chọn được node phù hợp.")

        with col2:
            st.markdown("### Câu trả lời")
            st.write(answer)

        with st.expander("Context đã dùng"):
            st.code(context_text, language="markdown")

else:
    st.warning("Chưa thấy file tree JSON. Hãy bấm Build PageIndex Tree trước.")