import streamlit as st
import re
from typing import List, Dict
import os
import subprocess

# ==========================
# Chia Markdown thành node theo heading
# ==========================
def parse_markdown_to_nodes(md_text: str) -> List[Dict]:
    nodes = []
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        header_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            node = {"title": title, "text": "", "nodes": []}
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                stack[-1][1]["nodes"].append(node)
            else:
                nodes.append(node)
            stack.append((level, node))
        else:
            if stack:
                stack[-1][1]["text"] += line + "\n"
            else:
                if not nodes:
                    nodes.append({"title": "Document", "text": line + "\n", "nodes": []})
                else:
                    nodes[0]["text"] += line + "\n"
    return nodes

# ==========================
# Flatten tree node
# ==========================
def flatten_tree(nodes: List[Dict], parent_path="") -> List[Dict]:
    result = []
    for node in nodes:
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title
        result.append({"title": title, "text": node.get("text", "")})
        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path))
    return result

# ==========================
# Load Markdown từ file
# ==========================
def load_markdown_from_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        st.error(f"❌ File {file_path} không tồn tại.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    nodes = parse_markdown_to_nodes(text)
    return flatten_tree(nodes)

# ==========================
# Ollama query subprocess
# ==========================
def query_ollama(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "chat", "qwen", "--prompt", prompt],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}"

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="DocAnalyzer - Intelligence Analyzer", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📄 DocAnalyzer")
    st.markdown("Hệ thống phân tích nội dung tài liệu")
    st.markdown("---")
    st.header("⚡ Chức năng nhanh")
    st.markdown("""
    - Nhập câu hỏi về tài liệu.
    - Xem các node liên quan.
    - Truy xuất Top N node làm context.
    """)
    st.markdown("---")
    st.header("💡 Hướng dẫn")
    st.markdown("""
    1. Upload file Markdown mới nếu cần.
    2. Nhập câu hỏi ở khung chính.
    3. Chọn số node để phân tích.
    4. Nhấn 'Bắt đầu phân tích' và xem kết quả.
    """)

# ==========================
# Upload file Markdown mới
# ==========================
uploaded_file = st.sidebar.file_uploader("📂 Upload file Markdown", type=["md"])
all_nodes = []

# Load file mặc định
default_file = "technova_ai_demo_data.md"
all_nodes.extend(load_markdown_from_file(default_file))

# Load file upload nếu có
if uploaded_file is not None:
    uploaded_text = uploaded_file.getvalue().decode("utf-8")
    uploaded_nodes = flatten_tree(parse_markdown_to_nodes(uploaded_text))
    all_nodes.extend(uploaded_nodes)
    st.sidebar.success(f"✅ Đã load {len(uploaded_nodes)} node từ file upload.")

# ==========================
# Main content
# ==========================
st.title("📊 DocAnalyzer - Intelligence Analyzer")
col1, col2 = st.columns([2,1])

with col1:
    query = st.text_input("Nhập câu hỏi của bạn về tài liệu:", placeholder="Ví dụ: Liệt kê các mục trong báo cáo demo...")
    num_nodes = st.slider("Số node retrieval", 1, 10, 3)

    if st.button("🚀 Bắt đầu phân tích") and query:
        st.subheader("📌 Node được chọn")
        for node in all_nodes[:num_nodes]:
            st.markdown(f"**{node['title']}**")
            st.write(node['text'])

        # Chuẩn bị prompt
        context_text = "\n\n".join([n['text'] for n in all_nodes[:num_nodes]])
        full_prompt = f"{query}\n\n{context_text}"

        # Gọi Ollama
        answer = query_ollama(full_prompt)
        st.subheader("📝 Câu trả lời tự động (Ollama)")
        st.write(answer)

with col2:
    st.header("📂 Dữ liệu nguồn")
    with st.expander("Xem chi tiết các node đã trích xuất", expanded=True):
        for idx, node in enumerate(all_nodes[:num_nodes]):
            st.markdown(f"**{idx+1}. {node['title']}**")
            st.write(node['text'][:200] + "...")  # chỉ show tóm tắt