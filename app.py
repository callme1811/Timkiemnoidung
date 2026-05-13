import streamlit as st
import re
from typing import List, Dict
import os

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
def load_markdown(file_path):
    if not os.path.exists(file_path):
        st.warning(f"❌ File {file_path} không tồn tại, bỏ qua.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    nodes = parse_markdown_to_nodes(text)
    return flatten_tree(nodes)

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
    - Upload thêm file Markdown mới.
    """)
    st.markdown("---")
    st.header("💡 Hướng dẫn")
    st.markdown("""
    1. Upload file Markdown nếu muốn thêm tài liệu.
    2. Nhập câu hỏi ở khung chính.
    3. Chọn số node để phân tích.
    4. Nhấn 'Bắt đầu phân tích' và xem kết quả.
    """)

# ==========================
# Upload thêm tài liệu
# ==========================
uploaded_file = st.sidebar.file_uploader("📂 Upload file Markdown", type=["md"])
uploaded_nodes = []
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    uploaded_nodes = flatten_tree(parse_markdown_to_nodes(text))
    st.sidebar.success(f"✅ Đã tải lên: {uploaded_file.name}")

# ==========================
# Load file mặc định
# ==========================
default_nodes = load_markdown("technova_ai_demo_data.md")
# Gộp nodes mặc định + nodes upload
nodes = default_nodes + uploaded_nodes

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
        for node in nodes[:num_nodes]:
            st.markdown(f"**{node['title']}**")
            st.write(node['text'])

        # ==========================
        # Ollama integration
        # ==========================
        try:
            from ollama import Ollama
            ollama_available = True
        except ModuleNotFoundError:
            ollama_available = False

        if ollama_available:
            try:
                # Sửa model theo Ollama bạn cài
                client = Ollama()
                answer = client.chat(
                    model="qwen",
                    messages=[{"role": "user", "content": f"{query}\n\nContext:\n" + "\n".join([n['text'] for n in nodes[:num_nodes]])}]
                )
                st.subheader("📝 Câu trả lời tự động (Ollama)")
                st.write(answer)
            except Exception as e:
                st.error(f"[OLLAMA_ERROR] {e}")
                st.info("Chạy 'ollama serve' ở terminal khác trước khi dùng.")
        else:
            st.warning("Ollama chưa cài hoặc server chưa chạy, chỉ hiển thị nội dung node Markdown.")

with col2:
    st.header("📂 Dữ liệu nguồn")
    with st.expander("Xem chi tiết các node đã trích xuất", expanded=True):
        for idx, node in enumerate(nodes[:num_nodes]):
            st.markdown(f"**{idx+1}. {node['title']}**")
            st.write(node['text'][:200] + "...")