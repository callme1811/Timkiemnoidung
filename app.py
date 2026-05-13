import streamlit as st
import re
from typing import List, Dict

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
# Load Markdown
# ==========================
@st.cache_data
def load_markdown(file_path="demo_document.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    nodes = parse_markdown_to_nodes(text)
    return flatten_tree(nodes)

nodes = load_markdown()

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
    1. Nhập câu hỏi ở khung chính.
    2. Chọn số node để phân tích.
    3. Nhấn 'Bắt đầu phân tích' và xem kết quả.
    """)

# Main content
st.title("📊 DocAnalyzer - Intelligence Analyzer")

col1, col2 = st.columns([2,1])

# --------------------------
# Cột nhập câu hỏi + trả lời Ollama
# --------------------------
with col1:
    query = st.text_input("Nhập câu hỏi của bạn về tài liệu:", placeholder="Ví dụ: Liệt kê các mục trong báo cáo demo...")
    num_nodes = st.slider("Số node retrieval", 1, 10, 3)
    
    if st.button("🚀 Bắt đầu phân tích") and query:
        st.subheader("📌 Node được chọn")
        for node in nodes[:num_nodes]:
            st.markdown(f"**{node['title']}**")
            st.write(node['text'])

        # Kết nối Ollama
        try:
            from ollama import OllamaClient
            client = OllamaClient(host="127.0.0.1", port=11434)
            answer = client.chat(query, context=[n['text'] for n in nodes[:num_nodes]])
            st.subheader("📝 Câu trả lời")
            st.write(answer)
        except Exception as e:
            st.error(f"[OLLAMA_ERROR] {e}")
            st.info("Chạy 'ollama serve' ở terminal khác trước khi dùng.")

# --------------------------
# Cột dữ liệu nguồn
# --------------------------
with col2:
    st.header("📂 Dữ liệu nguồn")
    with st.expander("Xem chi tiết các node đã trích xuất", expanded=True):
        for idx, node in enumerate(nodes[:num_nodes]):
            st.markdown(f"**{idx+1}. {node['title']}**")
            st.write(node['text'][:200] + "...")  # chỉ show tóm tắt