import streamlit as st
from typing import List, Dict

# ==========================
# Hàm flatten tree (chỉ lấy title + text)
# ==========================
def flatten_tree(nodes: List[Dict], parent_path="") -> List[Dict]:
    result = []
    for node in nodes:
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title
        # Chỉ lấy title và text
        result.append({
            "title": title,
            "text": node.get("text", "")
        })
        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path))
    return result

# ==========================
# Load dữ liệu demo (có thể sửa thành đọc file JSON của bạn)
# ==========================
import json

@st.cache_data
def load_demo_data():
    with open("/workspaces/Timkiemnoidung/technova_ai_demo_data.md", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_demo_data()
nodes = flatten_tree(data["structure"])

# ==========================
# Giao diện Streamlit
# ==========================
st.title("TechNova AI - Document QA")

# Nhập câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:")

# Chọn số node lấy ra
num_nodes = st.slider("Số node retrieval", 1, 10, 3)

if st.button("Tìm kiếm và trả lời") and query:
    st.write("### Node được chọn:")
    for node in nodes[:num_nodes]:
        st.markdown(f"**{node['title']}**")
        st.write(node['text'])

    # ==========================
    # Kết nối Ollama (bạn phải chạy 'ollama serve' trước)
    # ==========================
    try:
        from ollama import OllamaClient
        client = OllamaClient(host="127.0.0.1", port=11434)  # sửa nếu cần
        answer = client.chat(query, context=[n['text'] for n in nodes[:num_nodes]])
        st.write("### Câu trả lời:")
        st.write(answer)
    except Exception as e:
        st.error(f"[OLLAMA_ERROR] {e}")
        st.info("Đừng quên chạy 'ollama serve' ở terminal khác trước khi dùng.")