import streamlit as st
import re
from typing import List, Dict

# ==========================
# Chia Markdown thành node theo heading
# ==========================
def parse_markdown_to_nodes(md_text: str) -> List[Dict]:
    """
    Tách Markdown thành node:
    - # -> cấp 1
    - ## -> cấp 2
    - ### -> cấp 3
    """
    nodes = []
    lines = md_text.split("\n")
    stack = []  # lưu node cấp trên

    for line in lines:
        header_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            node = {"title": title, "text": "", "nodes": []}

            # Xác định cấp cha dựa vào stack
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                # node này là con của node trên stack
                stack[-1][1]["nodes"].append(node)
            else:
                nodes.append(node)

            stack.append((level, node))
        else:
            # Gán text vào node cuối cùng trên stack
            if stack:
                stack[-1][1]["text"] += line + "\n"
            else:
                # nếu chưa có header, tạo node default
                if not nodes:
                    nodes.append({"title": "Document", "text": line + "\n", "nodes": []})
                else:
                    nodes[0]["text"] += line + "\n"

    return nodes

# ==========================
# Flatten tree node thành list
# ==========================
def flatten_tree(nodes: List[Dict], parent_path="") -> List[Dict]:
    result = []
    for node in nodes:
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title
        result.append({
            "title": title,
            "text": node.get("text", "")
        })
        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path))
    return result

# ==========================
# Load file Markdown
# ==========================
@st.cache_data
def load_markdown(file_path="technova_ai_demo_data.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    nodes = parse_markdown_to_nodes(text)
    return flatten_tree(nodes)

nodes = load_markdown()

# ==========================
# Streamlit UI
# ==========================
st.title("TechNova AI - Document QA")

query = st.text_input("Nhập câu hỏi của bạn:")

num_nodes = st.slider("Số node retrieval", 1, 10, 3)

if st.button("Tìm kiếm và trả lời") and query:
    st.write("### Node được chọn:")
    for node in nodes[:num_nodes]:
        st.markdown(f"**{node['title']}**")
        st.write(node['text'])

    # ==========================
    # Ollama
    # ==========================
    try:
        from ollama import OllamaClient
        client = OllamaClient(host="127.0.0.1", port=11434)
        answer = client.chat(query, context=[n['text'] for n in nodes[:num_nodes]])
        st.write("### Câu trả lời:")
        st.write(answer)
    except Exception as e:
        st.error(f"[OLLAMA_ERROR] {e}")
        st.info("Đừng quên chạy 'ollama serve' ở terminal khác trước khi dùng.")