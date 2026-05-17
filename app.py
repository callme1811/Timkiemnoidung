import re
import time
import hashlib
from pathlib import Path
import os
from PIL import Image

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# Thử import RealESRGAN, nếu thiếu báo lỗi
try:
    from realesrgan import RealESRGAN
except ImportError:
    st.error("Thiếu thư viện 'realesrgan'. Cài đặt pip install realesrgan-ncnn-py hoặc bản torch")
    st.stop()

# ====================== CẤU HÌNH ======================
APP_TITLE = "DocAnalyzer AI"
BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

# ====================== GEMINI AI ======================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 700
TOP_K = 3

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    st.warning("API key chưa cấu hình đúng. Một số tính năng có thể không hoạt động.")

genai.configure(api_key=GEMINI_API_KEY)

# ====================== SESSION STATE ======================
for key in ["messages", "question_history", "last_context", "processed_images"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key=="processed_images" else []

# ====================== REAL-ESRGAN ======================
@st.cache_resource
def load_sr_model():
    try:
        model = RealESRGAN('cuda', scale=2)
        model.load_weights('weights/RealESRGAN_x4plus_anime_6B.pth')
    except Exception:
        model = RealESRGAN('cpu', scale=2)
        model.load_weights('weights/RealESRGAN_x4plus_anime_6B.pth')
    return model

try:
    sr_model = load_sr_model()
except Exception as e:
    sr_model = None
    st.sidebar.warning(f"Không thể nạp RealESRGAN: {e}")

def enhance_image(image_path, scale=2):
    if sr_model is None:
        return image_path, Image.open(image_path)
    img = Image.open(image_path).convert('RGB')
    sr_img = sr_model.predict(img)
    output_path = OUTPUT_DIR / f"sr_{Path(image_path).name}"
    sr_img.save(output_path)
    return output_path, sr_img

# ====================== HỖ TRỢ FILE ======================
def clean_answer(text):
    text = re.sub(r"\[SOURCE\s*\d+\]", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip()

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def save_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    save_path = UPLOADS_DIR / f"{file_hash[:10]}_{safe_name}"
    if not save_path.exists():
        save_path.write_bytes(file_bytes)
    return save_path

# ====================== TRÍCH XUẤT VĂN BẢN ======================
def split_text(text, chunk_size=1000, overlap=120):
    text = text.strip()
    if not text: return []
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(current_chunk)+len(para)+1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{para}" if current_chunk else para
        else:
            if current_chunk: chunks.append(current_chunk)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start+chunk_size
                    chunks.append(para[start:end])
                    start = max(end-overlap, start+1)
            else:
                current_chunk = para
    if current_chunk: chunks.append(current_chunk)
    return chunks

def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    nodes = []
    counter = 0
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_index, chunk in enumerate(split_text(page_text), start=1):
            counter += 1
            nodes.append({
                "node_id": str(counter).zfill(4),
                "title": f"Page {page_index}",
                "path": f"{pdf_path.name} > Page {page_index} > Chunk {chunk_index}",
                "source_file": pdf_path.name,
                "page": page_index,
                "chunk": chunk_index,
                "text": chunk,
            })
    return nodes

def extract_txt_nodes(file_path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text(text)
    nodes = []
    for i, chunk in enumerate(chunks, start=1):
        nodes.append({
            "node_id": str(i).zfill(4),
            "title": file_path.name,
            "path": f"{file_path.name} > Chunk {i}",
            "source_file": file_path.name,
            "page": None,
            "chunk": i,
            "text": chunk,
        })
    return nodes

def extract_markdown_nodes(md_text):
    lines = md_text.splitlines()
    nodes = []
    in_code_block = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block: continue
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
        end = nodes[i+1]["line_num"]-1 if i+1<len(nodes) else len(lines)
        node["start_line"] = start
        node["end_line"] = end
        node["text"] = "\n".join(lines[start-1:end]).strip()
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
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "text": node["text"],
            "nodes": [],
        }
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["node"]["nodes"].append(tree_node)
        else:
            root.append(tree_node)
        stack.append({"level": node["level"], "node": tree_node})
    return root

def flatten_tree(nodes, parent_path="", source_file=""):
    result = []
    for node in nodes:
        title = node.get("title","")
        path = f"{parent_path} > {title}" if parent_path else title
        for chunk_index, chunk in enumerate(split_text(node.get("text","")), start=1):
            result.append({
                "node_id": node.get("node_id",""),
                "title": title,
                "path": f"{source_file} > {path} > Chunk {chunk_index}",
                "source_file": source_file,
                "page": None,
                "chunk": chunk_index,
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
                "text": chunk,
            })
        if node.get("nodes"):
            result.extend(flatten_tree(node["nodes"], path, source_file))
    return result

def parse_markdown(file_path):
    md_text = file_path.read_text(encoding="utf-8", errors="ignore")
    flat_nodes, lines = extract_markdown_nodes(md_text)
    if not flat_nodes: return extract_txt_nodes(file_path)
    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)
    return flatten_tree(tree, source_file=file_path.name)

def parse_document(file_path):
    suffix = file_path.suffix.lower()
    if suffix==".pdf": return extract_pdf_text(file_path)
    if suffix in [".txt"]: return extract_txt_nodes(file_path)
    if suffix in [".md", ".markdown"]: return parse_markdown(file_path)
    return []

@st.cache_data(show_spinner=False)
def parse_uploaded_files_cached(file_paths):
    all_nodes = []
    for file_path_str in file_paths:
        nodes = parse_document(Path(file_path_str))
        all_nodes.extend(nodes)
    return all_nodes

# ====================== KEYWORD SEARCH ======================
def normalize_text(text):
    text = re.sub(r"[^\w\s]"," ", text.lower())
    text = re.sub(r"\s+"," ", text).strip()
    return text

def keyword_score(question, text, title=""):
    q_words = re.findall(r"\w+", normalize_text(question))
    content = normalize_text(f"{title} {text}")
    score = sum(2 for w in q_words if w in content)
    if normalize_text(question) in content:
        score += 5
    return score

def select_relevant_nodes(question, nodes, top_k=TOP_K):
    if not nodes: return []
    ranked = sorted(nodes, key=lambda n: keyword_score(question, n.get("text",""), n.get("title","")), reverse=True)
    selected = [n for n in ranked if keyword_score(question, n.get("text",""), n.get("title",""))>0]
    if not selected: selected=ranked
    return selected[:top_k]

def build_context(selected_nodes):
    return "\n\n".join([n.get("text","") for n in selected_nodes])

# ====================== GEMINI PROMPT ======================
def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.

NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không được bịa thông tin ngoài tài liệu.
- Giải thích dễ hiểu cho người mới.
- Nếu thiếu thông tin, nói rõ tài liệu không cung cấp.
- Không hiển thị SOURCE.
- Không nhắc [SOURCE 1], [SOURCE 2], [SOURCE 3].
- Trả lời ngắn gọn, tối đa 500 từ.

QUESTION:
{question}

CONTEXT:
{context_text}
"""

def ask_gemini(question, context_text):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = build_prompt(question, context_text)
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt, generation_config={"temperature":0.25,"max_output_tokens":MAX_OUTPUT_TOKENS}, stream=False)
            answer = getattr(response,"text","") or ""
            answer = clean_answer(answer)
            if answer.strip(): return answer
            raise Exception("Gemini không trả về nội dung.")
        except Exception as e:
            err=str(e)
            if "503" in err or "overloaded" in err.lower(): time.sleep(2*(attempt+1)); continue
            if "429" in err: raise Exception("Gemini hết quota hoặc giới hạn tốc độ.")
            if "403" in err: raise Exception("API key hoặc project Gemini không có quyền truy cập model.")
            raise e
    raise Exception("Gemini quá tải sau nhiều lần thử.")

# ====================== STREAMLIT UI & CSS ======================
st.markdown("""
<style>
.block-container{ max-width:1200px; padding-top:25px; }
.stButton button{ width:100%; height:50px; border-radius:14px; font-size:16px; font-weight:700; }
.answer-box{ padding:24px; border-radius:18px; background:#111827; border:1px solid #374151; margin-top:10px; line-height:1.7; color:#f3f4f6;}
.history-box{ padding:10px; border-radius:12px; margin-bottom:8px; background:#111827; border:1px solid #374151; font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("📚 Lịch sử")
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.messages=[]
        st.session_state.question_history=[]
        st.session_state.last_context=""
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    history_list=st.session_state.get("question_history",[])
    if history_list:
        for i,q in enumerate(history_list[::-1],start=1):
            st.markdown(f"<div class='history-box'><b>{i}.</b> {q}</div>",unsafe_allow_html=True)
    else:
        st.markdown("<div class='history-box'>Chưa có câu hỏi nào</div>",unsafe_allow_html=True)

# ====================== MAIN ======================
st.title("📄 DocAnalyzer AI")
st.caption("Chat với PDF, Markdown và TXT bằng Gemini 2.5")

uploaded_files=st.file_uploader("📂 Tải tài liệu lên", type=["pdf","md","markdown","txt"], accept_multiple_files=True)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

question = st.chat_input("Hỏi nội dung tài liệu...")

if question:
    if not uploaded_files:
        st.warning("Vui lòng upload tài liệu."); st.stop()
    st.session_state.messages.append({"role":"user","content":question})
    if question not in st.session_state.question_history: st.session_state.question_history.append(question)
    with st.chat_message("user"): st.markdown(question)
    with st.spinner("📖 Đang đọc tài liệu..."):
        saved_paths=[]
        for uploaded_file in uploaded_files: saved_paths.append(str(save_uploaded_file(uploaded_file)))
        all_nodes=parse_uploaded_files_cached(tuple(saved_paths))
    if not all_nodes:
        st.error("Không đọc được nội dung tài liệu. Nếu PDF là dạng scan ảnh, cần OCR."); st.stop()
    selected_nodes=select_relevant_nodes(question, all_nodes, top_k=TOP_K)
    context_text=build_context(selected_nodes)
    st.session_state.last_context=context_text
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤖 Gemini đang trả lời..."):
                answer=ask_gemini(question,context_text)
            st.markdown(f"<div class='answer-box'>{answer}</div>",unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":answer})
        except Exception as e:
            st.error(f"Lỗi Gemini: {e}")
    with st.expander("📚 Xem context đã dùng"): st.code(context_text)