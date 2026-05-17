# app_full_enhanced.py

import re
import time
import hashlib
from pathlib import Path
import os

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image
from realesrgan import RealESRGAN

# ====================== Cấu hình cơ bản ======================
APP_TITLE = "DocAnalyzer AI + ECG Enhancer"
BASE_DIR = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 700
TOP_K = 3

# ====================== Gemini API ======================
GEMINI_API_KEY = "YOUR_REAL_KEY_HERE"  # Thay bằng key thật
genai.configure(api_key=GEMINI_API_KEY)

# ====================== Session State ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# ====================== Real-ESRGAN ======================
@st.cache_resource
def load_sr_model():
    model = RealESRGAN('cuda', scale=2)  # 'cpu' nếu không có GPU
    model.load_weights('weights/RealESRGAN_x4plus_anime_6B.pth')
    return model

sr_model = load_sr_model()

def enhance_image(image_path):
    img = Image.open(image_path).convert('RGB')
    sr_img = sr_model.predict(img)
    output_path = OUTPUT_DIR / f"sr_{Path(image_path).name}"
    sr_img.save(output_path)
    return output_path, sr_img

# ====================== Helper functions ======================
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

# ====================== DocAnalyzer AI ======================
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
        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk = f"{current_chunk}\n{para}" if current_chunk else para
        else:
            if current_chunk.strip(): chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = max(end - overlap, start + 1)
            else:
                current_chunk = para
    if current_chunk.strip(): chunks.append(current_chunk.strip())
    return chunks

def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    nodes = []
    counter = 0
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks = split_text(page_text)
        for chunk_index, chunk in enumerate(chunks, start=1):
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

def parse_document(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf": return extract_pdf_text(file_path)
    if suffix in [".txt", ".md", ".markdown"]: return extract_txt_nodes(file_path)
    return []

@st.cache_data(show_spinner=False)
def parse_uploaded_files_cached(file_paths):
    all_nodes = []
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        nodes = parse_document(file_path)
        all_nodes.extend(nodes)
    return all_nodes

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def keyword_score(question, text, title=""):
    q_words = re.findall(r"\w+", normalize_text(question))
    content = normalize_text(f"{title} {text}")
    score = 0
    for word in q_words:
        if word in content: score += 2
    if normalize_text(question) in content: score += 5
    return score

def select_relevant_nodes(question, nodes, top_k=TOP_K):
    ranked = sorted(nodes, key=lambda n: keyword_score(n.get("text", ""), n.get("title", ""), question), reverse=True)
    selected = [n for n in ranked if keyword_score(n.get("text", ""), n.get("title", ""), question) > 0]
    return selected[:top_k] if selected else ranked[:top_k]

def build_context(selected_nodes):
    return "\n\n".join([n.get("text", "") for n in selected_nodes])

def build_prompt(question, context_text):
    return f"""
Bạn là AI chuyên phân tích tài liệu.
NHIỆM VỤ:
- Chỉ trả lời dựa trên CONTEXT được cung cấp.
- Không bịa thông tin ngoài tài liệu.
- Giải thích dễ hiểu.
- Nếu thiếu thông tin, nói rõ tài liệu không cung cấp.
- Trả lời ngắn gọn, tối đa 500 từ.

QUESTION:
{question}

CONTEXT:
{context_text}
"""

def ask_gemini(question, context_text):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = build_prompt(question, context_text)
    response = model.generate_content(prompt, generation_config={"temperature":0.25, "max_output_tokens":MAX_OUTPUT_TOKENS})
    return clean_answer(getattr(response, "text", ""))

# ====================== Streamlit UI ======================
st.title(APP_TITLE)
st.caption("📄 DocAnalyzer AI + Làm nét ECG")

uploaded_files = st.file_uploader("Upload ảnh hoặc PDF/TXT/Markdown", type=["png","jpg","jpeg","pdf","txt","md","markdown"], accept_multiple_files=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Hỏi nội dung tài liệu...")

if uploaded_files:
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        saved_paths.append(str(file_path))

        # ---------------------- Bổ sung Real-ESRGAN ----------------------
        if uploaded_file.type.startswith("image"):
            out_path, sr_img = enhance_image(file_path)
            st.image(sr_img, caption=f"Ảnh đã làm nét: {uploaded_file.name}", use_column_width=True)
        elif uploaded_file.type == "application/pdf":
            st.warning("PDF sẽ parse nội dung text, không làm nét trực tiếp.")
        # -------------------------------------------------------------------

    if question:
        st.session_state.messages.append({"role":"user","content":question})
        if question not in st.session_state.question_history:
            st.session_state.question_history.append(question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("📖 Đang đọc tài liệu..."):
            all_nodes = parse_uploaded_files_cached(tuple(saved_paths))
            if not all_nodes:
                st.error("Không đọc được nội dung tài liệu. Nếu PDF là scan ảnh, cần OCR.")
                st.stop()
            selected_nodes = select_relevant_nodes(question, all_nodes, top_k=TOP_K)
            context_text = build_context(selected_nodes)
            st.session_state.last_context = context_text

        with st.chat_message("assistant"):
            try:
                with st.spinner("🤖 Gemini đang trả lời..."):
                    answer = ask_gemini(question, context_text)
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role":"assistant","content":answer})
            except Exception as e:
                st.error(f"Lỗi Gemini: {e}")

        with st.expander("📚 Context đã dùng"):
            st.code(context_text)