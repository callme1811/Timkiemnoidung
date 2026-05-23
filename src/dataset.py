import re
from pathlib import Path

from pypdf import PdfReader

from .preprocessing import split_text


def extract_pdf_text_and_images(pdf_path, uploads_dir: Path, existing_images: list[str] | None = None):
    reader = PdfReader(str(pdf_path))
    nodes = []
    counter = 0
    scanned_pages = []
    extracted_images = list(existing_images or [])

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if not page_text.strip() or len(page_text.strip()) < 50:
            scanned_pages.append(page_index)
            for img_idx, img_file in enumerate(page.images, start=1):
                try:
                    img_bytes = img_file.data
                    ext = Path(img_file.name).suffix or ".png"
                    img_name = f"{Path(pdf_path).stem}_page{page_index}_img{img_idx}{ext}"
                    img_path = uploads_dir / img_name
                    img_path.write_bytes(img_bytes)
                    if str(img_path) not in extracted_images:
                        extracted_images.append(str(img_path))
                except Exception:
                    pass

        chunks = split_text(page_text)
        for chunk_index, chunk in enumerate(chunks, start=1):
            counter += 1
            nodes.append({
                "node_id": str(counter).zfill(4),
                "title": f"Page {page_index}",
                "path": f"{Path(pdf_path).name} > Page {page_index} > Chunk {chunk_index}",
                "source_file": Path(pdf_path).name,
                "page": page_index,
                "chunk": chunk_index,
                "text": chunk,
            })

    return nodes, scanned_pages, extracted_images


def extract_txt_nodes(file_path):
    file_path = Path(file_path)
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

        if in_code_block:
            continue

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
        end = nodes[i + 1]["line_num"] - 1 if i + 1 < len(nodes) else len(lines)
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
        title = node.get("title", "")
        path = f"{parent_path} > {title}" if parent_path else title
        text_chunks = split_text(node.get("text", ""))

        for chunk_index, chunk in enumerate(text_chunks, start=1):
            result.append({
                "node_id": node.get("node_id", ""),
                "title": title,
                "path": f"{source_file} > {path} > Chunk {chunk_index}",
                "source_file": source_file,
                "page": None,
                "chunk": chunk_index,
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
                "text": chunk,
            })

        children = node.get("nodes", [])
        if children:
            result.extend(flatten_tree(children, path, source_file))

    return result


def parse_markdown(file_path):
    file_path = Path(file_path)
    md_text = file_path.read_text(encoding="utf-8", errors="ignore")
    flat_nodes, lines = extract_markdown_nodes(md_text)

    if not flat_nodes:
        return extract_txt_nodes(file_path)

    flat_nodes = add_text_to_nodes(flat_nodes, lines)
    tree = build_tree(flat_nodes)
    return flatten_tree(nodes=tree, source_file=file_path.name)


def parse_document(file_path, uploads_dir: Path | None = None):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        if uploads_dir is None:
            uploads_dir = file_path.parent
        nodes, _, _ = extract_pdf_text_and_images(file_path, uploads_dir)
        return nodes

    if suffix in [".md", ".markdown"]:
        return parse_markdown(file_path)

    if suffix == ".txt":
        return extract_txt_nodes(file_path)

    return []
