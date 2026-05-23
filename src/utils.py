import hashlib
from pathlib import Path


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def safe_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_")


def save_uploaded_file(uploaded_file, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    save_path = uploads_dir / f"{file_hash[:10]}_{safe_filename(uploaded_file.name)}"

    if not save_path.exists():
        save_path.write_bytes(file_bytes)

    return save_path


def get_mime_type(path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"
