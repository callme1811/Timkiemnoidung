from pathlib import Path

APP_TITLE = "DocAnalyzer AI"
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

SUPPORTED_DOCUMENT_TYPES = ["pdf", "md", "markdown", "txt"]
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
SUPPORTED_UPLOAD_TYPES = SUPPORTED_DOCUMENT_TYPES + SUPPORTED_IMAGE_TYPES

DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
