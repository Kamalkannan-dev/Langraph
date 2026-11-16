import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Keep local services (Ollama, DB) off the corporate HTTP(S)_PROXY
_no_proxy = {h for h in os.getenv("NO_PROXY", os.getenv("no_proxy", "")).split(",") if h}
_no_proxy.update({"localhost", "127.0.0.1", "::1"})
os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(_no_proxy))

# Folder holding the high-level catalogs Modules_high_lvl.json and Tables_high_lvl.json
METADATA_DIR = Path(os.getenv("METADATA_DIR", PROJECT_DIR))
MODULES_FILE = METADATA_DIR / "Modules_high_lvl.json"
TABLES_FILE = METADATA_DIR / "Tables_high_lvl.json"
# One JSON file per table (table_name, columns, ...)
TABLE_METADATA_DIR = Path(os.getenv("TABLE_METADATA_DIR", METADATA_DIR / "table_metadata"))
# One JSON file per join (table1, table2, join_condition, relationship)
JOIN_DETAILS_DIR = Path(os.getenv("JOIN_DETAILS_DIR", METADATA_DIR / "join_details"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:latest")

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_DIR / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "table_metadata")

# e.g. postgresql://user:password@host:5432/dbname ; leave unset to only generate SQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sm2_01_adm:093b5a4be52251c2@localhost:5433/sm2_01_db",
)
DB_SCHEMA = os.getenv("DB_SCHEMA","sm2")  # optional, e.g. "sm2"
MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "30000"))
MAX_SQL_ATTEMPTS = int(os.getenv("MAX_SQL_ATTEMPTS", "3"))
