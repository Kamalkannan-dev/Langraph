"""Read-only PostgreSQL execution."""
import re
from decimal import Decimal

import psycopg

from . import config

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy|call)\b",
    re.IGNORECASE,
)


def _strip_comments_and_literals(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql: str) -> str | None:
    """Returns an error message, or None if the SQL is a single read-only query."""
    if not sql or not sql.strip():
        return "Empty SQL."
    body = _strip_comments_and_literals(sql).strip().rstrip(";").strip()
    if ";" in body:
        return "Only a single SQL statement is allowed."
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        return "Only SELECT (or WITH ... SELECT) queries are allowed."
    if m := _FORBIDDEN.search(body):
        return f"Forbidden keyword in query: {m.group(1).upper()}"
    return None


def _jsonable(v):
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def execute(sql: str) -> tuple[list[str], list[list], bool]:
    """Runs the query in a read-only transaction. Returns (columns, rows, truncated)."""
    with psycopg.connect(config.DATABASE_URL, autocommit=False, connect_timeout=10) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {int(config.STATEMENT_TIMEOUT_MS)}")
            if config.DB_SCHEMA:
                cur.execute("SELECT set_config('search_path', %s, true)", (config.DB_SCHEMA,))
            cur.execute(sql.strip().rstrip(";"))
            columns = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchmany(config.MAX_ROWS + 1)
        conn.rollback()
    truncated = len(rows) > config.MAX_ROWS
    return columns, [[_jsonable(v) for v in r] for r in rows[: config.MAX_ROWS]], truncated
