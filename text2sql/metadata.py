"""Loads the JSON metadata files.

  * METADATA_DIR: Modules_high_lvl.json / Tables_high_lvl.json -> high-level catalogs
  * TABLE_METADATA_DIR: one object with "table_name" and "columns" per file
                        -> full table metadata (dict keyed by table name)
  * JOIN_DETAILS_DIR:   one object with "join_condition" per file -> join definition
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Join:
    table1: str
    alias1: str
    table2: str
    alias2: str
    condition: str
    relationship: str

    def describe(self) -> str:
        return (f"{self.table1} {self.alias1} JOIN {self.table2} {self.alias2} "
                f"ON {self.condition}  -- {self.relationship}")


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_table_ref(ref: str) -> tuple[str, str]:
    """'tb0005_calendriers tb5' -> ('tb0005_calendriers', 'tb5')"""
    parts = ref.split()
    return parts[0], (parts[1] if len(parts) > 1 else parts[0])


@lru_cache
def load_modules() -> list[dict]:
    return _read(config.MODULES_FILE)


@lru_cache
def load_tables_high_level() -> list[dict]:
    return _read(config.TABLES_FILE)


def _json_files(folder: Path, required: tuple[str, ...]):
    """Yields (path, data) for every JSON object in `folder` that has all `required` keys."""
    if not folder.is_dir():
        print(f"[metadata] folder not found: {folder}")
        return
    for path in sorted(folder.glob("*.json")):
        try:
            data = _read(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[metadata] skipping {path.name}: {exc}")
            continue
        if not isinstance(data, dict) or not all(k in data for k in required):
            print(f"[metadata] skipping {path.name}: missing one of {required}")
            continue
        yield path, data


@lru_cache
def _scan() -> tuple[dict[str, dict], tuple[Join, ...]]:
    tables: dict[str, dict] = {}
    for _, data in _json_files(config.TABLE_METADATA_DIR, ("table_name", "columns")):
        for col in data["columns"]:
            col["name"] = col["name"].strip()
        tables[data["table_name"].strip()] = data

    joins: list[Join] = []
    for _, data in _json_files(config.JOIN_DETAILS_DIR, ("table1", "table2", "join_condition")):
        t1, a1 = _split_table_ref(data["table1"])
        t2, a2 = _split_table_ref(data["table2"])
        joins.append(Join(t1, a1, t2, a2, data["join_condition"].strip(),
                          data.get("relationship", "")))
    return tables, tuple(joins)


def load_table_details() -> dict[str, dict]:
    return _scan()[0]


def load_joins() -> tuple[Join, ...]:
    return _scan()[1]


def tables_for_modules(modules: list[str]) -> list[dict]:
    wanted = {m.lower() for m in modules}
    return [t for t in load_tables_high_level() if t["module"].lower() in wanted]
