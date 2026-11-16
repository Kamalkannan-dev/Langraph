"""LangGraph Text-to-SQL workflow.

question -> select_modules -> select_tables -> resolve_joins -> load_metadata
         -> generate_sql -> validate_sql -> execute_sql -> generate_answer
(validation / execution errors loop back to generate_sql up to MAX_SQL_ATTEMPTS)
"""
import json
from functools import lru_cache
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from . import config, db
from .joins import build_from_clause, resolve_joins as _resolve_joins
from .metadata import load_joins, load_modules, load_table_details, tables_for_modules


class AgentState(TypedDict, total=False):
    question: str
    modules: list[str]
    module_reasoning: str
    tables: list[str]
    table_reasoning: str
    joins: list[str]
    from_clause: Optional[str]
    schema_context: str
    sql: str
    sql_reasoning: str
    error: Optional[str]
    attempts: int
    columns: list[str]
    rows: list[list]
    truncated: bool
    answer: str


@lru_cache
def _llm() -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)


def _structured(schema: type[BaseModel], system: str, human: str) -> BaseModel:
    return _llm().with_structured_output(schema).invoke([SystemMessage(system), HumanMessage(human)])


# ---------------------------------------------------------------- Step 1: modules

def select_modules(state: AgentState) -> AgentState:
    modules = load_modules()
    names = tuple(m["module"] for m in modules)
    schema = create_model(
        "ModuleSelection",
        reasoning=(str, Field(description="One or two sentences explaining the choice")),
        modules=(list[Literal[names]], Field(  # type: ignore[valid-type]
            description="Modules needed to answer the question; empty list if none is relevant")),
    )
    catalog = "\n".join(f"- {m['module']}: {m['description']}" for m in modules)
    system = (
        "You route a user's database question to the business modules whose tables are needed to "
        "answer it. Modules are interrelated: select every module whose data is needed (for "
        "filters, joins or output), but do not select unrelated modules. If the question is not "
        "about any module, return an empty list.\n\nMODULES:\n" + catalog
    )
    result = _structured(schema, system, state["question"])
    return {"modules": list(dict.fromkeys(result.modules)), "module_reasoning": result.reasoning}


def route_after_modules(state: AgentState) -> str:
    return "select_tables" if state.get("modules") else "no_match"


def no_match(state: AgentState) -> AgentState:
    return {"answer": "The question does not match any known module, so no SQL was generated. "
                      "Available modules: " + ", ".join(m["module"] for m in load_modules())}


# ---------------------------------------------------------------- Step 2: tables

def _detail_candidates(modules: list[str]) -> list[dict]:
    """Fallback catalog built from the detailed table metadata when the high-level one is empty."""
    wanted = {m.lower() for m in modules}
    return [t for t in load_table_details().values() if t.get("module", "").lower() in wanted]


def select_tables(state: AgentState) -> AgentState:
    candidates = tables_for_modules(state["modules"]) or _detail_candidates(state["modules"])
    if not candidates:
        return {"tables": [], "table_reasoning": "No tables are catalogued for the selected modules."}

    names = tuple(t["table_name"] for t in candidates)
    schema = create_model(
        "TableSelection",
        reasoning=(str, Field(description="For each requested piece of information and each filter, "
                                          "name the table that stores it")),
        tables=(list[Literal[names]], Field(  # type: ignore[valid-type]
            description="All tables needed for SELECT columns, filters, aggregations and joins")),
    )
    catalog = "\n".join(
        f"- {t['table_name']} (module: {t.get('module', '')}): {t.get('description', '')} "
        f"Business concepts: {', '.join(t.get('business_concepts', []))}"
        for t in candidates
    )
    system = (
        "You select the database tables needed to write a PostgreSQL query answering the user's "
        "question. First list every piece of information the question asks for or filters on "
        "(e.g. sector, year, shift type, active status), then pick the table that directly stores "
        "each one according to its description and business concepts. Never plan to derive a "
        "value that another table stores directly. Intermediate join tables are added "
        "automatically, so do not add tables only for joining.\n\nTABLES:\n" + catalog
    )
    result = _structured(schema, system, state["question"])
    return {"tables": list(dict.fromkeys(result.tables)), "table_reasoning": result.reasoning}


def route_after_tables(state: AgentState) -> str:
    return "resolve_joins" if state.get("tables") else "no_tables"


def no_tables(state: AgentState) -> AgentState:
    return {"answer": "No relevant tables were found for the question, so no SQL was generated. "
                      f"Selected modules: {', '.join(state.get('modules', []))}"}


# ---------------------------------------------------------------- Step 3: joins

def resolve_joins(state: AgentState) -> AgentState:
    tables, joins = _resolve_joins(state["tables"])
    return {"tables": tables, "joins": [j.describe() for j in joins],
            "from_clause": build_from_clause(tables, joins)}


# ---------------------------------------------------------------- Step 4: full metadata

def _render_table(table: dict, alias: str | None) -> str:
    head = f"TABLE {table['table_name']}" + (f" (alias {alias})" if alias else "")
    lines = [f"{head}: {table.get('description', '')}"]
    lines += [f"  - {c['name']} {c['type']}: {c.get('description', '')}" for c in table["columns"]]
    return "\n".join(lines)


def load_metadata(state: AgentState) -> AgentState:
    tables = state["tables"]
    details = load_table_details()  # cached in-memory dict: table name -> metadata
    found = {t: details[t] for t in tables if t in details}

    aliases: dict[str, str] = {}
    for j in load_joins():
        aliases.setdefault(j.table1, j.alias1)
        aliases.setdefault(j.table2, j.alias2)

    parts = [_render_table(found[t], aliases.get(t)) for t in tables if t in found]
    unknown = [t for t in tables if t not in found]
    if unknown:
        parts.append("Tables without column metadata (avoid if possible): " + ", ".join(unknown))
    if state.get("joins"):
        parts.append("JOIN RELATIONSHIPS (use exactly these join conditions):\n"
                     + "\n".join(f"  - {j}" for j in state["joins"]))
    if state.get("from_clause"):
        parts.append("RECOMMENDED FROM CLAUSE (correct join order; drop trailing JOINs to tables "
                     "you don't need, but keep any table that connects needed tables):\n"
                     + state["from_clause"])
    return {"schema_context": "\n\n".join(parts), "attempts": 0, "error": None}


# ---------------------------------------------------------------- Step 5: SQL generation

class SQLQuery(BaseModel):
    reasoning: str = Field(description="Brief plan: tables, joins, filters and aggregations to use")
    sql: str = Field(description="A single PostgreSQL SELECT query")


def generate_sql(state: AgentState) -> AgentState:
    system = (
        "You are an expert PostgreSQL developer. Write ONE read-only SELECT query that answers the "
        "user's question using ONLY the tables, columns and join conditions in the schema below.\n"
        "Rules:\n"
        "- Never invent tables or columns.\n"
        "- Use the given table aliases and join conditions exactly. If a RECOMMENDED FROM CLAUSE "
        "is given, build the query on it; every alias must be joined before it is referenced.\n"
        "- Qualify every column with its table alias.\n"
        "- Use the code values documented in column descriptions (e.g. 1 = Active).\n"
        f"- Unless the user asks for everything or an aggregate, add LIMIT {config.MAX_ROWS}.\n"
        "- Do not end the query with a semicolon.\n\n"
        "SCHEMA:\n" + state["schema_context"]
    )
    human = f"Question: {state['question']}"
    if state.get("error"):
        human += (f"\n\nYour previous query failed.\nPrevious SQL:\n{state.get('sql', '')}\n"
                  f"Error: {state['error']}\nFix the query.")
    result = _structured(SQLQuery, system, human)
    return {"sql": result.sql.strip().rstrip(";").strip(), "sql_reasoning": result.reasoning,
            "attempts": state.get("attempts", 0) + 1, "error": None}


def validate_sql(state: AgentState) -> AgentState:
    return {"error": db.validate_sql(state["sql"])}


def _retry_or(state: AgentState, nxt: str) -> str:
    if state.get("error"):
        return "generate_sql" if state["attempts"] < config.MAX_SQL_ATTEMPTS else "failed"
    return nxt


def route_after_validate(state: AgentState) -> str:
    return _retry_or(state, "execute_sql" if config.DATABASE_URL else "sql_only")


def sql_only(state: AgentState) -> AgentState:
    return {"answer": "SQL generated (not executed because DATABASE_URL is not set)."}


def failed(state: AgentState) -> AgentState:
    return {"answer": f"Could not produce a valid query after {state['attempts']} attempts. "
                      f"Last error: {state['error']}"}


# ---------------------------------------------------------------- Step 6: execution + answer

def execute_sql(state: AgentState) -> AgentState:
    try:
        columns, rows, truncated = db.execute(state["sql"])
    except Exception as exc:  # DB errors are fed back to the LLM
        return {"error": f"{type(exc).__name__}: {str(exc).strip()}"}
    return {"columns": columns, "rows": rows, "truncated": truncated, "error": None}


def route_after_execute(state: AgentState) -> str:
    return _retry_or(state, "generate_answer")


def generate_answer(state: AgentState) -> AgentState:
    rows = state.get("rows", [])
    if not rows:
        return {"answer": "The query ran successfully but returned no rows."}
    preview = [dict(zip(state["columns"], r)) for r in rows[:50]]
    system = (
        "You answer the user's question using ONLY the SQL query result provided. Be concise and "
        "factual; present lists or tables in markdown where helpful. Do not invent data."
    )
    human = (
        f"Question: {state['question']}\nSQL:\n{state['sql']}\n\n"
        f"Result ({len(rows)} rows{', truncated' if state.get('truncated') else ''}"
        f"{', showing first 50' if len(rows) > 50 else ''}):\n"
        f"{json.dumps(preview, ensure_ascii=False, default=str)}"
    )
    return {"answer": _llm().invoke([SystemMessage(system), HumanMessage(human)]).content}


# ---------------------------------------------------------------- Graph

def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [
        ("select_modules", select_modules), ("no_match", no_match),
        ("select_tables", select_tables), ("no_tables", no_tables), ("resolve_joins", resolve_joins),
        ("load_metadata", load_metadata), ("generate_sql", generate_sql),
        ("validate_sql", validate_sql), ("execute_sql", execute_sql),
        ("generate_answer", generate_answer), ("sql_only", sql_only), ("failed", failed),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "select_modules")
    g.add_conditional_edges("select_modules", route_after_modules, ["select_tables", "no_match"])
    g.add_conditional_edges("select_tables", route_after_tables, ["resolve_joins", "no_tables"])
    g.add_edge("resolve_joins", "load_metadata")
    g.add_edge("load_metadata", "generate_sql")
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges("validate_sql", route_after_validate,
                            ["generate_sql", "execute_sql", "sql_only", "failed"])
    g.add_conditional_edges("execute_sql", route_after_execute,
                            ["generate_sql", "generate_answer", "failed"])
    for terminal in ("no_match", "no_tables", "generate_answer", "sql_only", "failed"):
        g.add_edge(terminal, END)
    return g.compile()
