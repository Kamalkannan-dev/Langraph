"""Chat UI for the Text-to-SQL agent.

  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from text2sql.graph import build_graph

_STEPS = {
    "select_modules": "Selecting modules",
    "select_tables": "Selecting tables",
    "resolve_joins": "Resolving joins",
    "load_metadata": "Loading table metadata",
    "generate_sql": "Generating SQL",
    "validate_sql": "Validating SQL",
    "execute_sql": "Executing SQL",
    "generate_answer": "Writing answer",
}


def _step_detail(node: str, u: dict) -> str | None:
    if node == "select_modules":
        return f"**Modules:** {', '.join(u.get('modules') or []) or '(none)'}\n\n{u.get('module_reasoning', '')}"
    if node == "select_tables":
        return f"**Tables:** {', '.join(u.get('tables') or []) or '(none)'}\n\n{u.get('table_reasoning', '')}"
    if node == "resolve_joins":
        joins = "\n".join(f"- {j}" for j in u.get("joins") or []) or "_No joins needed_"
        return f"**Tables incl. bridges:** {', '.join(u.get('tables') or [])}\n\n**Joins:**\n{joins}"
    if node == "generate_sql":
        return f"**Attempt {u.get('attempts')}:** {u.get('sql_reasoning', '')}\n\n```sql\n{u.get('sql', '')}\n```"
    if node in ("validate_sql", "execute_sql"):
        if u.get("error"):
            return f":red[**Error:** {u['error']}]"
        if node == "execute_sql":
            return f"{len(u.get('rows') or [])} rows returned"
        return "SQL is valid"
    return None


@st.cache_resource
def get_app():
    return build_graph()


def render_result(msg: dict) -> None:
    if msg.get("steps"):
        with st.expander("Reasoning", expanded=False):
            for title, detail in msg["steps"]:
                st.markdown(f"**{title}**")
                if detail:
                    st.markdown(detail)
                st.divider()
    if msg.get("sql"):
        with st.expander("SQL", expanded=True):
            st.code(msg["sql"], language="sql")
    if msg.get("rows"):
        with st.expander(f"Result ({len(msg['rows'])} rows{', truncated' if msg.get('truncated') else ''})"):
            st.dataframe(pd.DataFrame(msg["rows"], columns=msg.get("columns")), width="stretch")
    if msg.get("error_msg"):
        st.error(msg["error_msg"])
    st.markdown(msg.get("content", ""))


def run_question(question: str) -> dict:
    final: dict = {}
    steps: list[tuple[str, str | None]] = []
    with st.status("Thinking...", expanded=True) as status:
        try:
            for update in get_app().stream({"question": question}, stream_mode="updates"):
                for node, values in update.items():
                    values = values or {}
                    final.update(values)
                    if node in _STEPS:
                        detail = _step_detail(node, values)
                        steps.append((_STEPS[node], detail))
                        status.update(label=f"{_STEPS[node]}...")
                        st.markdown(f"✅ {_STEPS[node]}")
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error", expanded=False)
            return {"role": "assistant", "content": "", "steps": steps, "sql": final.get("sql"),
                    "error_msg": f"{type(exc).__name__}: {exc}"}
    return {
        "role": "assistant",
        "content": final.get("answer", ""),
        "steps": steps,
        "sql": final.get("sql"),
        "columns": final.get("columns"),
        "rows": final.get("rows"),
        "truncated": final.get("truncated"),
    }


def main() -> None:
    st.set_page_config(page_title="SM2 Text-to-SQL", page_icon="💬", layout="wide")
    st.title("💬 SM2 Text-to-SQL Assistant")

    with st.sidebar:
        st.header("Options")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(msg)

    if question := st.chat_input("Ask a question about the SM2 database..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            reply = run_question(question)
            render_result(reply)
        st.session_state.messages.append(reply)


main()
