"""Text-to-SQL agent CLI.

  python main.py ask "your question"    # run one question
  python main.py                        # interactive mode
  streamlit run app.py                  # chat web UI
"""
import argparse

from text2sql.graph import build_graph

_TRACE = {
    "select_modules": lambda u: f"modules: {u.get('modules')}  ({u.get('module_reasoning')})",
    "select_tables": lambda u: f"tables: {u.get('tables')}  ({u.get('table_reasoning')})",
    "resolve_joins": lambda u: f"tables incl. bridges: {u.get('tables')}\n  joins: {u.get('joins')}",
    "generate_sql": lambda u: f"attempt {u.get('attempts')}:\n{u.get('sql')}",
    "validate_sql": lambda u: f"error: {u.get('error')}" if u.get("error") else "ok",
    "execute_sql": lambda u: (f"error: {u.get('error')}" if u.get("error")
                              else f"{len(u.get('rows', []))} rows"),
}


def ask(app, question: str, verbose: bool = True) -> dict:
    final: dict = {"question": question}
    for update in app.stream({"question": question}, stream_mode="updates"):
        for node, values in update.items():
            final.update(values or {})
            if verbose and node in _TRACE:
                print(f"[{node}] {_TRACE[node](values or {})}")
    print("\n--- SQL ---\n" + (final.get("sql") or "(none)"))
    print("\n--- Answer ---\n" + final.get("answer", ""))
    return final


def main():
    parser = argparse.ArgumentParser(description="LangGraph Text-to-SQL agent")
    sub = parser.add_subparsers(dest="cmd")
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("-q", "--quiet", action="store_true", help="Hide intermediate steps")
    args = parser.parse_args()

    app = build_graph()
    if args.cmd == "ask":
        ask(app, args.question, verbose=not args.quiet)
        return

    while True:
        try:
            question = input("\nQuestion (blank to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(app, question)


if __name__ == "__main__":
    main()
