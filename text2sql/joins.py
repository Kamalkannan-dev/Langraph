"""Static join resolution from the join JSON files."""
from collections import deque

from .metadata import Join, load_joins


def _adjacency() -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for j in load_joins():
        adj.setdefault(j.table1, []).append(j.table2)
        adj.setdefault(j.table2, []).append(j.table1)
    return adj


def _shortest_path(adj, sources: set[str], target: str) -> list[str] | None:
    """BFS from any table in `sources` to `target`; returns the tables on the path."""
    queue = deque([(s, [s]) for s in sources])
    seen = set(sources)
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def resolve_joins(tables: list[str]) -> tuple[list[str], list[Join]]:
    """Returns (tables incl. bridge tables needed to connect them, joins between those tables)."""
    adj = _adjacency()
    result = list(dict.fromkeys(tables))
    if result:
        connected = {result[0]}
        for table in list(result[1:]):
            path = _shortest_path(adj, connected, table)
            if path:
                for t in path:
                    if t not in result:
                        result.append(t)  # bridge table
                connected.update(path)
            else:
                connected.add(table)

    selected = set(result)
    joins = [j for j in load_joins() if j.table1 in selected and j.table2 in selected]
    return result, joins


def build_from_clause(tables: list[str], joins: list[Join]) -> str | None:
    """Builds a correctly ordered 'FROM ... JOIN ... ON ...' clause spanning the joined tables."""
    if not joins:
        return None
    joined_tables = [t for t in tables if any(t in (j.table1, j.table2) for j in joins)]
    alias = {}
    for j in joins:
        alias.setdefault(j.table1, j.alias1)
        alias.setdefault(j.table2, j.alias2)
    start = joined_tables[0]
    included, pending = {start}, list(joins)
    lines = [f"FROM {start} {alias[start]}"]
    while pending:
        for j in pending:
            if (j.table1 in included) != (j.table2 in included):
                new = j.table2 if j.table1 in included else j.table1
                lines.append(f"JOIN {new} {alias[new]} ON {j.condition}")
                included.add(new)
                pending.remove(j)
                break
            if j.table1 in included and j.table2 in included:
                lines[-1] += f" AND {j.condition}"  # extra edge closing a cycle
                pending.remove(j)
                break
        else:
            # Disconnected component: start a new chain from it
            j = pending[0]
            lines.append(f"CROSS JOIN {j.table1} {alias[j.table1]}  -- no known join path")
            included.add(j.table1)
    return "\n".join(lines)
