from __future__ import annotations

from html import escape
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _display_html(html: str) -> str:
    try:
        from IPython.display import HTML, display
    except ImportError:
        print(html)
        return html

    display(HTML(html))
    return html


def _format_value(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"

    try:
        if np.isscalar(value) and np.isinf(value):
            return "∞"
    except TypeError:
        pass

    if isinstance(value, float):
        if not isfinite(value):
            return "∞"
        rounded = round(value, precision)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{precision}f}".rstrip("0").rstrip(".")

    if isinstance(value, np.generic):
        return _format_value(value.item(), precision=precision)

    return str(value)


def _style_block() -> str:
    return """
    <style>
      .algo-card {
        --ink: #172033;
        --muted: #667085;
        --line: #d9e2ec;
        --header: #0f766e;
        --header-ink: #ffffff;
        --body: #f8fafc;
        --body-alt: #eef6f4;
        display: inline-block;
        margin: 10px 0 18px;
        padding: 14px 16px 16px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.10);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
      }
      .algo-title {
        margin: 0 0 10px;
        font-size: 15px;
        font-weight: 750;
        letter-spacing: 0;
      }
      .algo-caption {
        margin: -4px 0 10px;
        color: var(--muted);
        font-size: 12px;
      }
      .algo-matrix {
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 7px;
        font-size: 13px;
      }
      .algo-matrix th,
      .algo-matrix td {
        min-width: 42px;
        height: 34px;
        padding: 0 12px;
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        text-align: center;
        vertical-align: middle;
        white-space: nowrap;
      }
      .algo-matrix th:last-child,
      .algo-matrix td:last-child {
        border-right: 0;
      }
      .algo-matrix tr:last-child th,
      .algo-matrix tr:last-child td {
        border-bottom: 0;
      }
      .algo-matrix thead th {
        background: var(--header);
        color: var(--header-ink);
        font-weight: 750;
      }
      .algo-matrix tbody th {
        background: #e0f2f1;
        color: #134e4a;
        font-weight: 750;
      }
      .algo-matrix tbody td {
        background: var(--body);
        font-variant-numeric: tabular-nums;
      }
      .algo-matrix tbody tr:nth-child(even) td {
        background: var(--body-alt);
      }
      .algo-inf {
        color: #94a3b8;
        font-weight: 700;
      }
      .algo-path {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 14px;
      }
      .algo-node {
        display: inline-flex;
        min-width: 30px;
        height: 30px;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #0f766e;
        color: white;
        font-weight: 750;
      }
      .algo-arrow {
        color: #64748b;
        font-weight: 700;
      }
    </style>
    """


def display_matrix(
    matrix: Sequence[Sequence[Any]],
    *,
    row_labels: Sequence[Any] | Mapping[int, Any] | None = None,
    col_labels: Sequence[Any] | Mapping[int, Any] | None = None,
    title: str = "Matrix",
    caption: str | None = None,
    precision: int = 2,
) -> str:
    """Render a matrix as a compact, report-friendly HTML table."""
    array = np.asarray(matrix, dtype=object)
    if array.ndim != 2:
        raise ValueError("display_matrix expects a 2D matrix")

    rows, cols = array.shape
    row_labels = _normalise_labels(row_labels, rows)
    col_labels = _normalise_labels(col_labels, cols)

    header_cells = ["<th></th>"] + [f"<th>{escape(str(label))}</th>" for label in col_labels]
    body_rows = []
    for row_idx in range(rows):
        cells = [f"<th>{escape(str(row_labels[row_idx]))}</th>"]
        for col_idx in range(cols):
            text = escape(_format_value(array[row_idx, col_idx], precision=precision))
            css_class = " class=\"algo-inf\"" if text == "∞" else ""
            cells.append(f"<td{css_class}>{text}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    caption_html = f"<div class=\"algo-caption\">{escape(caption)}</div>" if caption else ""
    html = f"""
    {_style_block()}
    <div class="algo-card">
      <div class="algo-title">{escape(title)}</div>
      {caption_html}
      <table class="algo-matrix">
        <thead><tr>{''.join(header_cells)}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """
    return _display_html(html)


def display_mapping(
    mapping: Mapping[Any, Any],
    *,
    title: str = "Values",
    key_name: str = "Node",
    value_name: str = "Value",
    precision: int = 2,
) -> str:
    rows = [[key, value] for key, value in mapping.items()]
    return display_matrix(
        rows,
        row_labels=[""] * len(rows),
        col_labels=[key_name, value_name],
        title=title,
        precision=precision,
    )


def display_path(
    path: Iterable[Any],
    *,
    title: str = "Shortest path",
    cost: Any | None = None,
    precision: int = 2,
) -> str:
    path = list(path)
    nodes = []
    for idx, node in enumerate(path):
        if idx:
            nodes.append("<span class=\"algo-arrow\">-></span>")
        nodes.append(f"<span class=\"algo-node\">{escape(str(node))}</span>")

    cost_html = ""
    if cost is not None:
        cost_html = f"<div class=\"algo-caption\">Cost: {_format_value(cost, precision=precision)}</div>"

    html = f"""
    {_style_block()}
    <div class="algo-card">
      <div class="algo-title">{escape(title)}</div>
      {cost_html}
      <div class="algo-path">{''.join(nodes)}</div>
    </div>
    """
    return _display_html(html)


def display_graph(
    graph: Mapping[Any, Sequence[tuple[Any, Any]]],
    *,
    title: str = "Graph",
    path: Sequence[Any] | None = None,
    directed: bool | None = None,
    figsize: tuple[float, float] = (7.0, 4.8),
    seed: int = 7,
) -> None:
    """Draw a weighted adjacency-list graph with optional path highlighting."""
    import matplotlib.pyplot as plt
    import networkx as nx

    directed = _infer_directed(graph) if directed is None else directed
    graph_cls = nx.DiGraph if directed else nx.Graph
    nx_graph = graph_cls()

    for node, neighbors in graph.items():
        nx_graph.add_node(node)
        for neighbor, weight in neighbors:
            nx_graph.add_edge(node, neighbor, weight=weight)

    path_edges = set()
    if path:
        path_edges = set(zip(path, path[1:]))
        if not directed:
            path_edges |= {(b, a) for a, b in path_edges}

    pos = nx.spring_layout(nx_graph, seed=seed, weight=None, k=1.1)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    node_colors = ["#ef4444" if path and node in path else "#0f766e" for node in nx_graph.nodes]
    edge_colors = ["#ef4444" if edge in path_edges else "#94a3b8" for edge in nx_graph.edges]
    edge_widths = [2.8 if edge in path_edges else 1.4 for edge in nx_graph.edges]

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_size=960,
        node_color=node_colors,
        edgecolors="#ffffff",
        linewidths=2.2,
        ax=ax,
    )
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        font_size=11,
        font_weight="bold",
        font_color="white",
        ax=ax,
    )
    edge_options = {
        "edge_color": edge_colors,
        "width": edge_widths,
        "ax": ax,
    }
    if directed:
        edge_options.update({"arrows": True, "arrowsize": 18, "connectionstyle": "arc3,rad=0.08"})
    nx.draw_networkx_edges(nx_graph, pos, **edge_options)
    edge_labels = {
        (source, target): _format_value(data.get("weight"))
        for source, target, data in nx_graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        nx_graph,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d9e2ec", "alpha": 0.96},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="#172033", pad=14)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def _normalise_labels(labels: Sequence[Any] | Mapping[int, Any] | None, size: int) -> list[Any]:
    if labels is None:
        return list(range(size))
    if isinstance(labels, Mapping):
        return [labels[index] for index in range(size)]
    return list(labels)


def _infer_directed(graph: Mapping[Any, Sequence[tuple[Any, Any]]]) -> bool:
    weights = {(node, neighbor): weight for node, neighbors in graph.items() for neighbor, weight in neighbors}
    for (node, neighbor), weight in weights.items():
        if weights.get((neighbor, node)) != weight:
            return True
    return False
