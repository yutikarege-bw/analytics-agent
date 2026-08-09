"""
Text-to-visualization agent (Google ADK 2.0 + Gemini), refactored for Streamlit.

Differences vs. the original notebook:
  * No `display()` / `fig.show()` inside tools. Tools push artifacts onto a per-run
    collector that Streamlit renders afterwards.
  * Tool payloads are flat, typed arguments instead of hand-rolled JSON strings, so
    Gemini gets a real function schema and can't drift on format.
  * The DataFrame never round-trips through the model. SQL results stay in Python;
    the chart tool reads the last result by reference.
  * ADK 2.0: tools raise instead of swallowing exceptions, so the framework's retry
    machinery (ReflectAndRetryToolPlugin) can actually see failures.
"""

from __future__ import annotations

import csv
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from google import genai
from google.genai import types as genai_types

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

APP_NAME = "viz_app"

# `gemini-flash-latest` is an alias that tracks the current Flash model. Pin an explicit
# version (e.g. "gemini-3-flash-preview") if you need reproducible behaviour.
DEFAULT_AGENT_MODEL = os.getenv("VIZ_AGENT_MODEL", "gemini-flash-latest")
DEFAULT_REFINE_MODEL = os.getenv("VIZ_REFINE_MODEL", "gemini-flash-latest")

MAX_RESULT_ROWS = 5000  # hard cap on rows pulled out of DuckDB
MAX_ROWS_TO_MODEL = 50  # rows echoed back to Gemini (the chart reads the full frame)

REFINE_SYSTEM_INSTRUCTION = """You are an assistant that reformulates user queries into structured prompts for data analysis and visualization.
Always follow these steps:
1. Identify the key data points, dimensions, metrics, filters, or chart types mentioned in the user's query.
2. Reformulate the user's input into a clear, concise, and structured prompt that can be accurately interpreted by an LLM.
3. Ensure that all data points, relationships, and visualization intents from the original request are preserved in the final reformulated prompt and will be included in the visualization.
4. If any part of the user's request is ambiguous or vague, make reasonable assumptions but preserve intent.
5. When a TOTAL value is mentioned, you MUST include it in the visualization. You MUST NOT include a total when it is not mentioned. For example: "how the **total** is split between..." means the total must be present in the visualization.
Return only the reformulated prompt, with no preamble."""

AGENT_INSTRUCTION = """You turn questions about uploaded spreadsheets into a single Plotly chart.

Follow these steps in order. Do not skip any.

1. Call `preview_data_sources` first. It returns every available table with its exact
   SQL table name, real column names and dtypes. Match the user's wording to actual
   table and column names, and state the mapping you chose in one short line.

2. Call `run_sql_query` with a DuckDB SQL query.
   - Use the exact `table` names from the preview in your FROM clause. Several files,
     and several sheets of one workbook, can be loaded at once - join across them when
     the question needs it.
   - The first table is also aliased as `data`.
   - Quote column names containing spaces with double quotes: SELECT "Product Area" FROM sales
   - Produce LONG format, one row per (category, series, value). If the source is wide,
     unpivot it in SQL with UNION ALL or UNPIVOT rather than reshaping it yourself.
   - Filter out NULL/empty category values when building a stacked or 100% stacked bar.
   - Aggregate in SQL (GROUP BY) so the result is chart-sized, not raw rows.

3. Call `create_visualization`. It reads the result of the last `run_sql_query`
   automatically, so you only pass column names and chart options - never the data.
   - `x`, `y`, and `color` must be column names present in the SQL result.
   - Use the chart type the user asked for. For a Sankey, pass `source`, `target`
     and `value` instead of `y`. For a heatmap, pass the numeric value column as
     `color`. Use `100%_single_stacked_bar` when the user wants one bar showing how
     a single total splits across categories.
   - Supported `plot_type` values: line, bar, stacked_bar, 100%_stacked_bar,
     100%_single_stacked_bar, scatter, box, histogram, pie, heatmap, sankey.

4. When the chart tool reports success, reply with two or three sentences describing
   what the chart shows and the mapping you used. Then stop. Do not call more tools.

Limits: call any single tool at most 3 times per request. If a tool keeps failing,
stop and explain the problem plainly instead of guessing at new table or column names."""


# --------------------------------------------------------------------------------------
# Per-run artifact collection
#
# Streamlit renders charts, not the tools. Tools append to the RunState that the UI
# installed before calling the runner. A ContextVar keeps this safe across asyncio
# tasks; the module-level fallback covers ADK executing a sync tool in a worker thread.
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    """One queryable table: a CSV file, or one sheet of a workbook."""

    table: str  # SQL-safe name the agent writes in FROM clauses
    df: pd.DataFrame
    source: str  # original file name
    sheet: str = ""  # sheet name, for Excel


@dataclass
class RunState:
    """Everything a single agent run produced, for the UI to render."""

    datasets: list[Dataset] = field(default_factory=list)
    preview: Optional[pd.DataFrame] = None
    queries: list[str] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list[go.Figure] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    chart_height: int = 520

    @property
    def last_table(self) -> Optional[pd.DataFrame]:
        return self.tables[-1] if self.tables else None

    @property
    def table_names(self) -> list[str]:
        return [d.table for d in self.datasets]

    def note(self, message: str) -> None:
        self.log.append(message)

    def add(self, df: pd.DataFrame, source: str, sheet: str = "") -> Dataset:
        """Register a DataFrame under a unique SQL-safe table name."""
        base = sanitize_table_name(sheet or os.path.splitext(source)[0])
        if sheet and any(d.table == base for d in self.datasets):
            base = sanitize_table_name(f"{os.path.splitext(source)[0]}_{sheet}")
        name, n = base, 2
        while any(d.table == name for d in self.datasets):
            name, n = f"{base}_{n}", n + 1
        dataset = Dataset(table=name, df=df, source=source, sheet=sheet)
        self.datasets.append(dataset)
        return dataset


_current_run: ContextVar[Optional[RunState]] = ContextVar("adk_viz_run", default=None)
_fallback_run: Optional[RunState] = None


def set_current_run(run: RunState) -> None:
    """Install the RunState the tools should write into. Call before runner.run_async."""
    global _fallback_run
    _fallback_run = run
    _current_run.set(run)


def get_current_run() -> RunState:
    run = _current_run.get() or _fallback_run
    if run is None:
        raise RuntimeError("No active run: call set_current_run() before running the agent.")
    return run


# --------------------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------------------


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise blank-ish values to None so DuckDB and Plotly behave predictably."""
    df = df.replace(r"^\s*$", None, regex=True)
    df = df.replace(["nan", "NaN", "null", "NULL", "None"], None)
    df = df.where(pd.notnull(df), None)
    return df


def sanitize_table_name(raw: str) -> str:
    """Turn a file or sheet name into something safe to type in a FROM clause."""
    name = re.sub(r"\W+", "_", str(raw).strip().lower()).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name[:48]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column names, drop fully-empty columns and rows, normalise blanks."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Excel exports routinely carry trailing unnamed columns and blank rows.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$", na=False) | df.notna().any()]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    seen: dict[str, int] = {}
    columns = []
    for col in df.columns:  # DuckDB rejects duplicate column names
        if col in seen:
            seen[col] += 1
            columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            columns.append(col)
    df.columns = columns
    return preprocess_dataframe(df.reset_index(drop=True))


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a delimited file without assuming UTF-8 or commas.

    Real-world exports are semicolon-separated (European Excel), Latin-1 encoded, or
    have a BOM. Sniff the delimiter, then fall back to pandas' own inference.
    """
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as handle:
                sample = handle.read(64 * 1024)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue

        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","

        try:
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            last_error = exc
            try:  # engine="python" with sep=None infers the delimiter itself
                return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            except Exception as exc2:  # noqa: BLE001 - retried under the next encoding
                last_error = exc2
                continue

    raise ValueError(f"Could not parse '{os.path.basename(path)}' as delimited text: {last_error}")


def load_datasets(path: str, display_name: str) -> list[tuple[pd.DataFrame, str]]:
    """Read one uploaded file into [(dataframe, sheet_name)].

    A CSV yields one frame. A workbook yields one frame per non-empty sheet, so the
    agent can query - and join - every sheet without the user picking one up front.
    """
    lower = display_name.lower()

    if lower.endswith(".tsv"):
        return [(_tidy(pd.read_csv(path, sep="\t")), "")]
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return [(_tidy(read_csv_robust(path)), "")]

    book = pd.ExcelFile(path)
    out: list[tuple[pd.DataFrame, str]] = []
    for sheet in book.sheet_names:
        frame = _tidy(book.parse(sheet))
        if not frame.empty and len(frame.columns):
            out.append((frame, str(sheet)))
    if not out:
        raise ValueError(f"'{display_name}' has no sheet with usable data.")
    return out


def _jsonable_rows(df: pd.DataFrame, limit: int) -> list[dict]:
    """Rows safe to hand back to the model: no NaN, no inf, no numpy scalars."""
    out = df.head(limit).copy()
    out = out.replace([float("inf"), float("-inf")], None)
    out = out.where(pd.notna(out), None)
    records = out.to_dict(orient="records")
    for row in records:
        for key, val in row.items():
            if hasattr(val, "item"):
                row[key] = val.item()
            elif isinstance(val, pd.Timestamp):
                row[key] = val.isoformat()
    return records


# --------------------------------------------------------------------------------------
# Tools
#
# ADK 2.0 note: these deliberately do NOT wrap their bodies in `except Exception`.
# A raised exception is visible to the framework, which is what lets
# ReflectAndRetryToolPlugin feed the error back to the model for a corrected retry.
# --------------------------------------------------------------------------------------


def preview_data_sources(tool_context: ToolContext) -> dict:
    """Inspect the uploaded data before writing any SQL.

    Lists every available table with its exact SQL table name, column names, dtypes
    and a few sample rows. Always call this first: the user's wording rarely matches
    the real column names, and you need the exact table names for your FROM clause.

    Returns:
        dict: {"tables": [{"table": str, "source": str, "sheet": str, "row_count": int,
               "columns": [...], "dtypes": {...}, "sample_rows": [...]}]}
    """
    run = get_current_run()
    if not run.datasets:
        raise ValueError("No data is loaded. Ask the user to upload a CSV or Excel file.")

    tables = []
    for dataset in run.datasets:
        sample = dataset.df.head(5).astype(str)
        tables.append(
            {
                "table": dataset.table,
                "source": dataset.source,
                "sheet": dataset.sheet,
                "row_count": int(len(dataset.df)),
                "columns": [str(c) for c in dataset.df.columns],
                "dtypes": {str(k): str(v) for k, v in dataset.df.dtypes.items()},
                "sample_rows": _jsonable_rows(sample, 5),
            }
        )

    run.preview = run.datasets[0].df.head(5).astype(str)
    run.note(
        "Previewed "
        + ", ".join(f"`{d.table}` ({len(d.df)}x{len(d.df.columns)})" for d in run.datasets)
    )

    tool_context.state["tables"] = {t["table"]: t["columns"] for t in tables}
    return {
        "tables": tables,
        "note": (
            "Query these using their `table` name. The first table is also aliased as `data`."
            if len(tables) > 1
            else "Query this table by its `table` name, or as `data`."
        ),
    }


def run_sql_query(query: str, tool_context: ToolContext) -> dict:
    """Run a DuckDB SQL query against the uploaded data.

    Every table listed by `preview_data_sources` is registered under its `table` name,
    and the first one is also aliased as `data`. You can join across them freely.
    Aggregate here (GROUP BY) and return long-format rows: one row per
    (category, series, value). The full result stays in Python for charting;
    only a preview is returned to you.

    Args:
        query: A DuckDB SQL SELECT statement.

    Returns:
        dict: {"columns": [...], "row_count": int, "rows_preview": [...], "truncated": bool}
    """
    run = get_current_run()
    if not run.datasets:
        raise ValueError("No data is loaded. Ask the user to upload a CSV or Excel file.")
    if not query or not query.strip():
        raise ValueError("`query` must be a non-empty SQL string.")

    with duckdb.connect() as con:
        for dataset in run.datasets:
            con.register(dataset.table, dataset.df)
        con.register("data", run.datasets[0].df)  # backwards-compatible default alias
        try:
            result = con.execute(query).fetchdf()
        except duckdb.Error as exc:
            # Re-raise with the table list so the retry plugin gives the model
            # enough context to fix the query instead of guessing again.
            raise ValueError(
                f"SQL failed: {exc}. Available tables: {', '.join(run.table_names)}. "
                "Call preview_data_sources for exact column names."
            ) from exc

    truncated = len(result) > MAX_RESULT_ROWS
    if truncated:
        result = result.head(MAX_RESULT_ROWS)

    run.queries.append(query.strip())
    run.tables.append(result)
    run.note(f"SQL returned {len(result)} rows x {len(result.columns)} columns")

    tool_context.state["last_query"] = query.strip()
    tool_context.state["last_result_columns"] = [str(c) for c in result.columns]

    return {
        "columns": [str(c) for c in result.columns],
        "row_count": int(len(result)),
        "rows_preview": _jsonable_rows(result, MAX_ROWS_TO_MODEL),
        "truncated": truncated,
        "note": "The full result is held in memory. Pass column names to create_visualization.",
    }


def create_visualization(
    plot_type: str,
    x: str,
    tool_context: ToolContext,
    y: str = "",
    color: str = "",
    title: str = "",
    orientation: str = "v",
    barmode: str = "group",
    size: str = "",
    nbins: int = 0,
    source: str = "",
    target: str = "",
    value: str = "",
) -> dict:
    """Render a Plotly chart from the most recent `run_sql_query` result.

    The data is read from the last SQL result automatically - do not pass rows.
    Every column name you pass must exist in that result.

    Args:
        plot_type: One of line, bar, stacked_bar, 100%_stacked_bar,
            100%_single_stacked_bar, scatter, box, histogram, pie, heatmap, sankey.
        x: Column for the x axis (or category names for pie).
        y: Column for the y axis (numeric). Optional for histogram.
        color: Column used to split into series / colours.
        title: Chart title.
        orientation: "v" or "h" for bar charts.
        barmode: "group", "stack" or "overlay".
        size: Column controlling marker size for scatter charts.
        nbins: Bin count for histograms. 0 lets Plotly decide.
        source: Sankey source column.
        target: Sankey target column.
        value: Sankey value column.

    Returns:
        dict: {"status": "success", "plot_type": str, "points": int}
    """
    run = get_current_run()
    df = run.last_table
    if df is None or df.empty:
        raise ValueError("No query result available. Call run_sql_query first.")

    df = df.copy()
    plot_type = (plot_type or "").strip().lower()
    orientation = orientation if orientation in ("v", "h") else "v"

    def need(col: str, label: str) -> str:
        if not col:
            raise ValueError(f"`{label}` is required for plot_type='{plot_type}'.")
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' is not in the query result. Available: {list(df.columns)}"
            )
        return col

    color = color if color in df.columns else ""
    size = size if size in df.columns else ""

    if plot_type == "line":
        need(x, "x"), need(y, "y")
        fig = px.line(df, x=x, y=y, color=color or None, title=title)
        fig.update_traces(mode="lines+markers")

    elif plot_type == "bar":
        need(x, "x"), need(y, "y")
        plot_x, plot_y = x, y
        # A numeric category axis renders as a continuous scale; cast it to string.
        if orientation == "v" and pd.api.types.is_numeric_dtype(df[x]):
            plot_x = f"__{x}_cat"
            df[plot_x] = df[x].astype(str)
        elif orientation == "h" and pd.api.types.is_numeric_dtype(df[y]):
            plot_y = f"__{y}_cat"
            df[plot_y] = df[y].astype(str)
        fig = px.bar(
            df,
            x=plot_x,
            y=plot_y,
            color=color or None,
            barmode=barmode,
            orientation=orientation,
            title=title,
        )
        magnitude = df[y] if orientation == "v" else df[x]
        if pd.api.types.is_numeric_dtype(magnitude) and len(magnitude.dropna()):
            axis_update = fig.update_yaxes if orientation == "v" else fig.update_xaxes
            spread = float(magnitude.max() - magnitude.min()) or 1.0
            pad = 0.05 * spread
            axis_update(range=[float(magnitude.min()) - pad, float(magnitude.max()) + pad])
            median = float(magnitude.median())
            if float(magnitude.min()) > 0 and float(magnitude.max()) / max(1e-9, median) > 100:
                axis_update(type="log")

    elif plot_type == "stacked_bar":
        need(x, "x"), need(y, "y"), need(color, "color")
        df[color] = df[color].fillna("Unknown").astype(str)
        fig = px.bar(
            df, x=x, y=y, color=color, barmode="stack", orientation=orientation, title=title
        )

    elif plot_type == "100%_stacked_bar":
        need(x, "x"), need(y, "y"), need(color, "color")
        df = df[df[x].notna() & df[color].notna()]
        df_pct = df.groupby([x, color], as_index=False)[y].sum().rename(columns={y: "value"})
        df_pct["pct"] = df_pct.groupby(x)["value"].transform(lambda s: s / s.sum() * 100)
        fig = px.bar(
            df_pct, x=x, y="pct", color=color, barmode="stack", orientation=orientation, title=title
        )
        fig.update_yaxes(title="Percentage (%)", ticksuffix="%", range=[0, 100])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="inside")
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

    elif plot_type == "100%_single_stacked_bar":
        need(x, "x"), need(y, "y")
        df = df[df[x].notna()]
        totals = df.groupby(x, as_index=False)[y].sum().rename(columns={y: "value"})
        totals["pct"] = totals["value"] / totals["value"].sum() * 100
        totals["__bar"] = title or "Total"
        fig = px.bar(
            totals, x="__bar", y="pct", color=x, barmode="stack", title=title, custom_data=["value"]
        )
        fig.update_yaxes(title="Share of total (%)", ticksuffix="%", range=[0, 100])
        fig.update_xaxes(title="")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="inside")

    elif plot_type == "scatter":
        need(x, "x"), need(y, "y")
        fig = px.scatter(df, x=x, y=y, color=color or None, size=size or None, title=title)

    elif plot_type == "box":
        need(x, "x")
        fig = px.box(df, x=x, y=y or None, color=color or None, title=title)

    elif plot_type == "histogram":
        need(x, "x")
        fig = px.histogram(
            df, x=x, color=color or None, nbins=nbins if nbins and nbins > 0 else None, title=title
        )

    elif plot_type == "pie":
        need(x, "x"), need(y, "y")
        fig = px.pie(df, names=x, values=y, color=x, title=title)

    elif plot_type == "heatmap":
        need(x, "x"), need(y, "y")
        value_col = color if color else next(
            (c for c in df.columns if c not in (x, y) and pd.api.types.is_numeric_dtype(df[c])),
            None,
        )
        if value_col is None:
            raise ValueError("Heatmap needs a numeric value column; pass it as `color`.")
        pivot = df.pivot_table(index=y, columns=x, values=value_col, aggfunc="sum")
        fig = px.imshow(pivot, title=title, text_auto=True, aspect="auto")

    elif plot_type == "sankey":
        need(source, "source"), need(target, "target"), need(value, "value")
        sdf = df[df[source].notna() & df[target].notna()].copy()
        sdf[source] = sdf[source].astype(str).str.strip()
        sdf[target] = sdf[target].astype(str).str.strip()
        nodes = list(dict.fromkeys(sdf[source].tolist() + sdf[target].tolist()))
        index_of = {name: i for i, name in enumerate(nodes)}
        labels = []
        for node in nodes:
            total = sdf.loc[sdf[source] == node, value].sum() + sdf.loc[sdf[target] == node, value].sum()
            labels.append(f"{node} ({total:,.0f})")
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="freeform",
                    node=dict(
                        pad=30,
                        thickness=24,
                        line=dict(color="black", width=0.6),
                        label=labels,
                    ),
                    link=dict(
                        source=sdf[source].map(index_of),
                        target=sdf[target].map(index_of),
                        value=sdf[value],
                    ),
                )
            ]
        )
        fig.update_layout(title_text=title, font=dict(size=13))

    else:
        raise ValueError(
            f"Unsupported plot_type '{plot_type}'. Use one of: line, bar, stacked_bar, "
            "100%_stacked_bar, 100%_single_stacked_bar, scatter, box, histogram, pie, "
            "heatmap, sankey."
        )

    # Streamlit sizes width itself, so only height is fixed here.
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 17}},
        template="plotly_white",
        height=run.chart_height,
        margin=dict(l=50, r=40, t=70, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    if plot_type not in ("pie", "heatmap", "sankey"):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridcolor="LightGray")

    run.figures.append(fig)
    run.note(f"Rendered {plot_type} chart")
    tool_context.state["last_visualization"] = {
        "plot_type": plot_type,
        "x": x,
        "y": y,
        "color": color,
        "title": title,
    }
    return {"status": "success", "plot_type": plot_type, "points": int(len(df))}


# --------------------------------------------------------------------------------------
# Agent, App and Runner (ADK 2.0)
# --------------------------------------------------------------------------------------


def build_runner(model: str = DEFAULT_AGENT_MODEL) -> tuple[Runner, InMemorySessionService, str]:
    """Build the ADK 2.0 App and Runner. Cache the result; it is reusable across turns."""
    root_agent = Agent(
        name="data_agent",
        model=model,
        description="Answers questions about uploaded spreadsheets by querying them and charting the result.",
        instruction=AGENT_INSTRUCTION,
        tools=[preview_data_sources, run_sql_query, create_visualization],
        output_key="last_agent_response",
    )

    # App is the ADK 2.0 container for a workflow: root agent + plugins + config.
    app = App(
        name=APP_NAME,
        root_agent=root_agent,
        plugins=[ReflectAndRetryToolPlugin(max_retries=2)],
    )

    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    return runner, session_service, APP_NAME


async def run_agent_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
    on_event=None,
) -> str:
    """Stream one agent turn. Returns the final text response."""
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
    final_text = ""

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if on_event is not None:
            on_event(event)
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts)

    return final_text or "The agent finished without a text response."


def describe_event(event) -> Optional[str]:
    """One-line human summary of an ADK event, for the Streamlit activity log."""
    content = getattr(event, "content", None)
    if not content or not getattr(content, "parts", None):
        return None
    for part in content.parts:
        call = getattr(part, "function_call", None)
        if call:
            args = call.args or {}
            if call.name == "run_sql_query":
                return f"SQL -> {str(args.get('query', ''))[:300]}"
            if call.name == "create_visualization":
                return f"Chart -> {args.get('plot_type')} (x={args.get('x')}, y={args.get('y')})"
            return f"Calling `{call.name}`"
        response = getattr(part, "function_response", None)
        if response:
            return f"`{response.name}` returned"
    return None


# --------------------------------------------------------------------------------------
# Prompt refinement (plain google-genai call, no agent involved)
# --------------------------------------------------------------------------------------


def refine_prompt(user_query: str, api_key: str, model: str = DEFAULT_REFINE_MODEL) -> str:
    """Reformulate a free-form question into a structured analysis prompt."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_query,
        config=genai_types.GenerateContentConfig(
            system_instruction=REFINE_SYSTEM_INSTRUCTION,
            temperature=0,
        ),
    )
    return (response.text or user_query).strip()
