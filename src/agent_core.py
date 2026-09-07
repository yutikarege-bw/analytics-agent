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
import math
import os
import re
from decimal import Decimal
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

REFINE_SYSTEM_INSTRUCTION = """You reformulate user messages into structured prompts for data analysis and visualization.

FIRST decide whether the message is an analytical request about a dataset.
If it is NOT - a greeting, thanks, chit-chat, a question about the assistant itself, a
vague follow-up like "now as a pie chart", or anything with nothing to restructure -
return the message EXACTLY as written, unchanged. Do not expand it, do not add
assumptions, do not turn it into a data question.

If it IS an analytical request:
1. Identify the key data points, dimensions, metrics, filters, or chart types mentioned.
2. Reformulate it into a clear, concise, structured prompt another model can act on.
3. Preserve every data point, relationship and visualization intent from the original.
4. Where the request is ambiguous, make reasonable assumptions but preserve intent.
5. Do not invent a chart request. If the user asked a question in words, keep it a
   question in words - only mention a chart if they asked for one.
6. When a TOTAL is mentioned, it must appear in the output. When it is not mentioned,
   do not add one.

Return only the resulting prompt, with no preamble or commentary."""

AGENT_INSTRUCTION = """You are a data analyst assistant. The user has uploaded spreadsheets, and each
turn tells you which tables and columns are currently loaded.

FIRST decide what kind of reply the message needs:

A. CONVERSATION - greetings, thanks, "what can you do", questions about your own
   behaviour, or anything not about the data. Just reply in plain language.
   Call no tools at all.

B. ANALYSIS IN WORDS - a question about the data whose answer is a number, a list, a
   comparison, an explanation, or a "what's in this file" overview. Query the data and
   answer in prose. Do NOT make a chart.

C. CHART - the user asks to see, plot, draw, graph, visualise, or explicitly names a
   chart type. Query the data, then call `create_visualization`.

If a data question is ambiguous between B and C, answer in words and offer a chart at
the end rather than making one uninvited. Never make a chart for a question that was
already fully answered by a sentence.

Tools:

- `preview_data_sources` gives sample values, dtypes and exact column names. Call it
  when you need to see actual values or aren't sure what a column contains. You do not
  need it for a simple lookup if the table catalogue above already names the column.

- `run_sql_query` runs DuckDB SQL. Use the exact `table` names from the catalogue in
  your FROM clause. Several files, and several blocks of one sheet, can be loaded at
  once - join across them when the question needs it. The first table is also aliased
  as `data`.
  - Quote column names containing spaces or symbols: SELECT "FY-1 Payment (EUR)" FROM input
  - If tables carry a `row_id` column, they came from side-by-side blocks of one sheet
    and are row-aligned: join them on `row_id` to line their values back up.
  - Aggregate in SQL (GROUP BY) so the result is answer-sized, not raw rows.
  - For a chart, return LONG format: one row per (category, series, value). If the
    source is wide, unpivot in SQL with UNION ALL or UNPIVOT.
  - Exclude NULL/empty categories when building a stacked or 100% stacked bar.

- `create_visualization` reads the result of the most recent `run_sql_query`
  automatically, so you pass only column names and chart options, never the data.
  - `x`, `y` and `color` must be column names present in that SQL result.
  - For a Sankey pass `source`, `target` and `value` instead of `y`. For a heatmap pass
    the numeric value column as `color`. Use `100%_single_stacked_bar` for one bar
    showing how a single total splits across categories.
  - Supported `plot_type`: line, bar, stacked_bar, 100%_stacked_bar,
    100%_single_stacked_bar, scatter, box, histogram, pie, heatmap, sankey.

Answering: state the numbers you found, with thousands separators and the unit from the
column name. Say in one short line which tables and columns you used, so the user can
check you matched their wording correctly. Keep it to a few sentences unless asked for
more.

If the data cannot answer the question, say so plainly and name what is missing. Do not
invent columns. Call any single tool at most 3 times per request; if it keeps failing,
stop and explain the problem."""


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
    label: str = ""  # title found above the block within the sheet
    note: str = ""  # how the table was located


@dataclass
class RunState:
    """Everything a single agent run produced, for the UI to render."""

    datasets: list[Dataset] = field(default_factory=list)
    previous_result: Optional[pd.DataFrame] = None  # last turn's SQL result
    preview: Optional[pd.DataFrame] = None
    queries: list[str] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list[go.Figure] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    chart_height: int = 520

    @property
    def last_table(self) -> Optional[pd.DataFrame]:
        """This turn's latest result, falling back to the previous turn's."""
        return self.tables[-1] if self.tables else self.previous_result

    @property
    def table_names(self) -> list[str]:
        return [d.table for d in self.datasets]

    def note(self, message: str) -> None:
        self.log.append(message)

    def add(
        self, df: pd.DataFrame, source: str, sheet: str = "", label: str = "", note: str = ""
    ) -> Dataset:
        """Register a DataFrame under a unique SQL-safe table name."""
        stem = os.path.splitext(source)[0]
        if label and sheet:
            base = sanitize_table_name(f"{sheet}_{label[:24]}")
        else:
            base = sanitize_table_name(label[:24] if label else (sheet or stem))
        # Same sheet name in two different files: disambiguate with the file stem.
        clash = next((d for d in self.datasets if d.table == base), None)
        if clash is not None and clash.source != source:
            base = sanitize_table_name(f"{stem}_{sheet or label}")
        name, n = base, 2
        while any(d.table == name for d in self.datasets):
            name, n = f"{base}_{n}", n + 1
        dataset = Dataset(table=name, df=df, source=source, sheet=sheet, label=label, note=note)
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
    """Normalise blank-ish placeholder strings so DuckDB and Plotly behave predictably.

    Numeric columns keep their native NaN - DuckDB reads that as NULL correctly, and
    forcing None into a float column just becomes NaN again. Anything crossing the
    wire to the model is sanitised separately by `json_safe`.
    """
    # Select by exclusion: pandas 2 stores text as `object`, pandas 3 as `str`, so
    # testing for a specific text dtype silently matches nothing on one of them.
    text_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if text_cols:
        df[text_cols] = df[text_cols].replace(r"^\s*$", None, regex=True)
        df[text_cols] = df[text_cols].replace(["nan", "NaN", "null", "NULL", "None"], None)
    return df


def sanitize_table_name(raw: str) -> str:
    """Turn a file, sheet or block label into something safe to type in a FROM clause."""
    name = re.sub(r"\W+", "_", str(raw).strip().lower()).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name[:48]


@dataclass
class LoadedTable:
    """One table extracted from a file: a CSV, a sheet, or a block within a sheet."""

    df: pd.DataFrame
    sheet: str = ""
    label: str = ""  # title found above the block, e.g. "Question 2:"
    note: str = ""  # how it was found, shown in the UI


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NaT:
        return True
    return isinstance(value, str) and not value.strip()


def _is_numberish(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", "").replace("%", "").strip())
            return True
        except ValueError:
            return False
    return False


def _runs(flags: list[bool]) -> list[list[int]]:
    """Index runs where flags is True, e.g. [T,T,F,T] -> [[0,1],[3]]."""
    out: list[list[int]] = []
    current: list[int] = []
    for i, flag in enumerate(flags):
        if flag:
            current.append(i)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _dedupe_columns(names: list[str]) -> list[str]:
    """DuckDB rejects duplicate column names; suffix repeats."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _detect_header(block: pd.DataFrame, mask: pd.DataFrame) -> tuple[Optional[int], str]:
    """Find the header row of a block, plus any title text sitting above it.

    A header row is mostly-filled, mostly non-numeric, and sits directly above a body
    whose numeric columns are actually numeric. Rows above it (a stray "Question 2:"
    title, a note, a merged banner) become the block's label.
    """
    n_rows = len(block)
    if n_rows == 0:
        return None, ""

    look = min(15, n_rows)
    tail = block.iloc[min(look, n_rows - 1) :] if n_rows > look else block
    if tail.empty:
        tail = block

    # Which columns does the body treat as numeric?
    numeric_cols = []
    for pos in range(block.shape[1]):
        values = [v for v in tail.iloc[:, pos] if not _is_blank(v)]
        if values and sum(_is_numberish(v) for v in values) / len(values) > 0.7:
            numeric_cols.append(pos)

    candidates = []
    for i in range(look):
        fill = mask.iloc[i].mean()
        if fill < 0.6:
            continue  # a sparse row is a title, not a header
        row = block.iloc[i]
        cells = [v for v in row if not _is_blank(v)]
        if not cells:
            continue
        texty = sum(isinstance(v, str) and not _is_numberish(v) for v in cells) / len(cells)
        if texty < 0.6:
            continue
        if numeric_cols:
            # A real header sits above numbers: its cells in numeric columns are text.
            clashes = sum(1 for pos in numeric_cols if _is_numberish(block.iloc[i, pos]))
            if clashes > len(numeric_cols) * 0.3:
                continue
        candidates.append(i)

    if not candidates:
        return None, _label_from_rows(block, mask, 0)

    # The last qualifying row before the body: handles a title row that is also all text.
    header_idx = candidates[-1] if numeric_cols else candidates[0]
    return header_idx, _label_from_rows(block, mask, header_idx)


def _label_from_rows(block: pd.DataFrame, mask: pd.DataFrame, stop: int) -> str:
    """First meaningful text in the rows above the header - the block's title."""
    for i in range(min(stop, len(block))):
        for value in block.iloc[i]:
            if isinstance(value, str) and value.strip() and not _is_numberish(value):
                return value.strip().rstrip(":")[:60]
    return ""


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim, drop empty rows/columns, normalise blanks."""
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.loc[:, [bool(c) for c in frame.columns]]
    frame.columns = _dedupe_columns(list(frame.columns))
    return preprocess_dataframe(frame)  # index preserved: carries the sheet row number


def extract_tables(raw: pd.DataFrame, split_columns: bool = True) -> list[LoadedTable]:
    """Split a raw headerless grid into the rectangular tables it actually contains.

    Real workbooks put several tables side by side on one sheet, separated by blank
    spacer columns, each under its own title. Blank rows and columns are the separators,
    so group on them and detect a header inside each resulting rectangle.
    """
    if raw.empty:
        return []

    mask = raw.map(lambda v: not _is_blank(v))
    col_groups = (
        _runs(list(mask.any(axis=0)))
        if split_columns
        else [[i for i in range(raw.shape[1]) if mask.iloc[:, i].any()]]
    )

    tables: list[LoadedTable] = []
    for cols in col_groups:
        if not cols:
            continue
        sub, sub_mask = raw.iloc[:, cols], mask.iloc[:, cols]
        for rows in _runs(list(sub_mask.any(axis=1))):
            if len(rows) < 2:  # a lone stray cell is not a table
                continue
            block, block_mask = sub.iloc[rows], sub_mask.iloc[rows]
            header_idx, label = _detect_header(block.reset_index(drop=True), block_mask.reset_index(drop=True))

            if header_idx is None:
                frame = block.reset_index(drop=True)
                frame.columns = [f"col_{i + 1}" for i in range(frame.shape[1])]
                note = "no header row found - columns named col_1, col_2, ..."
            else:
                body = block.iloc[header_idx + 1 :]
                if body.empty:
                    continue
                frame = body.reset_index(drop=True)
                names = [
                    str(v).strip() if not _is_blank(v) else f"col_{i + 1}"
                    for i, v in enumerate(block.iloc[header_idx])
                ]
                frame.columns = names
                note = f"header taken from row {rows[header_idx] + 1} of the sheet"

            frame = _finalize(frame)
            if frame.empty or not len(frame.columns):
                continue
            frame.attrs["sheet_rows"] = [int(i) + 1 for i in frame.index]
            tables.append(LoadedTable(df=frame.reset_index(drop=True), label=label, note=note))

            if len(tables) >= 20:  # guard against pathological sheets
                return tables
    return tables


def read_csv_robust(path: str, header: Any = 0) -> pd.DataFrame:
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
            return pd.read_csv(path, sep=sep, encoding=encoding, header=header)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            last_error = exc
            try:  # engine="python" with sep=None infers the delimiter itself
                return pd.read_csv(path, sep=None, engine="python", encoding=encoding, header=header)
            except Exception as exc2:  # noqa: BLE001 - retried under the next encoding
                last_error = exc2
                continue

    raise ValueError(f"Could not parse '{os.path.basename(path)}' as delimited text: {last_error}")


def load_datasets(path: str, display_name: str, auto_detect: bool = True) -> list[LoadedTable]:
    """Read one uploaded file into a list of queryable tables.

    With auto_detect on, each sheet is scanned for the tables it actually contains -
    leading blank rows, floating titles and side-by-side blocks are handled. With it
    off, each sheet is read straight through with the first row as the header.
    """
    lower = display_name.lower()
    is_delimited = lower.endswith((".csv", ".tsv", ".txt"))

    if is_delimited:
        if not auto_detect:
            frame = pd.read_csv(path, sep="\t") if lower.endswith(".tsv") else read_csv_robust(path)
            return [LoadedTable(df=_finalize(frame), note="first row used as header")]
        raw = (
            pd.read_csv(path, sep="\t", header=None)
            if lower.endswith(".tsv")
            else read_csv_robust(path, header=None)
        )
        # Delimited files are one table; only strip preamble rows, never split columns.
        tables = extract_tables(raw, split_columns=False)
        return tables or [LoadedTable(df=_finalize(raw), note="raw grid")]

    book = pd.ExcelFile(path)
    out: list[LoadedTable] = []
    for sheet in book.sheet_names:
        if auto_detect:
            raw = book.parse(sheet, header=None)
            found = extract_tables(raw)
            for table in found:
                table.sheet = str(sheet)
                if len(found) > 1:
                    # Side-by-side blocks are row-aligned in the sheet, but splitting
                    # them loses that. row_id preserves the link so they can be joined.
                    rows = table.df.attrs.get("sheet_rows")
                    if rows and len(rows) == len(table.df):
                        table.df.insert(0, "row_id", rows)
                        table.note += "; row_id = sheet row, join blocks from this sheet on it"
            out.extend(found)
        else:
            frame = _finalize(book.parse(sheet))
            if not frame.empty and len(frame.columns):
                out.append(LoadedTable(df=frame, sheet=str(sheet), note="first row used as header"))

    if not out:
        raise ValueError(f"'{display_name}' has no sheet with usable data.")
    return out


def _clean_scalar(value: Any) -> Any:
    """Coerce one cell into something `json.dumps(..., allow_nan=False)` accepts.

    Operating on values rather than DataFrames is deliberate. `df.where(notna, None)`
    looks like it removes NaN, but on a float column pandas stores None back as NaN,
    so the scrub silently does nothing and the NaN reaches the JSON encoder - which
    emits a bare `NaN` token that the Gemini API rejects with a 400.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "isoformat"):  # date, time, datetime
        return value.isoformat()
    if isinstance(value, Decimal):
        return _clean_scalar(float(value))
    if hasattr(value, "item"):  # numpy scalar -> python scalar, then re-check
        try:
            return _clean_scalar(value.item())
        except (ValueError, AttributeError):
            return str(value)
    try:
        if pd.isna(value):  # pd.NA and friends
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return [_clean_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean_scalar(v) for k, v in value.items()}
    return str(value)


def json_safe(obj: Any) -> Any:
    """Recursively sanitise a tool's return value before it goes back to the model."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return _clean_scalar(obj)


def _jsonable_rows(df: pd.DataFrame, limit: int) -> list[dict]:
    """Rows safe to hand back to the model: no NaN, no inf, no numpy scalars."""
    records = df.head(limit).to_dict(orient="records")
    return [{str(k): _clean_scalar(v) for k, v in row.items()} for row in records]


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
    return json_safe({
        "tables": tables,
        "note": (
            "Query these using their `table` name. The first table is also aliased as `data`."
            if len(tables) > 1
            else "Query this table by its `table` name, or as `data`."
        ),
    })


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

    return json_safe(
        {
            "columns": [str(c) for c in result.columns],
            "row_count": int(len(result)),
            "rows_preview": _jsonable_rows(result, MAX_ROWS_TO_MODEL),
            "truncated": truncated,
            "note": "The full result is held in memory. Pass column names to create_visualization.",
        }
    )


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

    The data is read from the last SQL result automatically - do not pass rows. If the
    user asks to re-chart something from an earlier turn, that result is still available,
    but re-running the query is safer if the columns you need might differ.
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
        raise ValueError(
            "No query result to chart. Call run_sql_query first, then chart its columns."
        )

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
    return json_safe({"status": "success", "plot_type": plot_type, "points": int(len(df))})


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