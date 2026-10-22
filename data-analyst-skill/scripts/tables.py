"""Spreadsheet loading: messy CSV/Excel -> clean, queryable tables.

Lifted from the original analytics-agent. Handles delimiter/encoding sniffing, blank-row
padding, floating titles, and several tables side by side on one sheet.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


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
