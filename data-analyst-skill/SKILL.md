---
name: data-analyst
description: Analyze CSV and Excel files with SQL and answer in prose, a table, or an interactive Plotly chart. Use this whenever the user uploads or points at a .csv, .tsv, .xlsx, .xlsm or .xls file and asks a question about the data - totals, rankings, comparisons, trends, "what's in this file", joins across files or sheets - or asks to plot, chart, graph or visualise it. Especially use it for messy, human-built workbooks (title rows above headers, several tables side by side on one sheet, semicolon CSVs, odd encodings) where plain pandas read_excel would mis-read the header. Don't wait for the word "chart" or "SQL".
---

# Data analyst

Answer questions about spreadsheets by loading them into DuckDB, querying, and replying in
whichever form the question calls for: a sentence, a table, or a chart.

The scripts do the fiddly parts reliably (header detection, multi-table sheets, encodings,
chart construction). Your job is judgement: pick the right tables, write correct SQL, and
choose the right form of answer.

## Workflow

1. **Load** - `python scripts/load_data.py FILE [FILE ...]`
   Prints each table's exact SQL name, columns, dtypes, sample rows and how it was located.
   Read this before writing SQL: users' wording rarely matches real column names, and
   workbook sheets are often split into several tables.
2. **Query** - `python scripts/query.py "SELECT ..."`
   Prints a preview and keeps the full result for charting. Iterate until the result is
   exactly what the answer needs.
3. **Chart (only if wanted)** - `python scripts/make_chart.py --type bar --x ... --y ... --out chart.html`
4. **Reply**, and send the chart file to the user if one was made.

State lives in `.analytics/` in the working directory (override with `--workdir`). Run the
scripts from this skill's folder or give their full path.

Dependencies: `pandas duckdb plotly openpyxl` (and `kaleido` only if you want PNGs). Install
missing ones with `pip install --break-system-packages`.

## Choose the form of the answer

Decide before querying, because it changes what the SQL should return:

- **Conversation** (greetings, "what can you do", anything not about the data): just reply.
  No scripts needed.
- **Answer in words** (a number, a ranking, a comparison, "what's in this file"): query,
  then state the result in prose. Do not chart it - a chart of a one-sentence answer is noise.
- **Chart** (user says plot / show me / draw / graph / visualise, or names a chart type):
  query, then chart.
- **Ambiguous**: answer in words and offer a chart at the end, rather than making one
  uninvited.

Follow-ups like "now as a pie chart" reuse the previous result: `make_chart.py` reads
`.analytics/last_result.pkl`, so skip re-querying unless the columns you need differ.

## Writing SQL (DuckDB)

- Use the exact table names from the catalogue in `FROM`. The first table is also `data`.
- Double-quote columns with spaces or symbols: `SELECT "FY-1 Payment (EUR)" FROM input`.
- Join across files and sheets freely when the question needs it.
- If tables have a `row_id` column they came from side-by-side blocks of one sheet and are
  row-aligned: join on `row_id` to put them back together. Splitting them apart otherwise
  loses that alignment.
- Aggregate in SQL (`GROUP BY`) so the result is answer-sized rather than raw rows.
- For charts return long format - one row per (category, series, value). If the source is
  wide, unpivot with `UNPIVOT` or `UNION ALL`. Filter NULL categories out of stacked bars.
- If a query errors, the message lists the available tables; fix and retry. If the same
  step fails three times, stop and explain the problem instead of guessing.

## Charts

`make_chart.py --type` accepts: `line`, `bar`, `stacked_bar`, `100%_stacked_bar`,
`100%_single_stacked_bar` (one bar showing how a single total splits), `scatter`, `box`,
`histogram`, `pie`, `heatmap`, `sankey`.

- Column args (`--x --y --color`) must be columns of the last query result.
- `stacked_bar` and `100%_stacked_bar` need `--color` for the series.
- Sankey takes `--source --target --value` instead of `--y`.
- Heatmap takes the numeric column as `--color`.
- `--title`, `--orientation h`, `--size` (scatter), `--nbins` (histogram), `--height` are optional.
- Output is a self-contained interactive HTML file. Add `--png` for a static image when
  `kaleido` is available.

## Messy workbooks

Auto-detection is on by default. Each sheet is scanned for the tables it contains: blank
rows/columns separate blocks, the header row is found rather than assumed, and a title
above a block becomes part of its table name (`working_sheet_question_3`). The catalogue's
`note` says how each table was found ("header taken from row 3"). It is a heuristic, so if
a table looks wrong - merged header cells, or two tables touching with no blank gap - say
so, then retry with `--no-detect` (row 1 is the header) or ask the user to tidy the sheet.

## Writing the answer

State the numbers with thousands separators and the unit from the column name. Add one
short line naming the tables and columns you used so the user can check you matched their
wording. Keep it to a few sentences unless they asked for more. If the data cannot answer
the question, say so plainly and name what is missing; never invent columns.
