# Text-to-Visualization — Streamlit + Google ADK 2.0

A Streamlit port of the `google-adk-text-to-visualization-agent-gemini` notebook.
Upload a spreadsheet, ask a question in plain language, and a Gemini-backed ADK agent
previews the file, writes DuckDB SQL, and renders a Plotly chart.

```
agent_core.py    tools + ADK 2.0 App/Runner + prompt refinement
app.py           Streamlit UI
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini API key
streamlit run app.py
```

The key can also be pasted into the sidebar, or put in `.streamlit/secrets.toml`.

## What changed from the notebook

### Notebook-only mechanics that had to go

| Notebook | Streamlit |
| --- | --- |
| `display(df)` and `fig.show()` inside tools | tools append to a `RunState`; the UI calls `st.dataframe` / `st.plotly_chart` |
| `input()` for the question | `st.chat_input`, with conversation history in `st.session_state` |
| top-level `await` in a cell | one long-lived event loop per Streamlit session (`get_loop()`), driven with `run_until_complete` |
| hardcoded `"ipo_data_2025_04.xlsx"` | multi-file `st.file_uploader`, cached by file bytes |
| `print()` progress | live tool-call trace in an `st.status` panel |

Streamlit reruns the whole script on every interaction, so the `Runner` is built once
with `@st.cache_resource` and the ADK session id is created once and kept in
`st.session_state` — otherwise every message would start a fresh conversation.

## Uploading CSV and Excel

Drop any number of `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xlsm` or `.xls` files into the
sidebar. Each becomes a queryable DuckDB table:

- **Every sheet of a workbook is loaded**, each as its own table — no sheet picker, and
  no re-uploading to look at a different tab.
- **Table names are derived and de-duplicated**: `ventes.csv` → `ventes`, a second
  `ventes.csv` → `ventes_2`, sheet `Q Data` → `q_data`, and a colliding sheet name gets
  prefixed with its workbook. The preview panel shows the exact `FROM` name for each.
- **The agent can join across files and sheets.** Upload `sales.csv` and `managers.csv`
  and ask for revenue by manager; it writes the join itself.
- The first table is also aliased as `data`, so single-file prompts written against the
  notebook's convention still work.

CSV parsing does not assume UTF-8 commas, because exports rarely are. The reader sniffs
the delimiter (`,` `;` `\t` `|`), tries `utf-8-sig` → `utf-8` → `cp1252` → `latin-1`, and
falls back to pandas' own inference on a parser error. Loading also strips whitespace from
headers, drops all-empty rows and columns, and renames duplicate columns (`x`, `x_1`)
since DuckDB rejects them. A file that still can't be read raises a warning naming that
file and the others load anyway.

When SQL fails, the error handed back to the model includes the list of available tables,
so `ReflectAndRetryToolPlugin` can correct a wrong table name on the retry instead of
guessing twice.

### ADK 2.0 specifics

- **`App` + `Runner(app=...)`.** ADK 2.0 wraps the root agent in an `App` container
  (`google.adk.apps.App`), which is where plugins and workflow config now live.
  `Runner(agent=..., app_name=...)` still works, but `App` is the 2.0 shape.
- **Tools raise instead of swallowing.** The notebook wrapped every tool body in
  `except Exception: return json.dumps({"error": ...})`. Under 2.0 that masks failures
  from the framework and disables automatic retries. Tools here raise `ValueError` with
  an actionable message, and `ReflectAndRetryToolPlugin(max_retries=2)` feeds the error
  back to the model so it can correct a bad column name or malformed SQL itself.
- **Dropped `UnsafeLocalCodeExecutor`.** The agent never needed to execute code — it has
  three tools. Removing it also removes an arbitrary-code-execution surface.
- **Dropped `AutomaticFunctionCallingConfig(maximum_remote_calls=1)`.** That setting
  belongs to the raw `google-genai` auto-calling loop, not ADK's, and capping it at 1 was
  in direct conflict with an agent that must make three sequential tool calls.
- **Model default is `gemini-flash-latest`.** `gemini-2.5-flash` is scheduled for
  shutdown on 2026-10-16; the alias tracks the current Flash model. Pin an explicit
  version in the sidebar if you need reproducibility.

### Tool design changes

The notebook passed JSON strings between the model and the tools and spent a large part
of the system prompt begging Gemini to keep `plot_type` at the root of the object rather
than nested inside `data`. That whole class of failure is gone here:

- **Typed, flat tool arguments.** `create_visualization(plot_type=..., x=..., y=...,
  color=...)` generates a real function schema, so the model can't nest the keys wrong.
- **Data never round-trips through the model.** `run_sql_query` keeps the full result
  frame in Python and returns only a 50-row preview; `create_visualization` reads the
  last result by reference. On a 5,000-row aggregate this is the difference between a
  chart and a context-window overflow, and it removes any chance of the model silently
  mangling values while re-serializing them.
- **Wide→long reshaping moved into SQL.** The instruction asks for `UNION ALL` / `UNPIVOT`
  rather than asking the model to restructure JSON by hand.

Chart types are unchanged: `line`, `bar`, `stacked_bar`, `100%_stacked_bar`,
`100%_single_stacked_bar`, `scatter`, `box`, `histogram`, `pie`, `heatmap`, `sankey`.
The fixed `width=1100, height=800` layout became a height slider with Streamlit handling
width, and legends moved below the plot so they don't get clipped in a narrow column.

## Known trade-offs

- `InMemorySessionService` means history dies with the process. Swap in
  `DatabaseSessionService` if you want conversations to survive a restart.
- Every uploaded table is held in memory and re-registered with DuckDB on each query.
  Fine up to a few hundred thousand rows in total; past that, write the frames to Parquet
  once and point DuckDB at the files instead.
- One agent run at a time per user. The `RunState` collector uses a `ContextVar` with a
  module-level fallback, which is safe for normal single-user Streamlit sessions but is
  not built for many concurrent runs in one process.
