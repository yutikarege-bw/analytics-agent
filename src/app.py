"""
Streamlit front end for the ADK 2.0 text-to-visualization agent.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_core import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_REFINE_MODEL,
    LoadedTable,
    RunState,
    build_runner,
    describe_event,
    load_datasets,
    refine_prompt,
    run_agent_turn,
    set_current_run,
)

load_dotenv()

st.set_page_config(page_title="Text to Visualization", page_icon="📊", layout="wide")

MODEL_CHOICES = [
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


# --------------------------------------------------------------------------------------
# Session plumbing
# --------------------------------------------------------------------------------------


def get_loop() -> asyncio.AbstractEventLoop:
    """One long-lived event loop per Streamlit session.

    Streamlit reruns the script on every interaction; asyncio.run() would create and tear
    down a loop each time, which breaks anything the ADK runner keeps across turns.
    """
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
    loop = st.session_state.event_loop
    # Streamlit may run the script on a different thread each rerun; rebind the loop.
    asyncio.set_event_loop(loop)
    return loop


@st.cache_resource(show_spinner=False)
def get_runner(model: str, api_key_fingerprint: str):
    """Build the ADK App/Runner once per (model, key) pair."""
    return build_runner(model=model)


@st.cache_data(show_spinner=False)
def cached_datasets(file_bytes: bytes, file_name: str, auto_detect: bool) -> list[LoadedTable]:
    """Parse one upload into tables, cached on the file's bytes."""
    suffix = os.path.splitext(file_name)[1] or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return load_datasets(tmp_path, file_name, auto_detect=auto_detect)
    finally:
        os.unlink(tmp_path)


def ensure_session(runner, session_service, app_name: str) -> str:
    """Create the ADK session lazily and reuse its id for the whole conversation."""
    if "adk_session_id" not in st.session_state:
        st.session_state.adk_user_id = f"user_{uuid.uuid4().hex[:8]}"
        loop = get_loop()
        session = loop.run_until_complete(
            session_service.create_session(
                app_name=app_name, user_id=st.session_state.adk_user_id
            )
        )
        st.session_state.adk_session_id = session.id
    return st.session_state.adk_session_id


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    def _secret(name: str) -> str:
        try:
            return st.secrets.get(name, "")  # raises if no secrets.toml exists
        except Exception:
            return ""

    default_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or _secret("GEMINI_API_KEY")
        or _secret("GOOGLE_API_KEY")
    )
    api_key = st.text_input(
        "Gemini API key",
        value=default_key or "",
        type="password",
        help="Or set GEMINI_API_KEY / GOOGLE_API_KEY in a .env file.",
    )

    agent_model = st.selectbox("Agent model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_AGENT_MODEL) if DEFAULT_AGENT_MODEL in MODEL_CHOICES else 0)
    use_refiner = st.toggle("Rewrite analytical questions", value=True, help="A quick Gemini pass that restructures data questions. Chit-chat and follow-ups pass through untouched.")
    refine_model = st.selectbox("Refiner model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_REFINE_MODEL) if DEFAULT_REFINE_MODEL in MODEL_CHOICES else 0, disabled=not use_refiner)
    chart_height = st.slider("Chart height (px)", 320, 900, 520, step=20)

    st.divider()
    uploads = st.file_uploader(
        "CSV or Excel files",
        type=["xlsx", "xlsm", "xls", "csv", "tsv", "txt"],
        accept_multiple_files=True,
        help="Upload several files to join across them. Every sheet in a workbook is loaded.",
    )
    auto_detect = st.toggle(
        "Auto-detect table layout",
        value=True,
        help=(
            "Finds the real header row and splits sheets that hold several tables "
            "side by side. Turn off to read each sheet straight through with row 1 "
            "as the header."
        ),
    )

    st.divider()
    if st.button("Reset conversation", width="stretch"):
        for key in ("messages", "adk_session_id", "adk_user_id", "last_result"):
            st.session_state.pop(key, None)
        st.rerun()

if api_key:
    # ADK reads the key from the environment; explicitly stay off Vertex AI.
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


# --------------------------------------------------------------------------------------
# Header + data preview
# --------------------------------------------------------------------------------------

st.title("📊 Data Chat & Visualization")
st.caption("Ask about your data in plain language. Answers come back as text, a table, or a chart — whichever fits the question.")


def build_run_state() -> RunState:
    """Parse every upload into one RunState. Cheap: the parsing itself is cached."""
    run = RunState(chart_height=chart_height)
    run.previous_result = st.session_state.get("last_result")
    for upload in uploads or []:
        try:
            for table in cached_datasets(upload.getvalue(), upload.name, auto_detect):
                run.add(
                    table.df,
                    source=upload.name,
                    sheet=table.sheet,
                    label=table.label,
                    note=table.note,
                )
        except Exception as exc:  # one bad file shouldn't block the others
            st.warning(f"Couldn't read **{upload.name}**: {exc}")
    return run


data_state = build_run_state()

if data_state.datasets:
    total_rows = sum(len(d.df) for d in data_state.datasets)
    label = f"{len(data_state.datasets)} table" + ("s" if len(data_state.datasets) != 1 else "")
    with st.expander(f"Loaded data — {label}, {total_rows:,} rows total", expanded=False):
        tabs = st.tabs([d.table for d in data_state.datasets])
        for tab, dataset in zip(tabs, data_state.datasets):
            with tab:
                origin = f"{dataset.source} › {dataset.sheet}" if dataset.sheet else dataset.source
                if dataset.label:
                    origin += f" › “{dataset.label}”"
                st.caption(
                    f"`FROM {dataset.table}` — {origin} — "
                    f"{len(dataset.df):,} rows × {len(dataset.df.columns)} columns"
                    + (f"  \n_{dataset.note}_" if dataset.note else "")
                )
                st.dataframe(dataset.df.head(50), width="stretch")
                st.write("**Columns:** " + ", ".join(str(c) for c in dataset.df.columns))
else:
    st.info(
        "Upload one or more CSV or Excel files in the sidebar to analyse them. "
        "You can still chat without any data loaded."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("refined"):
            with st.expander("Refined prompt"):
                st.markdown(msg["refined"])
        if msg.get("log"):
            with st.expander("Agent activity", expanded=False):
                for line in msg["log"]:
                    st.markdown(f"- {line}")
        for q_i, query in enumerate(msg.get("queries", [])):
            st.code(query, language="sql")
        for t_i, table in enumerate(msg.get("tables", [])):
            st.dataframe(table, width="stretch", height=min(320, 40 + 28 * len(table)))
        for f_i, fig in enumerate(msg.get("figures", [])):
            st.plotly_chart(fig, width="stretch", key=f"fig_{idx}_{f_i}")
        if msg.get("content"):
            st.markdown(msg["content"])


for i, msg in enumerate(st.session_state.messages):
    render_message(msg, i)


# --------------------------------------------------------------------------------------
# Chat turn
# --------------------------------------------------------------------------------------

question = st.chat_input("Ask about your data, or just say hello")

if question:
    if not api_key:
        st.error("Add your Gemini API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

    runner, session_service, app_name = get_runner(agent_model, api_key[-6:])
    session_id = ensure_session(runner, session_service, app_name)
    loop = get_loop()

    run = data_state
    set_current_run(run)

    with st.chat_message("assistant"):
        refined = ""
        if use_refiner:
            with st.spinner("Thinking about the question..."):
                rewritten = refine_prompt(question, api_key=api_key, model=refine_model)
            # The refiner returns chit-chat and follow-ups verbatim; only surface a
            # rewrite when it genuinely restructured the question.
            if rewritten.strip().lower() != question.strip().lower():
                refined = rewritten
                with st.expander("Rewritten as"):
                    st.markdown(refined)

        if run.datasets:
            catalog = "\n".join(
                f"- `{d.table}` (from {d.source}"
                + (f" › {d.sheet}" if d.sheet else "")
                + (f" › “{d.label}”" if d.label else "")
                + "): "
                + ", ".join(str(c) for c in d.df.columns)
                for d in run.datasets
            )
            prompt = f"Tables currently loaded:\n{catalog}\n\n{refined or question}"
        else:
            prompt = f"No data is loaded yet.\n\n{refined or question}"

        status = st.status("Working...", expanded=True)

        def on_event(event) -> None:
            line = describe_event(event)
            if line:
                status.write(line)

        try:
            answer = loop.run_until_complete(
                run_agent_turn(
                    runner=runner,
                    user_id=st.session_state.adk_user_id,
                    session_id=session_id,
                    prompt=prompt,
                    on_event=on_event,
                )
            )
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # surface failures instead of a blank chat bubble
            status.update(label="Failed", state="error", expanded=True)
            answer = f"The agent run failed: `{exc}`"

        for query in run.queries:
            st.code(query, language="sql")
        for table in run.tables:
            st.dataframe(table, width="stretch", height=min(320, 40 + 28 * len(table)))
        for f_i, fig in enumerate(run.figures):
            st.plotly_chart(fig, width="stretch", key=f"live_fig_{len(st.session_state.messages)}_{f_i}")
        st.markdown(answer)

    if run.tables:
        st.session_state.last_result = run.tables[-1]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "refined": refined,
            "log": run.log,
            "queries": run.queries,
            "tables": run.tables,
            "figures": run.figures,
        }
    )