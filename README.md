# Analytics Agent

Chat with your CSV/Excel data in plain language and get answers back as text, a table, or a Plotly chart. Built with Streamlit, Google ADK + Gemini, and DuckDB.

## Setup

```bash
conda create -n analytics-agent python=3.11 -y
conda activate analytics-agent
conda install pip
pip install -r requirements.txt
```

Add your Gemini API key to a `.env` file in `src/`:

```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
cd src
streamlit run app.py
```

Upload a CSV or Excel file in the sidebar, then ask questions in the chat.
