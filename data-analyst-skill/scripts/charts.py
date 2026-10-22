"""Plotly figure builder. Lifted from the original analytics-agent's create_visualization."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOT_TYPES = [
    "line", "bar", "stacked_bar", "100%_stacked_bar", "100%_single_stacked_bar",
    "scatter", "box", "histogram", "pie", "heatmap", "sankey",
]


def build_figure(
    df: pd.DataFrame,
    plot_type: str,
    x: str = "",
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
    height: int = 520,
) -> go.Figure:
    if df is None or df.empty:
        raise ValueError("No query result to chart. Run a query first.")
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
        height=height,
        margin=dict(l=50, r=40, t=70, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    if plot_type not in ("pie", "heatmap", "sankey"):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridcolor="LightGray")
    return fig
