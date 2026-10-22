#!/usr/bin/env python3
"""Plot the last query result (or a CSV) as an interactive Plotly chart.

Usage:  python make_chart.py --type bar --x region --y revenue [--color product]
            [--title T] [--out chart.html] [--png]
Sankey:   --type sankey --source A --target B --value V
Heatmap:  --type heatmap --x col --y row --color value_col
Types: line bar stacked_bar 100%_stacked_bar 100%_single_stacked_bar scatter box
       histogram pie heatmap sankey
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from charts import PLOT_TYPES, build_figure  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=PLOT_TYPES, dest="plot_type")
    ap.add_argument("--x", default=""); ap.add_argument("--y", default="")
    ap.add_argument("--color", default=""); ap.add_argument("--title", default="")
    ap.add_argument("--orientation", default="v", choices=["v", "h"])
    ap.add_argument("--barmode", default="group")
    ap.add_argument("--size", default=""); ap.add_argument("--nbins", type=int, default=0)
    ap.add_argument("--source", default=""); ap.add_argument("--target", default="")
    ap.add_argument("--value", default="")
    ap.add_argument("--height", type=int, default=520)
    ap.add_argument("--workdir", default=".analytics")
    ap.add_argument("--csv", help="chart this CSV instead of the last query result")
    ap.add_argument("--out", default="chart.html")
    ap.add_argument("--png", action="store_true", help="also write a PNG (needs kaleido)")
    a = ap.parse_args()

    df = pd.read_csv(a.csv) if a.csv else pd.read_pickle(os.path.join(a.workdir, "last_result.pkl"))
    kwargs = {k: v for k, v in vars(a).items()
              if k not in ("csv", "out", "png", "workdir")}
    try:
        fig = build_figure(df, **kwargs)
    except ValueError as exc:
        sys.exit(f"Chart failed: {exc}")

    fig.write_html(a.out, include_plotlyjs=True, full_html=True)
    print(f"Wrote {a.out} ({a.plot_type}, {len(df):,} rows)")
    if a.png:
        png = os.path.splitext(a.out)[0] + ".png"
        try:
            fig.write_image(png, scale=2)
            print(f"Wrote {png}")
        except Exception as exc:  # kaleido missing or no Chrome
            print(f"PNG skipped: {exc}")


if __name__ == "__main__":
    main()
