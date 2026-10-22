#!/usr/bin/env python3
"""Run DuckDB SQL over the tables loaded by load_data.py.

Usage:  python query.py "SELECT ..." [--workdir DIR] [--rows 50] [--csv out.csv]
        python query.py --file query.sql

The full result is saved as DIR/last_result.pkl so make_chart.py can plot it without
the data passing through the conversation; only a preview is printed.
"""
import argparse, json, os, sys

import duckdb
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sql", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--workdir", default=".analytics")
    ap.add_argument("--rows", type=int, default=50, help="rows to print (full result is still saved)")
    ap.add_argument("--csv", help="also write the full result to this CSV")
    a = ap.parse_args()

    sql = open(a.file).read() if a.file else a.sql
    if not sql or not sql.strip():
        sys.exit("Provide SQL as an argument or via --file.")
    cat_path = os.path.join(a.workdir, "catalog.json")
    if not os.path.exists(cat_path):
        sys.exit(f"No catalog in {a.workdir}/. Run load_data.py first.")
    catalog = json.load(open(cat_path))

    with duckdb.connect() as con:
        for c in catalog:
            con.register(c["table"], pd.read_pickle(os.path.join(a.workdir, f"{c['table']}.pkl")))
        con.register("data", pd.read_pickle(os.path.join(a.workdir, f"{catalog[0]['table']}.pkl")))
        try:
            result = con.execute(sql).fetchdf()
        except duckdb.Error as exc:
            names = ", ".join(c["table"] for c in catalog)
            sys.exit(f"SQL failed: {exc}\nAvailable tables: {names}\n"
                     "Re-read the catalogue (columns with spaces need double quotes).")

    result.to_pickle(os.path.join(a.workdir, "last_result.pkl"))
    if a.csv:
        result.to_csv(a.csv, index=False)
    print(f"{len(result):,} rows x {len(result.columns)} columns: {list(result.columns)}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.head(a.rows).to_string(index=False))
    if len(result) > a.rows:
        print(f"... {len(result) - a.rows:,} more rows (saved in full to last_result.pkl)")


if __name__ == "__main__":
    main()
