#!/usr/bin/env python3
"""Load CSV/Excel files into a workspace of queryable tables.

Usage:  python load_data.py FILE [FILE ...] [--workdir DIR] [--no-detect]

Writes DIR/catalog.json (the table list Claude reads) and one pickle per table.
Prints a catalogue: exact SQL table names, columns, dtypes, sample rows, and how each
table was located in its sheet.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tables import load_datasets, sanitize_table_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workdir", default=".analytics")
    ap.add_argument("--no-detect", action="store_true",
                    help="read each sheet straight through with row 1 as header")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    used: dict[str, str] = {}  # table -> source
    catalog = []
    for path in a.files:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        try:
            loaded = load_datasets(path, name, auto_detect=not a.no_detect)
        except Exception as exc:  # one bad file shouldn't block the others
            print(f"!! Could not read {name}: {exc}", file=sys.stderr)
            continue
        for t in loaded:
            if t.label and t.sheet:
                base = sanitize_table_name(f"{t.sheet}_{t.label[:24]}")
            else:
                base = sanitize_table_name(t.label[:24] if t.label else (t.sheet or stem))
            if base in used and used[base] != name:  # same sheet name in two files
                base = sanitize_table_name(f"{stem}_{t.sheet or t.label}")
            table, n = base, 2
            while table in used:
                table, n = f"{base}_{n}", n + 1
            used[table] = name
            t.df.to_pickle(os.path.join(a.workdir, f"{table}.pkl"))
            catalog.append({
                "table": table, "source": name, "sheet": t.sheet, "label": t.label,
                "note": t.note, "rows": int(len(t.df)),
                "columns": [str(c) for c in t.df.columns],
                "dtypes": {str(k): str(v) for k, v in t.df.dtypes.items()},
            })
            sample = t.df.head(3).astype(str).to_dict("records")
            catalog[-1]["sample"] = sample

    if not catalog:
        sys.exit("No usable tables were loaded.")
    with open(os.path.join(a.workdir, "catalog.json"), "w") as fh:
        json.dump(catalog, fh, indent=2, default=str)

    print(f"Loaded {len(catalog)} table(s) into {a.workdir}/\n")
    for c in catalog:
        origin = c["source"] + (f" > {c['sheet']}" if c["sheet"] else "")
        if c["label"]:
            origin += f" > \"{c['label']}\""
        print(f"FROM {c['table']}   [{origin}]  {c['rows']:,} rows x {len(c['columns'])} cols")
        if c["note"]:
            print(f"  note: {c['note']}")
        print("  columns: " + ", ".join(f"{k} ({c['dtypes'][k]})" for k in c["columns"]))
        for row in c["sample"]:
            print("  e.g. " + json.dumps(row, ensure_ascii=False)[:240])
        print()
    print(f"The first table ({catalog[0]['table']}) is also aliased as `data` in queries.")


if __name__ == "__main__":
    main()
