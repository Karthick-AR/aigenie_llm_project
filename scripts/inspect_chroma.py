#!/usr/bin/env python3
"""Inspect Chroma SQLite store contents.

Usage examples (PowerShell):

# Basic table list + counts
python .\scripts\inspect_chroma.py

# Specify a different DB path and show 10 sample rows from each table
python .\scripts\inspect_chroma.py --db .\chroma_store\chroma.sqlite3 --limit 10

This script prints table names, row counts for common Chroma tables, and
optionally prints sample rows from selected tables for quick inspection.
"""

import sqlite3
import argparse
import os
import textwrap
import json


def connect(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def count_table(conn, table_name: str):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception as e:
        return f"error: {e}"


def sample_rows(conn, table_name: str, limit: int = 5):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return f"error: {e}"


def show_summary(db_path: str, limit: int = 5, show_sample: bool = True):
    conn = connect(db_path)
    print(f"Connected to: {db_path}")
    tables = list_tables(conn)
    print("\nTables found:")
    for t in tables:
        c = count_table(conn, t)
        print(f" - {t}: {c}")

    common = [
        'collections',
        'embeddings',
        'embedding_metadata',
        'segment_metadata',
        'collection_metadata',
    ]

    if show_sample:
        print('\nSample rows (limit={})'.format(limit))
        for t in common:
            if t in tables:
                print('\n--', t, f'(count={count_table(conn,t)})')
                s = sample_rows(conn, t, limit)
                if isinstance(s, str):
                    print('   ', s)
                elif len(s) == 0:
                    print('   (no rows)')
                else:
                    # Pretty-print first row or a few rows
                    for i, r in enumerate(s):
                        print('   [', i, ']', json.dumps(r, ensure_ascii=False))

    conn.close()


def main():
    p = argparse.ArgumentParser(
        description="Inspect Chroma sqlite DB: list tables, counts, and sample rows."
    )
    p.add_argument(
        "--db",
        default=os.path.join(os.getcwd(), "chroma_store", "chroma.sqlite3"),
        help="Path to Chroma sqlite DB (default: ./chroma_store/chroma.sqlite3)",
    )
    p.add_argument("--limit", type=int, default=5, help="Sample rows per table")
    p.add_argument(
        "--no-sample", action="store_true", help="Do not print table sample rows"
    )
    args = p.parse_args()

    try:
        show_summary(args.db, limit=args.limit, show_sample=not args.no_sample)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
