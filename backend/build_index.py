"""One-time build of a local SQLite index from the FEVER wiki-pages dump
(https://fever.ai/download/fever/wiki-pages.zip, ~5.4M Wikipedia pages,
June 2017 snapshot). Replaces live Wikipedia API calls in verify.py with a
local, deterministic, network-free retrieval index.

Usage: python build_index.py <path-to-wiki-pages-dir> <output-db-path>
"""
import glob
import json
import sqlite3
import sys


def build(wiki_pages_dir, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE pages (title TEXT PRIMARY KEY, lines TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE pages_fts USING fts5(title, intro, tokenize='porter unicode61')"
    )

    files = sorted(glob.glob(f"{wiki_pages_dir}/*.jsonl"))
    print(f"found {len(files)} shard files")

    page_rows = []
    fts_rows = []
    total = 0
    BATCH = 20000

    def flush():
        conn.executemany("INSERT OR IGNORE INTO pages VALUES (?, ?)", page_rows)
        conn.executemany("INSERT INTO pages_fts (title, intro) VALUES (?, ?)", fts_rows)
        page_rows.clear()
        fts_rows.clear()

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                title = doc.get("id", "")
                if not title:
                    continue
                text = doc.get("text", "")
                lines = doc.get("lines", "")
                intro = text[:1000]
                page_rows.append((title, lines))
                fts_rows.append((title, intro))
                total += 1
                if len(page_rows) >= BATCH:
                    flush()
                    conn.commit()
                    print(f"  {total:,} pages indexed", end="\r")
    if page_rows:
        flush()
        conn.commit()
    print(f"\ndone: {total:,} pages indexed into {db_path}")

    print("optimizing FTS index...")
    conn.execute("INSERT INTO pages_fts(pages_fts) VALUES('optimize')")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


if __name__ == "__main__":
    wiki_pages_dir, db_path = sys.argv[1], sys.argv[2]
    build(wiki_pages_dir, db_path)
