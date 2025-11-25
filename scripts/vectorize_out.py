#!/usr/bin/env python3
"""
Vectorize each non-empty line from a text file using an Ollama embedding model.

Example:
    poetry run python scripts/vectorize_out.py out.txt out_embeddings.json \
        --model nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ollama import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize lines from a text file and store them as JSON.",
    )
    parser.add_argument("input_txt", type=Path, help="Path to the text file (e.g., out.txt).")
    parser.add_argument("output_json", type=Path, help="Where to write the embedding JSON.")
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model to use (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of lines to vectorize.",
    )
    return parser.parse_args()


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def compute_embeddings(client: Client, lines: list[str], model: str, limit: int | None = None) -> list[dict]:
    records = []
    for idx, line in enumerate(lines):
        if not line:
            continue
        if limit is not None and len(records) >= limit:
            break
        try:
            response = client.embeddings(model=model, prompt=line)
            vector = response.get("embedding")
            if not vector:
                print(f"Warning: No embedding for line {idx + 1}, skipping", file=sys.stderr)
                continue
            records.append(
                {
                    "id": idx,
                    "text": line,
                    "model": model,
                    "embedding": vector,
                }
            )
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} lines, {len(records)} embeddings generated...", file=sys.stderr)
        except Exception as e:
            print(f"Error processing line {idx + 1}: {e}", file=sys.stderr)
            continue
    return records


def main() -> int:
    args = parse_args()
    if not args.input_txt.exists():
        raise SystemExit(f"Input text file not found: {args.input_txt}")

    # Check if Ollama is available
    client = Client()
    try:
        client.list()  # Test connection
    except Exception as e:
        print(f"Error: Cannot connect to Ollama server. Make sure Ollama is running.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return 1

    lines = load_lines(args.input_txt)
    print(f"Loaded {len(lines)} lines from {args.input_txt}", file=sys.stderr)
    
    start_time = time.time()
    records = compute_embeddings(client, lines, model=args.model, limit=args.limit)
    elapsed_time = time.time() - start_time
    
    if not records:
        raise SystemExit("No embeddings were generated.")

    write_start = time.time()
    args.output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_time = time.time() - write_start
    
    total_time = time.time() - start_time
    
    print(f"Wrote {len(records)} embeddings to {args.output_json}")
    print(f"Time: {elapsed_time:.2f}s vectorization, {write_time:.2f}s writing, {total_time:.2f}s total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

