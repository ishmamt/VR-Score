"""
scripts/jsonl_to_json.py — convert a JSONL results file to a JSON array.

Usage
-----
    python scripts/jsonl_to_json.py results/vr_scores.jsonl
    python scripts/jsonl_to_json.py results/vr_scores.jsonl --out results/vr_scores.json
    python scripts/jsonl_to_json.py results/vr_scores.jsonl --pretty
"""

import argparse
import json
import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path, pretty: bool) -> int:
    records = []
    errors  = 0

    with open(input_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed line {lineno}: {exc}", file=sys.stderr)
                errors += 1

    indent = 2 if pretty else None
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=indent)
        fh.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")
    if errors:
        print(f"Skipped {errors} malformed line(s).", file=sys.stderr)
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Convert JSONL to a JSON array.")
    p.add_argument("input", help="Input .jsonl file")
    p.add_argument("--out", default=None, help="Output .json file (default: same name, .json extension)")
    p.add_argument("--pretty", action="store_true", help="Pretty-print the output JSON")
    args = p.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.out) if args.out else input_path.with_suffix(".json")

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    errors = convert(input_path, output_path, args.pretty)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
