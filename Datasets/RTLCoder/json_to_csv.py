import json
import csv
import sys
from pathlib import Path

def jsonl_to_csv(jsonl_file, csv_file):
    data = []
    for line in Path(jsonl_file).open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON line: {line[:50]}…")
    if not data:
        print("❌ No valid JSON records found.")
        return

    # Flatten if nested (keep top-level keys only)
    keys = set()
    for o in data:
        keys |= o.keys()
    fieldnames = sorted(keys)

    with Path(csv_file).open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for o in data:
            writer.writerow({k: o.get(k, "") for k in fieldnames})

    print(f"✅ Wrote {len(data)} rows to CSV: {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv.py input.jsonl output.csv")
    else:
        jsonl_to_csv(sys.argv[1], sys.argv[2])
