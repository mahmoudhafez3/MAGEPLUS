# scripts/build_rtlcoder_corpus.py
import argparse, json, pandas as pd, pathlib

def main(csv, out):
    df = pd.read_csv(csv)
    # normalise column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # sanity-check
    if not {"instruction", "response"} <= set(df.columns):
        raise ValueError("CSV must contain 'Instruction' and 'Response' columns")

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            text = (
                f"### Instruction\n{row['instruction']}\n\n"
                f"### Response\n{row['response']}"
            )
            f.write(
                json.dumps({"id": i, "title": f"rtlcoder_{i}", "text": text},
                           ensure_ascii=False)
                + "\n"
            )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    main(**vars(p.parse_args()))
