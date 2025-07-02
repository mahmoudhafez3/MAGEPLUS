import argparse, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main(corpus, index):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    docs = [json.loads(l) for l in open(corpus, encoding="utf-8")]
    embs = model.encode([d["text"] for d in docs], convert_to_numpy=True, show_progress_bar=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, index)
    with open(index + ".meta", "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus}, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index", required=True)
    main(**vars(ap.parse_args()))
