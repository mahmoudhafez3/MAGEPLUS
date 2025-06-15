
import csv
import difflib
from typing import List, Dict

def load_corpus(csv_path: str) -> List[Dict[str,str]]:
    """Load RTL-Coder dataset as a list of {'instruction','response'} dicts."""
    corpus = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus.append({
                "instruction": row["Instruction"].strip(),
                "response": row["Response"].strip()
            })
    return corpus

def retrieve(query: str,
             corpus: List[Dict[str,str]],
             k: int = 3,
             cutoff: float = 0.1
) -> List[Dict[str,str]]:
    """
    Return top-k corpus entries whose 'instruction' best fuzzy-match `query`.
    """
    keys = [e["instruction"] for e in corpus]
    matches = difflib.get_close_matches(query, keys, n=k, cutoff=cutoff)
    return [e for e in corpus if e["instruction"] in matches]
