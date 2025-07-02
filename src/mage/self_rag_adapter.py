# import json, faiss
# from sentence_transformers import SentenceTransformer
# from mage.agent import TopAgent

# _RETR_TOK = "[Retrieval]"

# class SelfRAGRetriever:
#     def __init__(self, index_path: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.index = faiss.read_index(index_path)
#         self.model = SentenceTransformer(model_name)
#         meta = json.load(open(index_path + ".meta", encoding="utf-8"))
#         self.corpus = [json.loads(l) for l in open(meta["corpus"], encoding="utf-8")]

#     def search(self, query: str, k: int = 3):
#         emb = self.model.encode([query], convert_to_numpy=True)
#         D, I = self.index.search(emb, k)
#         return [self.corpus[i] for i in I[0] if i < len(self.corpus)]

# class SelfRAGAgent(TopAgent):
#     def __init__(self, retriever: SelfRAGRetriever, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.retriever = retriever

#     def _augment_prompt(self, prompt: str) -> str:
#         retrieved = self.retriever.search(prompt)
#         ctx = "\n\n".join(f"[Retrieval]\n{doc['text']}" for doc in retrieved)
#         print(f"[SelfRAGAgent] Injecting {len(retrieved)} retrieved docs into prompt.")
#         return f"{prompt}\n\n{ctx}"

#     def generate(self, query: str, *args, **kwargs):
#         return super().generate(self._augment_prompt(query), *args, **kwargs)



import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_RETR_TOK = "[Retrieval]"


class SelfRAGRetriever:
    """
    Tiny wrapper around FAISS + Sentence-Transformers.
    """

    def __init__(
        self,
        index_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(index_path)

        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)

        meta_path = index_path.with_suffix(index_path.suffix + ".meta")
        meta = json.load(meta_path.open("r", encoding="utf-8"))
        self.corpus = [json.loads(l) for l in Path(meta["corpus"]).open(encoding="utf-8")]

        logger.info(
            f"SelfRAGRetriever loaded: {len(self.corpus)} docs, index.ntotal={self.index.ntotal}"
        )

    # ------------------------------------------------------------------ #
    # SEARCH
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 3):
        """
        Return *k* docs as list[dict].
        Extra logging will print IDs and first 80 chars so you can see
        exactly what is coming back.
        """
        emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(emb, k)

        hits = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.corpus):
                continue
            doc = self.corpus[idx]
            hits.append(doc)
            preview = doc["text"].replace("\n", " ")[:80]
            logger.info(
                f"[SelfRAGRetriever] hit#{rank+1} id={doc.get('id', idx)}  "
                f"D={D[0][rank]:.4f}  {preview}..."
            )

        return hits


# ---------------------------------------------------------------------- #
#  Convenience:  Inject docs straight into a prompt (for debugging only)
# ---------------------------------------------------------------------- #
def augment_prompt(prompt: str, retriever: SelfRAGRetriever, k: int = 3) -> str:
    docs = retriever.search(prompt, k=k)
    ctx = "\n\n".join(f"{_RETR_TOK}\n{d['text']}" for d in docs)
    merged = f"{prompt}\n\n{ctx}" if ctx else prompt
    border = "-" * 88
    logger.info("%s\nAUGMENTED PROMPT:\n%s\n%s", border, merged, border)
    print(f"{border}\nAUGMENTED PROMPT:\n{merged}\n{border}", flush=True)
    return merged
