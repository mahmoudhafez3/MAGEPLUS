import json, faiss
from sentence_transformers import SentenceTransformer
from mage.agent import TopAgent

_RETR_TOK = "[Retrieval]"

class SelfRAGRetriever:
    def __init__(self, index_path: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)
        meta = json.load(open(index_path + ".meta", encoding="utf-8"))
        self.corpus = [json.loads(l) for l in open(meta["corpus"], encoding="utf-8")]

    def search(self, query: str, k: int = 3):
        emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(emb, k)
        return [self.corpus[i] for i in I[0] if i < len(self.corpus)]

class SelfRAGAgent(TopAgent):
    def __init__(self, retriever: SelfRAGRetriever, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever

    def _augment_prompt(self, prompt: str) -> str:
        retrieved = self.retriever.search(prompt)
        ctx = "\n\n".join(f"[Retrieval]\n{doc['text']}" for doc in retrieved)
        print(f"[SelfRAGAgent] Injecting {len(retrieved)} retrieved docs into prompt.")
        return f"{prompt}\n\n{ctx}"

    def generate(self, query: str, *args, **kwargs):
        return super().generate(self._augment_prompt(query), *args, **kwargs)

