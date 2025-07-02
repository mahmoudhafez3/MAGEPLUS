from typing import Tuple
from .agent import TopAgent                # the original agent
from .self_rag_adapter import SelfRAGRetriever, _RETR_TOK


class RAGTopAgent(TopAgent):
    """
    TopAgent that prepends Self-RAG context to every *spec* string
    before delegating to the normal TopAgent logic.
    """

    def __init__(self, retriever: SelfRAGRetriever, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever

    # -------------------------------------------------------------- #
    # override ONE public method: run()
    # -------------------------------------------------------------- #
    def run(
        self,
        benchmark_type_name: str,
        task_id: str,
        spec: str,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
    ) -> Tuple[bool, str]:
        docs = self.retriever.search(spec, k=3)
        ctx = "\n\n".join(f"{_RETR_TOK}\n{d['text']}" for d in docs)
        spec_aug = f"{spec}\n\n{ctx}" if ctx else spec

        print(f"[RAGTopAgent] injected {len(docs)} docs into spec for {task_id}")
        return super().run(
            benchmark_type_name,
            task_id,
            spec_aug,                   # ← pass augmented spec downstream
            golden_tb_path,
            golden_rtl_blackbox_path,
        )
