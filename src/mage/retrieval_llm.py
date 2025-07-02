# """
# RetrievalLLM – a thin wrapper that prepends Self-RAG context
# to every prompt before forwarding the request to the real LLM.
# """

# from typing import Any, Dict, List
# import logging, sys
# from llama_index.core.llms import ChatMessage, LLM

# from .self_rag_adapter import SelfRAGRetriever, _RETR_TOK


# class RetrievalLLM(LLM):
#     """A decorator that injects top-k retrieved docs into every prompt."""

#     def __init__(self, base_llm: LLM, retriever: SelfRAGRetriever, k: int = 3):
#         self._base_llm = base_llm
#         self._retriever = retriever
#         self._k = k

#     # --------------------------------------------------------------------- #
#     # helper
#     # --------------------------------------------------------------------- #
#     def _inject(self, prompt: str) -> str:
#         docs = self._retriever.search(prompt, k=self._k)
#         ctx = "\n\n".join(f"{_RETR_TOK}\n{d['text']}" for d in docs)
#         #print(f"[RetrievalLLM] injected {len(docs)} docs")
#         msg = f"[RetrievalLLM] injected {len(docs)} docs"
#         print(msg, file=sys.__stdout__)      # console
#         logging.getLogger(__name__).info(msg)  # log file
#         return f"{prompt}\n\n{ctx}" if ctx else prompt

#     # --------------------------------------------------------------------- #
#     #  synchronous completions
#     # --------------------------------------------------------------------- #
#     def complete(self, prompt: str, **kwargs: Any):
#         return self._base_llm.complete(self._inject(prompt), **kwargs)

#     def stream_complete(self, prompt: str, **kwargs: Any):
#         return self._base_llm.stream_complete(self._inject(prompt), **kwargs)

#     # --------------------------------------------------------------------- #
#     #  asynchronous completions
#     # --------------------------------------------------------------------- #
#     async def acomplete(self, prompt: str, **kwargs: Any):
#         return await self._base_llm.acomplete(self._inject(prompt), **kwargs)

#     async def astream_complete(self, prompt: str, **kwargs: Any):
#         async for chunk in self._base_llm.astream_complete(
#             self._inject(prompt), **kwargs
#         ):
#             yield chunk

#     # --------------------------------------------------------------------- #
#     #  synchronous chat
#     # --------------------------------------------------------------------- #
#     def chat(self, messages: List[ChatMessage], **kwargs: Any):
#         # inject into *last* user message
#         new_messages = messages.copy()
#         for i in range(len(new_messages) - 1, -1, -1):
#             if new_messages[i].role == "user":
#                 new_messages[i].content = self._inject(new_messages[i].content)
#                 break
#         return self._base_llm.chat(new_messages, **kwargs)

#     def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
#         new_messages = messages.copy()
#         for i in range(len(new_messages) - 1, -1, -1):
#             if new_messages[i].role == "user":
#                 new_messages[i].content = self._inject(new_messages[i].content)
#                 break
#         return self._base_llm.stream_chat(new_messages, **kwargs)

#     # --------------------------------------------------------------------- #
#     #  asynchronous chat
#     # --------------------------------------------------------------------- #
#     async def achat(self, messages: List[ChatMessage], **kwargs: Any):
#         new_messages = messages.copy()
#         for i in range(len(new_messages) - 1, -1, -1):
#             if new_messages[i].role == "user":
#                 new_messages[i].content = self._inject(new_messages[i].content)
#                 break
#         return await self._base_llm.achat(new_messages, **kwargs)

#     async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
#         new_messages = messages.copy()
#         for i in range(len(new_messages) - 1, -1, -1):
#             if new_messages[i].role == "user":
#                 new_messages[i].content = self._inject(new_messages[i].content)
#                 break
#         async for chunk in self._base_llm.astream_chat(new_messages, **kwargs):
#             yield chunk

#     # --------------------------------------------------------------------- #
#     #  metadata passthrough
#     # --------------------------------------------------------------------- #
#     @property
#     def metadata(self) -> Dict[str, Any]:  # type: ignore[override]
#         return self._base_llm.metadata

#     # --------------------------------------------------------------------- #
#     #  fallback for everything else
#     # --------------------------------------------------------------------- #
#     def __getattr__(self, item):
#         # delegate all other attributes/methods to underlying llm
#         return getattr(self._base_llm, item)


from typing import Any, Dict, List
import sys, logging

from llama_index.core.llms import ChatMessage, LLM

from .self_rag_adapter import SelfRAGRetriever, _RETR_TOK


class RetrievalLLM(LLM):
    def __init__(self, base_llm: LLM, retriever: SelfRAGRetriever, k: int = 3):
        self._base_llm = base_llm
        self._retriever = retriever
        self._k = k

    # ------------------------------------------------------------------ #
    # helper
    # ------------------------------------------------------------------ #
    def _inject(self, prompt: str) -> str:
        docs = self._retriever.search(prompt, k=self._k)
        ctx = "\n\n".join(f"{_RETR_TOK}\n{d['text']}" for d in docs)
        msg = f"[RetrievalLLM] injected {len(docs)} docs"
        # print to both redirected log and console
        print(msg)
        print(msg, file=sys.__stdout__)
        # also print short preview of each doc
        for i, d in enumerate(docs, 1):
            preview = d["text"].replace("\n", " ")[:200]
            line = f"[RetrievalLLM] DOC {i}: {preview}..."
            print(line)
            print(line, file=sys.__stdout__)
        logging.getLogger(__name__).info(msg)
        return f"{prompt}\n\n{ctx}" if ctx else prompt

    # ------------------------------------------------------------------ #
    # complete
    # ------------------------------------------------------------------ #
    def complete(self, prompt: str, **kwargs: Any):
        return self._base_llm.complete(self._inject(prompt), **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any):
        return self._base_llm.stream_complete(self._inject(prompt), **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any):
        return await self._base_llm.acomplete(self._inject(prompt), **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any):
        async for chunk in self._base_llm.astream_complete(self._inject(prompt), **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    # chat
    # ------------------------------------------------------------------ #
    def chat(self, messages: List[ChatMessage], **kwargs: Any):
        msgs = messages.copy()
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = self._inject(msgs[i].content)
                break
        return self._base_llm.chat(msgs, **kwargs)

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        msgs = messages.copy()
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = self._inject(msgs[i].content)
                break
        return self._base_llm.stream_chat(msgs, **kwargs)

    async def achat(self, messages: List[ChatMessage], **kwargs: Any):
        msgs = messages.copy()
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = self._inject(msgs[i].content)
                break
        return await self._base_llm.achat(msgs, **kwargs)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        msgs = messages.copy()
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = self._inject(msgs[i].content)
                break
        async for chunk in self._base_llm.astream_chat(msgs, **kwargs):
            yield chunk

    # ------------------------------------------------------------------ #
    # passthrough metadata
    # ------------------------------------------------------------------ #
    @property
    def metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        return self._base_llm.metadata

    # ------------------------------------------------------------------ #
    # delegate everything else
    # ------------------------------------------------------------------ #
    def __getattr__(self, item):
        return getattr(self._base_llm, item)
    

    @property
    def base_llm(self):
        return self._base_llm
