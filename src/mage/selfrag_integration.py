# src/mage/selfrag_integration.py

import os
import json
from vllm import LLM, SamplingParams
from external.self-rag.retrieval_lm.passage_retrieval import Retriever  # :contentReference[oaicite:1]{index=1}

# 1) Initialize SELF-RAG’s Retriever
def build_retriever(config: dict) -> Retriever:
    """
    config expects keys:
      - model_name_or_path
      - passages_jsonl
      - embeddings_dir
      - n_docs
    """
    retr = Retriever({})
    retr.setup_retriever_demo(
        config["model_name_or_path"],
        config["passages_jsonl"],
        config["embeddings_dir"],
        n_docs=config.get("n_docs", 5),
        save_or_load_index=True,
    )
    return retr

# 2) Format prompt with retrieval + reflection tokens
def selfrag_generate(
    llm: LLM,
    retriever: Retriever,
    instruction: str,
    sampling_params: SamplingParams,
) -> str:
    # perform on-demand retrieval
    docs = retriever.search_document_demo(instruction, sampling_params.n_docs)
    context = "\n\n".join(f"<paragraph>{doc['text']}</paragraph>" for doc in docs)
    # SELF-RAG prompt format
    prompt = f"### Instruction:\n{instruction}\n\n[Retrieval]{context}"
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
