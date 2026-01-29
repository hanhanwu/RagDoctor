import os
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import numpy as np
from datasets import load_dataset
from pathlib import Path
import pypdf
import pdfplumber

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    PromptTemplate
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.schema import NodeWithScore

# Embeddings & LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Advanced retrieval components
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import QueryFusionRetriever

# Query expansion
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
def load_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        output = "\n".join(parts).strip()
        output_file = pdf_path.replace('pdfs/', 'raw_text/').replace('.pdf', '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
        return output
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return ""