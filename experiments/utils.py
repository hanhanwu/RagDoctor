import os
import re
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
    output_file = pdf_path.replace('pdfs/', 'raw_text/').replace('.pdf', '.txt')

    # if text file already exists, load and return it
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading cached text {output_file}: {e}")

    try:
        reader = pypdf.PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        output = "\n".join(parts).strip()
        with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
        return output
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return ""
    

def get_latest_files_per_company(pdfs_folder='pdfs/'):
    """
    Select only files matching COMPANYNAME_YEAR_ pattern and keep the latest year per company.
    """
    pdf_files = [f for f in os.listdir(pdfs_folder) if f.endswith('.pdf')]
    
    company_files = {}
    
    for filename in pdf_files:
        # Match pattern: COMPANYNAME_YEAR_
        match = re.match(r'^([A-Z_]+)_(\d{4})_', filename)
        if match:
            company = match.group(1)
            year = int(match.group(2))
            
            if company not in company_files:
                company_files[company] = (filename, year)
            else:
                # Keep the file with the latest year
                if year > company_files[company][1]:
                    company_files[company] = (filename, year)
    
    return {company: pdfs_folder + filename for company, (filename, year) in company_files.items()}


def process_item(item, selected_doc_names, loaded_pdf):
    """
    Process item in FinanceBench, only items from selected documents.
    """
    doc_name = item.get('doc_name', None)
    # skip items not in selected docs
    if doc_name not in selected_doc_names:
        return None
    
    doc_path = 'pdfs/' + doc_name
    doc_content = loaded_pdf[doc_path]
    metadata = {k: v for k, v in item.items() 
               if k not in ['question', 'answer']}
    return Document(text=doc_content, metadata=metadata)