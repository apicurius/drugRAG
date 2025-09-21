#!/usr/bin/env python3
"""
Format A RAG Implementation - vLLM ONLY
Retrieval: Pinecone vector store
Reasoning: vLLM (Qwen or LLAMA3)
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List
from pinecone import Pinecone
import openai
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.embedding_client import create_embedding_client
from src.utils.token_manager import create_token_manager
from src.utils.binary_parser import parse_binary_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatARAG:
    """Format A: Drug -> [side effects] with vLLM reasoning"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize Format A RAG with vLLM

        Args:
            config_path: Path to configuration file
            model: vLLM model ("qwen" or "llama3")
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Pinecone for retrieval
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.namespace = "drug-side-effects-formatA"

        # Initialize robust embedding client
        self.embedding_client = create_embedding_client(
            api_key=self.config['openai_api_key'],
            model="text-embedding-ada-002"
        )

        # Initialize vLLM for reasoning
        self.model = model
        if model == "qwen":
            from src.models.vllm_model import VLLMQwenModel
            self.llm = VLLMQwenModel(config_path)
        elif model == "llama3":
            from src.models.vllm_model import VLLMLLAMA3Model
            self.llm = VLLMLLAMA3Model(config_path)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'qwen' or 'llama3'")

        # Initialize token manager for context truncation
        self.token_manager = create_token_manager(model_type=model)

        logger.info(f"✅ Format A RAG initialized with {model} via vLLM and token management")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using robust client that handles 400 errors"""
        return self.embedding_client.get_embedding(text)

    def query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """Binary query with vLLM reasoning"""
        # Retrieve from Pinecone
        query_text = f"{drug} {side_effect}"
        query_embedding = self.get_embedding(query_text)

        if not query_embedding:
            return {'answer': 'ERROR', 'confidence': 0.0}

        results = self.index.query(
            vector=query_embedding,
            top_k=10,  # Increased for better context
            namespace=self.namespace,
            include_metadata=True
        )

        # Build context from retrieved documents
        context_documents = []
        for match in results.matches:
            if match.metadata and match.score > 0.5:  # Standard threshold for recall
                drug_name = match.metadata.get('drug', '')
                drug_text = match.metadata.get('text', '')
                if drug_name and drug_text:
                    context_documents.append(f"Drug: {drug_name}\n{drug_text}")

        # Create base prompt template for token counting
        base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {side_effect} an adverse effect of {drug}?

### RAG Results:

{{context}}"""

        # Use token manager to intelligently truncate context
        if context_documents:
            context, docs_included = self.token_manager.truncate_context_documents(context_documents, base_prompt)
            if docs_included < len(context_documents):
                logger.debug(f"Format A token limit: included {docs_included}/{len(context_documents)} documents for {drug}-{side_effect}")
        else:
            context = f"No data found for {drug}"

        # Build final prompt with truncated context
        prompt = base_prompt.format(context=context)

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            response = self.llm.generate_response(prompt, max_tokens=100, temperature=0.1)

            # Use standardized binary parser (notebook-compatible)
            answer = parse_binary_response(response)

            return {
                'answer': answer,
                'confidence': 0.9 if answer != 'UNKNOWN' else 0.3,
                'drug': drug,
                'side_effect': side_effect,
                'format': 'A',
                'model': f'vllm_{self.model}',
                'reasoning': response[:200]
            }

        except Exception as e:
            logger.error(f"vLLM reasoning error: {e}")
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'format': 'A',
                'model': f'vllm_{self.model}'
            }

    def query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple queries with FULL BATCH OPTIMIZATION
        This provides dramatic speedup over individual query processing
        """
        if not queries:
            return []

        logger.info(f"🚀 FORMAT A BATCH PROCESSING: {len(queries)} queries with optimized embeddings + retrieval + vLLM")

        # Step 1: Batch embedding generation (MAJOR SPEEDUP)
        query_texts = [f"{q['drug']} {q['side_effect']}" for q in queries]
        logger.info(f"📝 Generating {len(query_texts)} embeddings in batch...")

        embeddings = self.embedding_client.get_embeddings_batch(
            query_texts,
            batch_size=20  # Conservative batch size for large datasets
        )

        # Step 2: Concurrent Pinecone retrieval with progress tracking
        logger.info(f"🔍 Performing {len(embeddings)} Pinecone queries (concurrent)...")
        all_contexts = [None] * len(queries)  # Pre-allocate to maintain order

        def process_single_query(idx_query_embedding):
            """Process a single Pinecone query"""
            idx, query, embedding = idx_query_embedding

            if embedding is None:
                return idx, f"No data found for {query['drug']}"

            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=10,  # Increased for better context
                    namespace=self.namespace,
                    include_metadata=True
                )

                context_documents = []
                for match in results.matches:
                    if match.metadata and match.score > 0.5:  # Standard threshold for recall
                        drug_name = match.metadata.get('drug', '')
                        drug_text = match.metadata.get('text', '')
                        if drug_name and drug_text:
                            context_documents.append(f"Drug: {drug_name}\n{drug_text}")

                # Use token manager for context truncation
                base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{{context}}"""

                if context_documents:
                    context, docs_included = self.token_manager.truncate_context_documents(context_documents, base_prompt)
                else:
                    context = f"No data found for {query['drug']}"

                return idx, context

            except Exception as e:
                logger.error(f"Pinecone query failed for {query['drug']}: {e}")
                return idx, f"No data found for {query['drug']}"

        # Use ThreadPoolExecutor for concurrent Pinecone queries
        max_workers = min(10, len(queries))  # Limit concurrent connections
        query_data = [(i, query, embedding) for i, (query, embedding) in enumerate(zip(queries, embeddings))]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_idx = {executor.submit(process_single_query, qd): qd[0] for qd in query_data}

            # Collect results with progress bar
            with tqdm(total=len(queries), desc="🔍 Pinecone", unit="query", ncols=100) as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, context = future.result(timeout=30)
                        all_contexts[idx] = context
                        pbar.update(1)
                    except Exception as e:
                        idx = future_to_idx[future]
                        logger.error(f"Query {idx} failed: {e}")
                        all_contexts[idx] = f"No data found for {queries[idx]['drug']}"
                        pbar.update(1)

        # Ensure all contexts are filled (safety check)
        for i, context in enumerate(all_contexts):
            if context is None:
                all_contexts[i] = f"No data found for {queries[i]['drug']}"

        # Step 3: Prepare prompts for batch vLLM processing
        logger.info(f"🧠 Preparing {len(queries)} prompts for batch vLLM inference...")
        prompts = []

        for query, context in zip(queries, all_contexts):
            prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{context}"""
            prompts.append(prompt)

        # Step 4: Batch vLLM inference (OPTIMIZED)
        logger.info(f"⚡ Running batch vLLM inference...")
        try:
            # Use temperature=0.1 for RAG deterministic outputs
            responses = self.llm.generate_batch(prompts, max_tokens=100, temperature=0.1)
        except Exception as e:
            logger.error(f"Batch vLLM failed: {e}")
            # Fallback to individual processing
            return [self.query(q['drug'], q['side_effect']) for q in queries]

        # Step 5: Parse results
        results = []
        for query, response, context in zip(queries, responses, all_contexts):
            # Use standardized binary parser (notebook-compatible)
            answer = parse_binary_response(response)

            results.append({
                'answer': answer,
                'confidence': 0.9 if answer != 'UNKNOWN' else 0.3,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'format': 'A',
                'model': f'vllm_{self.model}_batch_optimized',
                'reasoning': response[:200],
                'full_response': response,
                'retrieval_context': context[:200]
            })

        success_rate = sum(1 for r in results if r['answer'] != 'UNKNOWN') / len(results) * 100
        logger.info(f"✅ FORMAT A BATCH COMPLETE: {success_rate:.1f}% successful, {len(results)} total results")

        return results