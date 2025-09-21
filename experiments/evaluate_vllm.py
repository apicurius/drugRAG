#!/usr/bin/env python3
"""
ULTRA-FAST DrugRAG Evaluation with vLLM (8 GPU Tensor Parallelism)
This uses vLLM server with tensor-parallel-size=8 for MAXIMUM SPEED

Usage:
1. Start vLLM server: ./start_vllm_server.sh
2. Run evaluation: python evaluate_vllm.py --test_size 1000 --architecture vllm_rag_a
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vllm_model import VLLMModel, VLLMQwenModel, VLLMLLAMA3Model
from src.evaluation.metrics import calculate_binary_classification_metrics, print_metrics_summary
from src.utils.embedding_client import create_embedding_client
from src.architectures.rag_format_a import FormatARAG
from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG
from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
from pinecone import Pinecone
import openai

# Suppress verbose HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Enhanced logging configuration
def setup_detailed_logging(architecture: str, test_size: int):
    """Setup comprehensive logging for evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"evaluation_logs/{timestamp}_{architecture}_{test_size}"
    os.makedirs(log_dir, exist_ok=True)

    # Main evaluation log
    main_handler = logging.FileHandler(f"{log_dir}/evaluation.log")
    main_handler.setLevel(logging.INFO)

    # Detailed interaction log (prompts, responses, etc.)
    detail_handler = logging.FileHandler(f"{log_dir}/detailed_interactions.log")
    detail_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    main_handler.setFormatter(simple_formatter)
    detail_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)

    # Configure loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(main_handler)
    root_logger.addHandler(detail_handler)
    root_logger.addHandler(console_handler)

    # Create specific loggers
    eval_logger = logging.getLogger('evaluation')
    prompt_logger = logging.getLogger('prompts')
    response_logger = logging.getLogger('responses')
    retrieval_logger = logging.getLogger('retrieval')
    analysis_logger = logging.getLogger('analysis')

    return {
        'log_dir': log_dir,
        'eval_logger': eval_logger,
        'prompt_logger': prompt_logger,
        'response_logger': response_logger,
        'retrieval_logger': retrieval_logger,
        'analysis_logger': analysis_logger
    }

logger = logging.getLogger(__name__)


class VLLMFormatARAG:
    """Format A RAG using vLLM for ULTRA-FAST inference"""

    def __init__(self, config_path: str = "../experiments/config.json"):
        """Initialize Format A RAG with vLLM"""
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

        # Initialize vLLM for reasoning - ULTRA FAST
        self.llm = VLLMModel(config_path)

        logger.info(f"✅ Format A RAG initialized with vLLM (8 GPU tensor parallelism)")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using robust client that handles 400 errors"""
        return self.embedding_client.get_embedding(text)

    def query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """Binary query with vLLM - ULTRA FAST"""
        retrieval_logger = logging.getLogger('retrieval')

        # Create query embedding
        query_text = f"{drug} {side_effect}"
        retrieval_logger.debug(f"RAG QUERY START: {drug} -> {side_effect}")
        retrieval_logger.debug(f"  Embedding query: '{query_text}'")

        query_embedding = self.get_embedding(query_text)

        if not query_embedding:
            retrieval_logger.error(f"  Failed to generate embedding for: {query_text}")
            return {'answer': 'ERROR', 'confidence': 0.0}

        # Query Pinecone
        retrieval_logger.debug(f"  Querying Pinecone with top_k=5, namespace={self.namespace}")
        results = self.index.query(
            vector=query_embedding,
            top_k=5,
            namespace=self.namespace,
            include_metadata=True
        )

        retrieval_logger.debug(f"  Retrieved {len(results.matches)} matches from Pinecone")

        # Build context
        context_parts = []
        for i, match in enumerate(results.matches):
            retrieval_logger.debug(f"    Match {i+1}: score={match.score:.3f}")
            if match.metadata and match.score > 0.5:
                drug_name = match.metadata.get('drug', '')
                drug_text = match.metadata.get('text', '')
                retrieval_logger.debug(f"      Drug: {drug_name}")
                retrieval_logger.debug(f"      Text: {drug_text[:100]}...")
                if drug_name and drug_text:
                    context_parts.append(f"Drug: {drug_name}\n{drug_text}")
            else:
                retrieval_logger.debug(f"      Skipped (score too low or no metadata)")

        context = "\n\n".join(context_parts[:3]) if context_parts else f"No specific data for {drug}"
        retrieval_logger.debug(f"  Final context length: {len(context)} characters")
        retrieval_logger.debug(f"  Context preview: {context[:200]}...")

        # Use vLLM for reasoning - THIS IS THE FAST PART
        prompt = f"""Based on the following medical information, answer whether {side_effect} is an adverse effect of {drug}.

Context from medical database:
{context}

Question: Is {side_effect} an adverse effect of {drug}?

Instructions: Review the information and answer YES or NO.

FINAL ANSWER: [YES or NO]"""

        # Call the enhanced vLLM query with detailed logging
        result = self.llm.query(drug, side_effect)

        # Add retrieval context to result
        result['retrieval_context'] = context[:500]  # Keep first 500 chars for logging
        result['num_retrieved_docs'] = len(context_parts)
        result['retrieval_scores'] = [match.score for match in results.matches[:3]]

        retrieval_logger.debug(f"RAG QUERY COMPLETE: {drug} -> {side_effect} = {result.get('answer', 'ERROR')}")
        return result

    def query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple queries with FULL BATCH OPTIMIZATION"""
        if not queries:
            return []

        retrieval_logger = logging.getLogger('retrieval')
        retrieval_logger.info(f"🚀 BATCH RAG PROCESSING: {len(queries)} queries with batch embeddings + retrieval")

        # Step 1: Batch embedding generation (MAJOR SPEEDUP)
        query_texts = [f"{q['drug']} {q['side_effect']}" for q in queries]
        retrieval_logger.info(f"📝 Generating {len(query_texts)} embeddings in batch...")

        # Use batch embedding processing instead of individual calls
        embeddings = self.embedding_client.get_embeddings_batch(
            query_texts,
            batch_size=20  # Conservative batch size for large datasets
        )

        # Step 2: Batch Pinecone retrieval
        retrieval_logger.info(f"🔍 Performing {len(embeddings)} Pinecone queries...")
        contexts = []

        for i, (query, embedding) in enumerate(zip(queries, embeddings)):
            if embedding is None:
                retrieval_logger.warning(f"No embedding for query {i+1}: {query['drug']} -> {query['side_effect']}")
                contexts.append(f"No specific data for {query['drug']}")
                continue

            # Individual Pinecone queries (Pinecone doesn't support batch queries)
            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=3,
                    namespace=self.namespace,
                    include_metadata=True
                )

                context_parts = []
                for match in results.matches:
                    if match.metadata and match.score > 0.5:
                        drug_name = match.metadata.get('drug', '')
                        drug_text = match.metadata.get('text', '')
                        if drug_name and drug_text:
                            context_parts.append(f"Drug: {drug_name}\n{drug_text}")

                context = "\n\n".join(context_parts[:2]) if context_parts else f"No specific data for {query['drug']}"
                contexts.append(context)

            except Exception as e:
                retrieval_logger.error(f"Pinecone query failed for {query['drug']}: {e}")
                contexts.append(f"No specific data for {query['drug']}")

        # Step 3: Prepare prompts for batch vLLM processing
        retrieval_logger.info(f"🧠 Preparing {len(queries)} prompts for batch vLLM inference...")
        prompts = []

        for query, context in zip(queries, contexts):
            # Shorter prompt to avoid truncation
            prompt = f"""Based on medical data: {context[:500]}

Question: Is {query['side_effect']} an adverse effect of {query['drug']}?

Answer YES or NO only.

FINAL ANSWER:"""
            prompts.append(prompt)

        # Step 4: Batch vLLM inference (OPTIMIZED)
        retrieval_logger.info(f"⚡ Running batch vLLM inference...")
        try:
            responses = self.llm.generate_batch(prompts, max_tokens=50)
        except Exception as e:
            retrieval_logger.error(f"Batch vLLM failed: {e}")
            # Fallback to query_batch method
            return self.llm.query_batch(queries)

        # Step 5: Parse results
        results = []
        for query, response, context in zip(queries, responses, contexts):
            response_upper = response.upper().strip()

            # Extract answer
            final_answer = 'UNKNOWN'
            confidence = 0.3

            if response_upper.startswith('YES') or 'FINAL ANSWER: YES' in response_upper:
                final_answer = 'YES'
                confidence = 0.95
            elif response_upper.startswith('NO') or 'FINAL ANSWER: NO' in response_upper:
                final_answer = 'NO'
                confidence = 0.95
            else:
                # Fallback counting
                yes_count = response_upper.count('YES')
                no_count = response_upper.count('NO')
                if yes_count > no_count and yes_count > 0:
                    final_answer = 'YES'
                    confidence = 0.8
                elif no_count > yes_count and no_count > 0:
                    final_answer = 'NO'
                    confidence = 0.8

            results.append({
                'answer': final_answer,
                'confidence': confidence,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'model': 'vllm-rag-batch-optimized',
                'reasoning': response[:100],
                'full_response': response,
                'retrieval_context': context[:200],
                'num_retrieved_docs': len([m for m in context.split('\n\n') if m.strip()]),
                'retrieval_scores': []  # Could be populated if needed
            })

        success_rate = sum(1 for r in results if r['answer'] != 'UNKNOWN') / len(results) * 100
        retrieval_logger.info(f"✅ BATCH RAG COMPLETE: {success_rate:.1f}% successful, {len(results)} total results")

        return results


class UltraFastEvaluator:
    """Ultra-fast evaluator using vLLM with 8 GPU tensor parallelism"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path

    def load_dataset(self, test_size: int = 100):
        """Load evaluation dataset"""
        df = pd.read_csv('../data/processed/evaluation_dataset.csv')

        # Get balanced sample
        if 'label' in df.columns:
            positive_samples = df[df['label'] == 1].sample(
                n=min(test_size//2, len(df[df['label'] == 1])),
                random_state=42
            )
            negative_samples = df[df['label'] == 0].sample(
                n=min(test_size//2, len(df[df['label'] == 0])),
                random_state=42
            )
            balanced_df = pd.concat([positive_samples, negative_samples])
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

            logger.info(f"✅ Balanced dataset: {len(positive_samples)} TRUE + {len(negative_samples)} FALSE")
            return balanced_df
        else:
            return df.sample(n=min(test_size, len(df)), random_state=42)

    def run_batch_evaluation(self, architecture: str, test_size: int):
        """Run evaluation with batch processing for MAXIMUM SPEED"""
        # Setup detailed logging
        log_config = setup_detailed_logging(architecture, test_size)
        eval_logger = log_config['eval_logger']
        analysis_logger = log_config['analysis_logger']

        logger.info("="*80)
        logger.info(f"VLLM ULTRA-FAST EVALUATION")
        logger.info(f"Architecture: {architecture}")
        logger.info(f"Test Size: {test_size}")
        logger.info(f"Using: 4 GPUs with Tensor Parallelism")
        logger.info(f"Detailed logs: {log_config['log_dir']}")
        logger.info("="*80)

        eval_logger.info(f"Starting evaluation: {architecture} with {test_size} samples")
        eval_logger.info(f"Log directory: {log_config['log_dir']}")

        # Load dataset
        dataset = self.load_dataset(test_size)

        # Initialize architecture
        if architecture == 'format_a_qwen':
            arch = FormatARAG(self.config_path, model="qwen")
        elif architecture == 'format_a_llama3':
            arch = FormatARAG(self.config_path, model="llama3")
        elif architecture == 'format_b_qwen':
            arch = FormatBRAG(self.config_path, model="qwen")
        elif architecture == 'format_b_llama3':
            arch = FormatBRAG(self.config_path, model="llama3")
        elif architecture == 'graphrag_qwen':
            arch = GraphRAG(self.config_path, model="qwen")
        elif architecture == 'graphrag_llama3':
            arch = GraphRAG(self.config_path, model="llama3")
        elif architecture == 'enhanced_format_b_qwen':
            arch = EnhancedRAGFormatB(self.config_path, model="qwen")
        elif architecture == 'enhanced_format_b_llama3':
            arch = EnhancedRAGFormatB(self.config_path, model="llama3")
        elif architecture == 'pure_llm_qwen':
            arch = VLLMQwenModel(self.config_path)
        elif architecture == 'pure_llm_llama3':
            arch = VLLMLLAMA3Model(self.config_path)
        else:
            logger.error(f"Unknown architecture: {architecture}")
            return

        # Start timing
        start_time = time.time()

        # Prepare batch queries
        queries = []
        for _, row in dataset.iterrows():
            queries.append({
                'drug': row['drug'],
                'side_effect': row['side_effect'],
                'label': row.get('label', None)
            })

        logger.info(f"\n🚀 Processing {len(queries)} queries with OPTIMIZED BATCH PROCESSING...")

        # Always use batch processing for maximum speed
        if hasattr(arch, 'query_batch'):
            # Optimized batch processing with embeddings + vLLM batching
            logger.info("   ✅ Using optimized batch processing (embeddings + vLLM)")
            batch_start = time.time()
            results = arch.query_batch(queries)
            batch_time = time.time() - batch_start
            logger.info(f"   ⚡ Batch processing completed in {batch_time:.2f}s ({len(queries)/batch_time:.1f} queries/sec)")
        else:
            # Fallback to individual processing (should not happen with optimized architectures)
            logger.warning("   ⚠️  Architecture doesn't support batch processing, using individual queries")
            results = []
            for q in tqdm(queries, desc="🔍 Processing queries individually", unit="query"):
                result = arch.query(q['drug'], q['side_effect'])
                results.append(result)

        # Calculate comprehensive metrics with detailed logging
        elapsed_time = time.time() - start_time

        # Prepare data for comprehensive metrics calculation
        y_true = []
        y_pred = []
        detailed_results = []

        eval_logger.info("="*60)
        eval_logger.info("GROUND TRUTH COMPARISON")
        eval_logger.info("="*60)

        for i, (query, result) in enumerate(zip(queries, results)):
            if query['label'] is not None:
                true_answer = 'YES' if query['label'] == 1 else 'NO'
                predicted = result.get('answer', 'UNKNOWN')
                is_correct = predicted == true_answer

                # Collect for comprehensive metrics
                y_true.append(true_answer)
                y_pred.append(predicted)

                # Log each comparison
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                eval_logger.info(f"Sample {i+1:3d}: {query['drug']:<20} -> {query['side_effect']:<25}")
                eval_logger.info(f"           Ground Truth: {true_answer:<8} | Predicted: {predicted:<8} | {status}")

                if not is_correct:
                    # Log misclassifications with full details
                    analysis_logger.warning(f"MISCLASSIFICATION #{i+1}:")
                    analysis_logger.warning(f"  Drug: {query['drug']}")
                    analysis_logger.warning(f"  Side Effect: {query['side_effect']}")
                    analysis_logger.warning(f"  Ground Truth: {true_answer}")
                    analysis_logger.warning(f"  Predicted: {predicted}")
                    analysis_logger.warning(f"  Confidence: {result.get('confidence', 'N/A')}")
                    if 'full_response' in result:
                        analysis_logger.warning(f"  Full Response: {result['full_response']}")

                # Collect detailed results for CSV
                detailed_results.append({
                    'sample_id': i+1,
                    'drug': query['drug'],
                    'side_effect': query['side_effect'],
                    'ground_truth': true_answer,
                    'predicted': predicted,
                    'is_correct': is_correct,
                    'confidence': result.get('confidence', 0.0),
                    'full_response': result.get('full_response', result.get('reasoning', '')),
                    'model': result.get('model', 'unknown'),
                    'architecture': architecture,
                    'retrieval_context': result.get('retrieval_context', ''),
                    'num_retrieved_docs': result.get('num_retrieved_docs', 0),
                    'retrieval_scores': str(result.get('retrieval_scores', [])),
                    'elapsed_time_s': elapsed_time / len(queries)  # Average time per query
                })

        # Calculate comprehensive binary classification metrics
        if y_true and y_pred:
            comprehensive_metrics = calculate_binary_classification_metrics(y_true, y_pred)

            # Legacy accuracy for backward compatibility
            accuracy = comprehensive_metrics['accuracy'] * 100
        else:
            comprehensive_metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'sensitivity': 0.0,
                'specificity': 0.0, 'f1_score': 0.0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
            }
            accuracy = 0.0

        # Save detailed results to CSV
        detailed_csv_path = f"{log_config['log_dir']}/detailed_results.csv"
        import pandas as pd
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(detailed_csv_path, index=False)
        eval_logger.info(f"Detailed results saved to: {detailed_csv_path}")

        eval_logger.info("="*60)
        eval_logger.info("COMPREHENSIVE BINARY CLASSIFICATION METRICS")
        eval_logger.info("="*60)
        eval_logger.info(f"Accuracy:    {comprehensive_metrics['accuracy']:.4f} ({accuracy:.2f}%)")
        eval_logger.info(f"F1 Score:    {comprehensive_metrics['f1_score']:.4f}")
        eval_logger.info(f"Precision:   {comprehensive_metrics['precision']:.4f}")
        eval_logger.info(f"Sensitivity: {comprehensive_metrics['sensitivity']:.4f}")
        eval_logger.info(f"Specificity: {comprehensive_metrics['specificity']:.4f}")
        eval_logger.info(f"")
        eval_logger.info(f"Confusion Matrix:")
        eval_logger.info(f"TP: {comprehensive_metrics['tp']}, TN: {comprehensive_metrics['tn']}")
        eval_logger.info(f"FP: {comprehensive_metrics['fp']}, FN: {comprehensive_metrics['fn']}")
        eval_logger.info("="*60)

        # Count predictions
        predictions = [r.get('answer', 'UNKNOWN') for r in results]
        pred_counts = pd.Series(predictions).value_counts().to_dict()

        # Print comprehensive results
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE BINARY CLASSIFICATION RESULTS")
        logger.info("="*80)
        logger.info(f"Accuracy:    {comprehensive_metrics['accuracy']:.4f} ({accuracy:.2f}%)")
        logger.info(f"F1 Score:    {comprehensive_metrics['f1_score']:.4f}")
        logger.info(f"Precision:   {comprehensive_metrics['precision']:.4f}")
        logger.info(f"Sensitivity: {comprehensive_metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {comprehensive_metrics['specificity']:.4f}")
        logger.info(f"")
        logger.info(f"Confusion Matrix: TP={comprehensive_metrics['tp']}, TN={comprehensive_metrics['tn']}, FP={comprehensive_metrics['fp']}, FN={comprehensive_metrics['fn']}")
        logger.info(f"Prediction Distribution: {pred_counts}")
        logger.info(f"Time: {elapsed_time:.2f} seconds")
        logger.info(f"Throughput: {len(queries)/elapsed_time:.2f} queries/second")
        logger.info(f"")
        logger.info(f"🚀 SPEEDUP vs single GPU Qwen: ~{(len(queries)/elapsed_time)/0.5:.1f}x")
        logger.info(f"   (Baseline single GPU Qwen: ~0.5 queries/second)")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_vllm_{architecture}_{test_size}_{timestamp}.json"

        output_results = []
        for query, result in zip(queries, results):
            output_results.append({
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'true_label': query['label'],
                'predicted': result.get('answer'),
                'confidence': result.get('confidence')
            })

        with open(results_file, 'w') as f:
            json.dump({
                'configuration': {
                    'architecture': architecture,
                    'test_size': test_size,
                    'gpu_config': '8x RTX A4000 with tensor parallelism'
                },
                'metrics': {
                    'accuracy': comprehensive_metrics['accuracy'],
                    'accuracy_percent': accuracy,
                    'f1_score': comprehensive_metrics['f1_score'],
                    'precision': comprehensive_metrics['precision'],
                    'sensitivity': comprehensive_metrics['sensitivity'],
                    'specificity': comprehensive_metrics['specificity'],
                    'tp': comprehensive_metrics['tp'],
                    'tn': comprehensive_metrics['tn'],
                    'fp': comprehensive_metrics['fp'],
                    'fn': comprehensive_metrics['fn'],
                    'prediction_distribution': pred_counts,
                    'elapsed_time': elapsed_time,
                    'throughput': len(queries)/elapsed_time
                },
                'detailed_results': output_results
            }, f, indent=2)

        logger.info(f"\n✅ Results saved to {results_file}")

        return {
            'accuracy': comprehensive_metrics['accuracy'],
            'accuracy_percent': accuracy,
            'f1_score': comprehensive_metrics['f1_score'],
            'precision': comprehensive_metrics['precision'],
            'sensitivity': comprehensive_metrics['sensitivity'],
            'specificity': comprehensive_metrics['specificity'],
            'throughput': len(queries)/elapsed_time,
            'time': elapsed_time
        }


def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast DrugRAG Evaluation with vLLM')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--architecture', type=str, default='vllm_rag_a',
                       choices=[
                           'format_a_qwen', 'format_a_llama3',
                           'format_b_qwen', 'format_b_llama3',
                           'graphrag_qwen', 'graphrag_llama3',
                           'enhanced_format_b_qwen', 'enhanced_format_b_llama3',
                           'pure_llm_qwen', 'pure_llm_llama3'
                       ],
                       help='Architecture to test')

    args = parser.parse_args()

    logger.info("\n" + "🚀"*40)
    logger.info("STARTING VLLM ULTRA-FAST EVALUATION")
    logger.info("Make sure vLLM server is running: ./start_vllm_server.sh")
    logger.info("🚀"*40 + "\n")

    evaluator = UltraFastEvaluator()
    evaluator.run_batch_evaluation(
        architecture=args.architecture,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()