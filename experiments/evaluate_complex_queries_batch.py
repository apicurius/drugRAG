#!/usr/bin/env python3
"""
Batch-Enabled Complex Query Evaluation with GPU Acceleration
Leverages 4 GPUs with tensor parallelism for maximum throughput
"""

import argparse
import json
import logging
import pandas as pd
import ast
from datetime import datetime
import sys
import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimal batch sizes for 4 GPUs
BATCH_SIZE = 50  # Queries per batch
MAX_WORKERS = 8  # Concurrent threads for parallel processing


class BatchComplexQueryEvaluator:
    """Batch-enabled evaluator for complex drug queries with GPU acceleration"""

    def __init__(self, config_path: str = "config.json"):
        """Initialize batch evaluator with configuration"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Query dataset mappings
        self.query_datasets = {
            'organ_specific': '../data/processed/comprehensive_organ_queries.csv',
            'severity_filtered': '../data/processed/comprehensive_severity_queries.csv',
            'drug_comparison': '../data/processed/comprehensive_comparison_queries.csv',
            'reverse_lookup': '../data/processed/reverse_queries.csv',
            'combination': '../data/processed/comprehensive_combination_queries.csv'
        }

        self.architecture_cache = {}
        logger.info(f"✅ Batch evaluator initialized with batch_size={BATCH_SIZE}, max_workers={MAX_WORKERS}")

    def _initialize_architecture(self, architecture: str, model: str):
        """Initialize and cache architecture instance"""
        cache_key = f"{architecture}_{model}"
        if cache_key in self.architecture_cache:
            return self.architecture_cache[cache_key]

        # Import and initialize based on architecture
        if architecture.startswith('format_a'):
            from src.architectures.rag_format_a import RAGFormatA
            arch = RAGFormatA(self.config_path, model)
        elif architecture.startswith('format_b'):
            from src.architectures.rag_format_b import RAGFormatB
            arch = RAGFormatB(self.config_path, model)
        elif architecture.startswith('graphrag'):
            from src.architectures.graphrag import GraphRAG
            arch = GraphRAG(self.config_path, model)
        elif architecture.startswith('enhanced_format_b'):
            from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
            arch = EnhancedRAGFormatB(self.config_path, model)
        elif architecture.startswith('enhanced_graphrag'):
            from src.architectures.enhanced_graphrag import EnhancedGraphRAG
            arch = EnhancedGraphRAG(self.config_path, model)
        elif architecture.startswith('advanced_rag_format_b'):
            from src.architectures.advanced_rag_format_b import AdvancedRAGFormatB
            arch = AdvancedRAGFormatB(self.config_path, model)
        elif architecture == 'pure':
            from src.models.vllm_model import VLLMQwenModel, VLLMLLAMA3Model
            arch = VLLMQwenModel(self.config_path) if model == 'qwen' else VLLMLLAMA3Model(self.config_path)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.architecture_cache[cache_key] = arch
        return arch

    def process_batch_organ_specific(self, arch_instance, batch_df, architecture):
        """Process a batch of organ-specific queries in parallel"""
        batch_results = []
        queries_data = []

        # Prepare batch data
        for idx, row in batch_df.iterrows():
            queries_data.append({
                'idx': idx,
                'query': row['query'],
                'drug': row['drug'],
                'organ_filter': row['organ_filter'],
                'ground_truth': ast.literal_eval(row['ground_truth']),
                'expected_count': row['num_results']
            })

        # Process based on architecture capabilities
        if hasattr(arch_instance, 'batch_complex_query_organ_specific'):
            # Architecture supports native batch processing
            drugs = [q['drug'] for q in queries_data]
            organs = [q['organ_filter'] for q in queries_data]

            start_time = time.time()
            batch_responses = arch_instance.batch_complex_query_organ_specific(drugs, organs)
            batch_time = time.time() - start_time

            for query_data, response in zip(queries_data, batch_responses):
                batch_results.append(self._process_result(query_data, response, batch_time / len(queries_data)))
        else:
            # Fallback to parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []

                for query_data in queries_data:
                    future = executor.submit(
                        self._process_single_organ_query,
                        arch_instance, query_data, architecture
                    )
                    futures.append((future, query_data))

                for future, query_data in tqdm(futures, desc="Processing batch", leave=False):
                    try:
                        result = future.result(timeout=30)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing query {query_data['idx']}: {e}")
                        batch_results.append(self._error_result(query_data))

        return batch_results

    def _process_single_organ_query(self, arch_instance, query_data, architecture):
        """Process a single organ-specific query"""
        start_time = time.time()

        try:
            if 'graphrag' in architecture:
                result = arch_instance.complex_query_organ_specific(
                    query_data['drug'], query_data['organ_filter']
                )
                predicted_effects = result.get('side_effects_found', [])
                confidence = 1.0 if result.get('count', 0) > 0 else 0.0
            elif 'enhanced_format_b' in architecture:
                result = arch_instance.organ_specific_query(
                    query_data['drug'], query_data['organ_filter']
                )
                predicted_effects = result.get('side_effects', [])
                confidence = result.get('confidence', 0.0)
            else:
                # Standard RAG approaches
                combined_query = f"What {query_data['organ_filter']} side effects does {query_data['drug']} cause?"
                result = arch_instance.query(combined_query, "")
                predicted_effects = self._extract_effects_from_response(result.get('llm_response', ''))
                confidence = result.get('confidence', 0.0)

            elapsed_time = time.time() - start_time

            return self._process_result(query_data, {
                'predicted_effects': predicted_effects,
                'confidence': confidence
            }, elapsed_time)

        except Exception as e:
            logger.error(f"Error in single query: {e}")
            return self._error_result(query_data)

    def _process_result(self, query_data, response, elapsed_time):
        """Process and evaluate a query result"""
        predicted_effects = response.get('predicted_effects', [])
        ground_truth = query_data['ground_truth']

        # Calculate metrics
        precision, recall, f1 = self._calculate_set_metrics(predicted_effects, ground_truth)
        query_success = self._evaluate_query_success(predicted_effects, ground_truth)

        return {
            'query': query_data['query'],
            'drug': query_data['drug'],
            'predicted_count': len(predicted_effects),
            'expected_count': query_data['expected_count'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'query_success': query_success,
            'elapsed_time': elapsed_time,
            'confidence': response.get('confidence', 0.0)
        }

    def _error_result(self, query_data):
        """Create an error result entry"""
        return {
            'query': query_data['query'],
            'drug': query_data['drug'],
            'predicted_count': 0,
            'expected_count': query_data['expected_count'],
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'query_success': 0,
            'elapsed_time': 0.0,
            'confidence': 0.0
        }

    def evaluate_organ_specific_batch(self, architecture: str, model: str = "qwen", limit: int = None):
        """Evaluate organ-specific queries using batch processing"""
        dataset_path = self.query_datasets['organ_specific']
        logger.info(f"🎯 Loading organ-specific queries from {dataset_path}")

        df = pd.read_csv(dataset_path)
        if limit:
            df = df.head(limit)

        logger.info(f"📊 Processing {len(df)} queries with batch_size={BATCH_SIZE}")

        # Initialize architecture
        arch_instance = self._initialize_architecture(architecture, model)

        # Process in batches
        all_results = []
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=total_batches, desc=f"🚀 Batch processing with {architecture}_{model}") as pbar:
            for batch_start in range(0, len(df), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(df))
                batch_df = df.iloc[batch_start:batch_end]

                batch_results = self.process_batch_organ_specific(
                    arch_instance, batch_df, architecture
                )
                all_results.extend(batch_results)
                pbar.update(1)

        return self._calculate_metrics(all_results)

    def _calculate_set_metrics(self, predicted: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
        """Calculate set-based precision, recall, and F1 score"""
        if not predicted and not ground_truth:
            return 1.0, 1.0, 1.0
        if not predicted or not ground_truth:
            return 0.0, 0.0, 0.0

        predicted_set = set([p.lower().strip() for p in predicted])
        ground_truth_set = set([g.lower().strip() for g in ground_truth])

        true_positives = len(predicted_set & ground_truth_set)
        precision = true_positives / len(predicted_set) if predicted_set else 0.0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _evaluate_query_success(self, predicted: List[str], ground_truth: List[str], threshold: float = 0.3) -> int:
        """Evaluate if a query is successful based on recall threshold"""
        if not ground_truth:
            return 1 if not predicted else 0

        predicted_set = set([p.lower().strip() for p in predicted])
        ground_truth_set = set([g.lower().strip() for g in ground_truth])

        true_positives = len(predicted_set & ground_truth_set)
        recall = true_positives / len(ground_truth_set)

        return 1 if recall >= threshold else 0

    def _extract_effects_from_response(self, response: str) -> List[str]:
        """Extract side effects from LLM response text"""
        effects = []

        # Common patterns for extracting effects
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                effect = line[2:].strip().lower()
                if effect and len(effect) > 2:
                    effects.append(effect)
            elif ':' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    effect = parts[1].strip().lower()
                    if effect and len(effect) > 2:
                        effects.append(effect)

        return effects

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results"""
        if not results:
            return {}

        df_results = pd.DataFrame(results)

        metrics = {
            'total_queries': len(results),
            'avg_precision': df_results['precision'].mean(),
            'avg_recall': df_results['recall'].mean(),
            'avg_f1_score': df_results['f1_score'].mean(),
            'std_f1_score': df_results['f1_score'].std(),
            'query_success_rate': df_results['query_success'].mean(),
            'avg_elapsed_time': df_results['elapsed_time'].mean(),
            'total_time': df_results['elapsed_time'].sum(),
            'throughput_qps': len(results) / df_results['elapsed_time'].sum() if df_results['elapsed_time'].sum() > 0 else 0,
            'avg_confidence': df_results['confidence'].mean()
        }

        # Add speedup calculation
        sequential_time = df_results['elapsed_time'].sum()
        actual_time = metrics['total_time']
        metrics['speedup'] = sequential_time / actual_time if actual_time > 0 else 1.0

        return metrics

    def run_comprehensive_evaluation(self, architectures: List[str], models: List[str],
                                    query_types: List[str], limit: int = None):
        """Run comprehensive batch evaluation across architectures and query types"""
        all_results = {}

        for architecture in architectures:
            for model in models:
                key = f"{architecture}_{model}"
                logger.info(f"\n{'='*60}")
                logger.info(f"🚀 Evaluating {architecture} with {model}")
                logger.info(f"{'='*60}")

                results = {}

                if 'organ_specific' in query_types:
                    logger.info("🎯 Processing organ-specific queries...")
                    results['organ_specific'] = self.evaluate_organ_specific_batch(
                        architecture, model, limit
                    )

                # Add other query types here as needed
                # if 'severity_filtered' in query_types:
                #     results['severity_filtered'] = self.evaluate_severity_batch(...)

                all_results[key] = results

                # Display results
                self._display_results(key, results)

        return all_results

    def _display_results(self, architecture_key: str, results: Dict):
        """Display evaluation results"""
        logger.info(f"\n📊 Results for {architecture_key}:")
        logger.info("="*50)

        for query_type, metrics in results.items():
            logger.info(f"\n{query_type.upper()}:")
            logger.info(f"  Total queries: {metrics['total_queries']}")
            logger.info(f"  Avg Precision: {metrics['avg_precision']:.3f}")
            logger.info(f"  Avg Recall: {metrics['avg_recall']:.3f}")
            logger.info(f"  Avg F1 Score: {metrics['avg_f1_score']:.3f} (±{metrics['std_f1_score']:.3f})")
            logger.info(f"  Success Rate: {metrics['query_success_rate']*100:.1f}%")
            logger.info(f"  Avg Time/Query: {metrics['avg_elapsed_time']:.3f}s")
            logger.info(f"  Throughput: {metrics['throughput_qps']:.1f} queries/sec")
            logger.info(f"  Speedup: {metrics['speedup']:.2f}x")

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"batch_evaluation_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'max_workers': MAX_WORKERS,
                    'gpu_count': 4,
                    'tensor_parallel_size': 4
                },
                'results': results
            }, f, indent=2)

        logger.info(f"✅ Results saved to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Batch-Enabled Complex Query Evaluation')
    parser.add_argument('--architectures', nargs='+',
                       default=['format_b', 'enhanced_format_b', 'graphrag'],
                       help='Architectures to evaluate')
    parser.add_argument('--models', nargs='+', default=['qwen'],
                       help='Models to use (qwen, llama3)')
    parser.add_argument('--query_types', nargs='+',
                       default=['organ_specific'],
                       help='Query types to evaluate')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of queries per type (for testing)')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for processing')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Maximum concurrent workers')

    args = parser.parse_args()

    # Update global settings
    global BATCH_SIZE, MAX_WORKERS
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.max_workers

    logger.info(f"🚀 Starting batch evaluation with {multiprocessing.cpu_count()} CPUs and 4 GPUs")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Max workers: {MAX_WORKERS}")

    # Initialize evaluator
    evaluator = BatchComplexQueryEvaluator()

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        architectures=args.architectures,
        models=args.models,
        query_types=args.query_types,
        limit=args.limit
    )

    # Save results
    output_file = evaluator.save_results(results, 'batch_results.json')

    logger.info(f"\n✅ Batch evaluation complete! Results saved to {output_file}")
    logger.info(f"🚀 Achieved {len(results)} architecture evaluations with GPU acceleration")


if __name__ == "__main__":
    main()