#!/usr/bin/env python3
"""
Batch Processing Utilities for GPU-Accelerated Query Processing
Optimized for 4 GPUs with tensor parallelism
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class GPUBatchProcessor:
    """
    Batch processor optimized for 4 GPUs with tensor parallelism
    Handles dynamic batching, load balancing, and error recovery
    """

    def __init__(self, num_gpus: int = 4, optimal_batch_size: int = 50):
        """
        Initialize batch processor

        Args:
            num_gpus: Number of GPUs available
            optimal_batch_size: Optimal batch size per GPU
        """
        self.num_gpus = num_gpus
        self.optimal_batch_size = optimal_batch_size
        self.max_workers = num_gpus * 2  # 2 threads per GPU for optimal utilization

        logger.info(f"✅ GPU Batch Processor initialized:")
        logger.info(f"   • GPUs: {num_gpus}")
        logger.info(f"   • Optimal batch size: {optimal_batch_size}")
        logger.info(f"   • Max parallel workers: {self.max_workers}")

    def process_batch_parallel(self,
                              items: List[Any],
                              process_fn: Callable,
                              batch_size: Optional[int] = None,
                              description: str = "Processing") -> List[Any]:
        """
        Process items in parallel batches across GPUs

        Args:
            items: List of items to process
            process_fn: Function to process each batch
            batch_size: Override batch size (default: optimal_batch_size)
            description: Progress bar description

        Returns:
            List of processed results maintaining order
        """
        if not items:
            return []

        batch_size = batch_size or self.optimal_batch_size
        num_batches = (len(items) + batch_size - 1) // batch_size

        logger.info(f"📊 Processing {len(items)} items in {num_batches} batches")

        results = [None] * len(items)
        batch_futures = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(items))
                batch_items = items[start_idx:end_idx]

                future = executor.submit(
                    self._process_batch_with_timing,
                    process_fn,
                    batch_items,
                    batch_idx
                )
                batch_futures.append((future, start_idx, end_idx))

            # Collect results with progress bar
            with tqdm(total=len(items), desc=description, unit="item") as pbar:
                for future, start_idx, end_idx in batch_futures:
                    try:
                        batch_results = future.result(timeout=300)
                        for i, result in enumerate(batch_results):
                            results[start_idx + i] = result
                        pbar.update(end_idx - start_idx)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        # Fill with error results
                        for i in range(start_idx, end_idx):
                            results[i] = {"error": str(e)}
                        pbar.update(end_idx - start_idx)

        return results

    def _process_batch_with_timing(self, process_fn: Callable, batch_items: List[Any], batch_idx: int) -> List[Any]:
        """Process a batch with timing information"""
        start_time = time.time()

        try:
            results = process_fn(batch_items)
            elapsed = time.time() - start_time
            throughput = len(batch_items) / elapsed if elapsed > 0 else 0

            logger.debug(f"Batch {batch_idx}: {len(batch_items)} items in {elapsed:.2f}s ({throughput:.1f} items/s)")
            return results

        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {e}")
            return [{"error": str(e)} for _ in batch_items]

    def adaptive_batch_size(self, num_items: int) -> int:
        """
        Calculate adaptive batch size based on workload

        Args:
            num_items: Total number of items to process

        Returns:
            Optimal batch size
        """
        if num_items <= self.optimal_batch_size:
            return num_items

        # Distribute evenly across GPUs
        batches_per_gpu = max(1, num_items // (self.num_gpus * self.optimal_batch_size))

        if batches_per_gpu <= 1:
            # Small workload - use smaller batches for better parallelism
            return max(10, num_items // (self.num_gpus * 2))
        else:
            # Large workload - use optimal batch size
            return self.optimal_batch_size


class ComplexQueryBatchProcessor:
    """
    Specialized batch processor for complex drug queries
    Handles different query types with optimized batching
    """

    def __init__(self, architecture, num_gpus: int = 4):
        """
        Initialize complex query batch processor

        Args:
            architecture: Architecture instance (e.g., RAGFormatB, GraphRAG)
            num_gpus: Number of GPUs available
        """
        self.architecture = architecture
        self.gpu_processor = GPUBatchProcessor(num_gpus)

        # Check architecture capabilities
        self.supports_batch = hasattr(architecture, 'query_batch') or \
                            hasattr(architecture, 'generate_batch')

        logger.info(f"✅ Complex Query Batch Processor initialized")
        logger.info(f"   • Architecture: {architecture.__class__.__name__}")
        logger.info(f"   • Native batch support: {self.supports_batch}")

    def batch_organ_queries(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process organ-specific queries in batches

        Args:
            queries: List of organ query dicts with 'drug' and 'organ' keys

        Returns:
            List of results
        """
        if self.supports_batch and hasattr(self.architecture, 'batch_organ_specific_query'):
            # Native batch support
            return self.architecture.batch_organ_specific_query(queries)
        else:
            # Use parallel processing
            def process_batch(batch_queries):
                results = []
                for q in batch_queries:
                    try:
                        if hasattr(self.architecture, 'complex_query_organ_specific'):
                            result = self.architecture.complex_query_organ_specific(
                                q['drug'], q['organ']
                            )
                        elif hasattr(self.architecture, 'organ_specific_query'):
                            result = self.architecture.organ_specific_query(
                                q['drug'], q['organ']
                            )
                        else:
                            # Fallback to general query
                            query_text = f"What {q['organ']} side effects does {q['drug']} cause?"
                            result = self.architecture.query(query_text, "")
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e), 'drug': q['drug'], 'organ': q['organ']})
                return results

            return self.gpu_processor.process_batch_parallel(
                queries, process_batch, description="Organ queries"
            )

    def batch_comparison_queries(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process drug comparison queries in batches

        Args:
            queries: List of comparison query dicts with 'drug1' and 'drug2' keys

        Returns:
            List of results
        """
        def process_batch(batch_queries):
            results = []
            for q in batch_queries:
                try:
                    if hasattr(self.architecture, 'complex_query_drug_comparison'):
                        result = self.architecture.complex_query_drug_comparison(
                            q['drug1'], q['drug2']
                        )
                    elif hasattr(self.architecture, 'drug_comparison_query'):
                        result = self.architecture.drug_comparison_query(
                            q['drug1'], q['drug2']
                        )
                    else:
                        query_text = f"Compare side effects of {q['drug1']} and {q['drug2']}"
                        result = self.architecture.query(query_text, "")
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'drug1': q['drug1'], 'drug2': q['drug2']})
            return results

        return self.gpu_processor.process_batch_parallel(
            queries, process_batch, description="Comparison queries"
        )

    def batch_severity_queries(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process severity-filtered queries in batches

        Args:
            queries: List of severity query dicts with 'drug' and 'severity' keys

        Returns:
            List of results
        """
        def process_batch(batch_queries):
            results = []
            for q in batch_queries:
                try:
                    if hasattr(self.architecture, 'complex_query_severity_analysis'):
                        result = self.architecture.complex_query_severity_analysis(
                            q['drug'], q['severity']
                        )
                    elif hasattr(self.architecture, 'severity_filtered_query'):
                        result = self.architecture.severity_filtered_query(
                            q['drug'], q['severity']
                        )
                    else:
                        query_text = f"What are the {q['severity']} side effects of {q['drug']}?"
                        result = self.architecture.query(query_text, "")
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'drug': q['drug'], 'severity': q['severity']})
            return results

        return self.gpu_processor.process_batch_parallel(
            queries, process_batch, description="Severity queries"
        )

    def process_mixed_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process mixed query types efficiently by grouping

        Args:
            queries: List of query dicts with 'query_type' field

        Returns:
            List of results in original order
        """
        # Group queries by type
        grouped = {}
        index_map = {}

        for idx, query in enumerate(queries):
            query_type = query.get('query_type', 'general')
            if query_type not in grouped:
                grouped[query_type] = []
                index_map[query_type] = []
            grouped[query_type].append(query)
            index_map[query_type].append(idx)

        # Process each group
        all_results = [None] * len(queries)

        for query_type, type_queries in grouped.items():
            logger.info(f"Processing {len(type_queries)} {query_type} queries")

            if query_type == 'organ_specific':
                results = self.batch_organ_queries(type_queries)
            elif query_type == 'drug_comparison':
                results = self.batch_comparison_queries(type_queries)
            elif query_type == 'severity_filtered':
                results = self.batch_severity_queries(type_queries)
            else:
                # General queries
                results = self._batch_general_queries(type_queries)

            # Map results back to original positions
            for result, original_idx in zip(results, index_map[query_type]):
                all_results[original_idx] = result

        return all_results

    def _batch_general_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process general queries in batches"""
        def process_batch(batch_queries):
            results = []
            for q in batch_queries:
                try:
                    query_text = q.get('query', q.get('question', ''))
                    result = self.architecture.query(query_text, "")
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'query': query_text})
            return results

        return self.gpu_processor.process_batch_parallel(
            queries, process_batch, description="General queries"
        )


def benchmark_batch_processing(architecture, queries: List[Dict], num_gpus: int = 4):
    """
    Benchmark batch processing performance

    Args:
        architecture: Architecture instance
        queries: List of queries to process
        num_gpus: Number of GPUs to use

    Returns:
        Performance metrics
    """
    processor = ComplexQueryBatchProcessor(architecture, num_gpus)

    # Sequential baseline
    logger.info("Running sequential baseline...")
    start_time = time.time()
    sequential_results = []
    for q in tqdm(queries[:10], desc="Sequential"):  # Sample for baseline
        if q.get('query_type') == 'organ_specific':
            result = architecture.complex_query_organ_specific(q['drug'], q['organ'])
        else:
            result = architecture.query(q.get('query', ''), "")
        sequential_results.append(result)
    sequential_time = (time.time() - start_time) / 10 * len(queries)  # Extrapolate

    # Batch processing
    logger.info(f"Running batch processing with {num_gpus} GPUs...")
    start_time = time.time()
    batch_results = processor.process_mixed_queries(queries)
    batch_time = time.time() - start_time

    # Calculate metrics
    speedup = sequential_time / batch_time if batch_time > 0 else 1.0
    throughput = len(queries) / batch_time if batch_time > 0 else 0

    metrics = {
        'total_queries': len(queries),
        'sequential_time_estimated': sequential_time,
        'batch_time_actual': batch_time,
        'speedup': speedup,
        'throughput_qps': throughput,
        'efficiency': speedup / num_gpus  # Parallel efficiency
    }

    logger.info(f"\n📊 BENCHMARK RESULTS:")
    logger.info(f"  Sequential (estimated): {sequential_time:.1f}s")
    logger.info(f"  Batch processing: {batch_time:.1f}s")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Throughput: {throughput:.1f} queries/sec")
    logger.info(f"  Parallel efficiency: {metrics['efficiency']:.1%}")

    return metrics