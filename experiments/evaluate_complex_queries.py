#!/usr/bin/env python3
"""
Complex Query Evaluation System for DrugRAG
Evaluates all 5 complex query types across all architectures:
1. Organ-Specific Queries (1,244 total)
2. Severity-Filtered Queries (461 total)
3. Drug Comparison Queries (147 total)
4. Reverse Lookup Queries (600 total)
5. Combination Queries (453 total)
"""

import json
import pandas as pd
import numpy as np
import argparse
import time
import logging
import sys
import os
from typing import Dict, List, Any, Optional
import ast
from tqdm import tqdm

# Add parent directory to path for metrics import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import calculate_binary_classification_metrics, print_metrics_summary

# Suppress verbose HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplexQueryEvaluator:
    """Evaluate complex queries across all architectures"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.query_datasets = {
            'organ_specific': '../data/processed/comprehensive_organ_queries.csv',  # 945 queries with cleaner ground truth
            'severity_filtered': '../data/processed/comprehensive_severity_queries.csv',  # 877 queries, more comprehensive
            'drug_comparison': '../data/processed/comprehensive_comparison_queries.csv',  # 147 queries, OK
            'reverse_lookup': '../data/processed/reverse_queries.csv',  # 600 queries, correct format
            'combination': '../data/processed/comprehensive_combination_queries.csv'  # 453 queries, correct dataset
        }

    def load_query_dataset(self, query_type: str) -> pd.DataFrame:
        """Load a specific query dataset"""
        file_path = self.query_datasets.get(query_type)
        if not file_path:
            raise ValueError(f"Unknown query type: {query_type}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} {query_type} queries")
        return df

    def evaluate_organ_specific_queries(self, architecture: str, model: str, test_size: int = 50) -> List[Dict]:
        """Evaluate organ-specific queries"""
        logger.info(f"🔬 Evaluating organ-specific queries: {architecture} ({model})")

        # Load dataset
        df = self.load_query_dataset('organ_specific')

        # Sample queries
        if test_size < len(df):
            df = df.sample(n=test_size, random_state=42)

        # Initialize architecture
        arch_instance = self._initialize_architecture(architecture, model)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="🎯 Organ-specific queries", unit="query"):
            query = row['query']
            drug = row['drug']
            organ_filter = row['organ_filter']
            ground_truth = ast.literal_eval(row['ground_truth'])
            expected_count = row['num_results']

            start_time = time.time()

            try:
                # Execute query based on architecture
                if 'graphrag' in architecture:
                    result = arch_instance.complex_query_organ_specific(drug, organ_filter)
                    predicted_effects = result.get('side_effects_found', [])
                    confidence = 1.0 if result.get('count', 0) > 0 else 0.0

                elif 'enhanced_format_b' in architecture:
                    result = arch_instance.organ_specific_query(drug, organ_filter)
                    predicted_effects = result.get('side_effects', [])
                    confidence = result.get('confidence', 0.0)

                else:
                    # Standard RAG approaches - convert to binary-style reasoning
                    combined_query = f"What {organ_filter} side effects does {drug} cause?"
                    result = arch_instance.query(combined_query, "")
                    predicted_effects = self._extract_effects_from_response(result.get('llm_response', ''))
                    confidence = result.get('confidence', 0.0)

                elapsed_time = time.time() - start_time

                # Calculate set-based metrics
                precision, recall, f1 = self._calculate_set_metrics(predicted_effects, ground_truth)

                # Calculate binary success metric
                query_success = self._evaluate_query_success(predicted_effects, ground_truth)

                results.append({
                    'query': query,
                    'query_type': 'organ_specific',
                    'drug': drug,
                    'organ_filter': organ_filter,
                    'ground_truth': ground_truth,
                    'ground_truth_count': expected_count,
                    'predicted_effects': predicted_effects,
                    'predicted_count': len(predicted_effects),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'query_success': query_success,  # Binary success metric
                    'confidence': confidence,
                    'response_time': elapsed_time,
                    'architecture': architecture,
                    'model': model
                })

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'query_type': 'organ_specific',
                    'error': str(e),
                    'architecture': architecture,
                    'model': model
                })

        return results

    def evaluate_severity_filtered_queries(self, architecture: str, model: str, test_size: int = 50) -> List[Dict]:
        """Evaluate severity-filtered queries"""
        logger.info(f"⚠️ Evaluating severity-filtered queries: {architecture} ({model})")

        # Load dataset
        df = self.load_query_dataset('severity_filtered')

        # Sample queries
        if test_size < len(df):
            df = df.sample(n=test_size, random_state=42)

        # Initialize architecture
        arch_instance = self._initialize_architecture(architecture, model)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="⚠️ Severity-filtered queries", unit="query"):
            query = row['query']
            drug = row['drug']
            severity_filter = row['severity_filter']
            ground_truth = ast.literal_eval(row['ground_truth'])
            expected_count = row['num_results']

            start_time = time.time()

            try:
                # Execute query based on architecture
                if 'graphrag' in architecture:
                    result = arch_instance.complex_query_severity_filter(drug, severity_filter)
                    predicted_effects = result.get('side_effects_found', [])
                    confidence = 1.0 if result.get('count', 0) > 0 else 0.0

                elif 'enhanced_format_b' in architecture:
                    result = arch_instance.severity_filtered_query(drug, severity_filter)
                    predicted_effects = result.get('side_effects', [])
                    confidence = result.get('confidence', 0.0)

                else:
                    # Standard RAG approaches
                    combined_query = f"List {severity_filter} adverse events of {drug}"
                    result = arch_instance.query(combined_query, "")
                    predicted_effects = self._extract_effects_from_response(result.get('llm_response', ''))
                    confidence = result.get('confidence', 0.0)

                elapsed_time = time.time() - start_time

                # Calculate set-based metrics
                precision, recall, f1 = self._calculate_set_metrics(predicted_effects, ground_truth)

                # Calculate binary success metric
                query_success = self._evaluate_query_success(predicted_effects, ground_truth)

                results.append({
                    'query': query,
                    'query_type': 'severity_filtered',
                    'drug': drug,
                    'severity_filter': severity_filter,
                    'ground_truth': ground_truth,
                    'ground_truth_count': expected_count,
                    'predicted_effects': predicted_effects,
                    'predicted_count': len(predicted_effects),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'query_success': query_success,  # Binary success metric
                    'confidence': confidence,
                    'response_time': elapsed_time,
                    'architecture': architecture,
                    'model': model
                })

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'query_type': 'severity_filtered',
                    'error': str(e),
                    'architecture': architecture,
                    'model': model
                })

        return results

    def evaluate_drug_comparison_queries(self, architecture: str, model: str, test_size: int = 50) -> List[Dict]:
        """Evaluate drug comparison queries"""
        logger.info(f"🔄 Evaluating drug comparison queries: {architecture} ({model})")

        # Load dataset
        df = self.load_query_dataset('drug_comparison')

        # Sample queries
        if test_size < len(df):
            df = df.sample(n=test_size, random_state=42)

        # Initialize architecture
        arch_instance = self._initialize_architecture(architecture, model)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="⚖️ Drug comparison queries", unit="query"):
            query = row['query']
            comparison_type = row.get('comparison_type', 'common')
            drug1 = row.get('drug1', '')
            drug2 = row.get('drug2', '')
            ground_truth = row['ground_truth']

            # Parse ground truth based on type
            if comparison_type == 'common':
                expected_effects = ast.literal_eval(ground_truth) if isinstance(ground_truth, str) else ground_truth
            else:
                # For unique comparisons or complex structure
                if isinstance(ground_truth, str):
                    try:
                        expected_effects = ast.literal_eval(ground_truth)
                    except:
                        expected_effects = []
                else:
                    expected_effects = ground_truth

            start_time = time.time()

            try:
                # Execute query based on architecture
                if 'graphrag' in architecture:
                    if comparison_type == 'common':
                        result = arch_instance.complex_query_drug_comparison(drug1, drug2)
                        predicted_effects = result.get('common_effects', [])
                    else:
                        # Handle unique comparisons
                        result = arch_instance.complex_query_drug_unique(drug1, drug2)
                        predicted_effects = result.get('unique_effects', [])
                    confidence = 1.0 if len(predicted_effects) > 0 else 0.0

                elif 'enhanced_format_b' in architecture:
                    result = arch_instance.drug_comparison_query(drug1, drug2, comparison_type)
                    predicted_effects = result.get('effects', [])
                    confidence = result.get('confidence', 0.0)

                else:
                    # Standard RAG approaches
                    result = arch_instance.query(query, "")
                    predicted_effects = self._extract_effects_from_response(result.get('llm_response', ''))
                    confidence = result.get('confidence', 0.0)

                elapsed_time = time.time() - start_time

                # Calculate set-based metrics
                precision, recall, f1 = self._calculate_set_metrics(predicted_effects, expected_effects)

                # Calculate binary success metric
                query_success = self._evaluate_query_success(predicted_effects, expected_effects)

                results.append({
                    'query': query,
                    'query_type': 'drug_comparison',
                    'comparison_type': comparison_type,
                    'drug1': drug1,
                    'drug2': drug2,
                    'ground_truth': expected_effects,
                    'ground_truth_count': len(expected_effects) if isinstance(expected_effects, list) else 0,
                    'predicted_effects': predicted_effects,
                    'predicted_count': len(predicted_effects),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'query_success': query_success,  # Binary success metric
                    'confidence': confidence,
                    'response_time': elapsed_time,
                    'architecture': architecture,
                    'model': model
                })

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'query_type': 'drug_comparison',
                    'error': str(e),
                    'architecture': architecture,
                    'model': model
                })

        return results

    def evaluate_reverse_lookup_queries(self, architecture: str, model: str, test_size: int = 50) -> List[Dict]:
        """Evaluate reverse lookup queries"""
        logger.info(f"🔍 Evaluating reverse lookup queries: {architecture} ({model})")

        # Load dataset
        df = self.load_query_dataset('reverse_lookup')

        # Sample queries
        if test_size < len(df):
            df = df.sample(n=test_size, random_state=42)

        # Initialize architecture
        arch_instance = self._initialize_architecture(architecture, model)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Reverse lookup queries", unit="query"):
            query = row['query']
            drug = row['drug']
            ground_truth = ast.literal_eval(row['ground_truth_side_effects'])
            expected_count = row['num_side_effects']

            start_time = time.time()

            try:
                # Execute query based on architecture
                if 'graphrag' in architecture:
                    result = arch_instance.complex_query_all_effects(drug)
                    predicted_effects = result.get('side_effects_found', [])
                    confidence = 1.0 if result.get('count', 0) > 0 else 0.0

                elif 'enhanced_format_b' in architecture:
                    result = arch_instance.reverse_lookup_query(drug)
                    predicted_effects = result.get('side_effects', [])
                    confidence = result.get('confidence', 0.0)

                else:
                    # Standard RAG approaches
                    result = arch_instance.query(f"List all side effects of {drug}", "")
                    predicted_effects = self._extract_effects_from_response(result.get('llm_response', ''))
                    confidence = result.get('confidence', 0.0)

                elapsed_time = time.time() - start_time

                # Calculate set-based metrics
                precision, recall, f1 = self._calculate_set_metrics(predicted_effects, ground_truth)

                # Calculate binary success metric
                query_success = self._evaluate_query_success(predicted_effects, ground_truth)

                results.append({
                    'query': query,
                    'query_type': 'reverse_lookup',
                    'drug': drug,
                    'ground_truth': ground_truth,
                    'ground_truth_count': expected_count,
                    'predicted_effects': predicted_effects,
                    'predicted_count': len(predicted_effects),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'query_success': query_success,  # Binary success metric
                    'confidence': confidence,
                    'response_time': elapsed_time,
                    'architecture': architecture,
                    'model': model
                })

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    'query': query,
                    'query_type': 'reverse_lookup',
                    'error': str(e),
                    'architecture': architecture,
                    'model': model
                })

        return results

    def _initialize_architecture(self, architecture: str, model: str):
        """Initialize the specified architecture"""
        if architecture.startswith('pure_llm'):
            if model == 'qwen':
                from src.models.vllm_model import VLLMQwenModel
                return VLLMQwenModel(self.config_path)
            elif model == 'llama3':
                from src.models.vllm_model import VLLMLLAMA3Model
                return VLLMLLAMA3Model(self.config_path)

        elif architecture.startswith('format_a'):
            from src.architectures.rag_format_a import RAGFormatA
            return RAGFormatA(self.config_path, model)

        elif architecture.startswith('format_b'):
            from src.architectures.rag_format_b import RAGFormatB
            return RAGFormatB(self.config_path, model)

        elif architecture.startswith('graphrag'):
            from src.architectures.graphrag import GraphRAG
            return GraphRAG(self.config_path, model)

        elif architecture.startswith('enhanced_format_b'):
            from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
            return EnhancedRAGFormatB(self.config_path, model)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _extract_effects_from_response(self, response: str) -> List[str]:
        """Extract side effects from LLM response"""
        # Simple extraction - look for lists, comma-separated values, etc.
        effects = []

        if not response:
            return effects

        response = response.lower()

        # Look for common patterns
        import re

        # Pattern 1: Bullet points or numbered lists
        bullet_pattern = r'[•\-\*\d+\.]\s*([a-zA-Z\s\-]+)'
        matches = re.findall(bullet_pattern, response)
        effects.extend([m.strip() for m in matches])

        # Pattern 2: Comma-separated in sentences
        if 'side effects include' in response or 'effects are' in response:
            # Extract text after these phrases
            start_phrases = ['side effects include', 'effects are', 'adverse effects include']
            for phrase in start_phrases:
                if phrase in response:
                    text_after = response.split(phrase, 1)[1]
                    # Extract until sentence end
                    sentence_end = min([text_after.find('.'), text_after.find('\n'), len(text_after)])
                    if sentence_end > 0:
                        effects_text = text_after[:sentence_end]
                        # Split by commas and clean
                        comma_effects = [e.strip() for e in effects_text.split(',')]
                        effects.extend(comma_effects)
                    break

        # Clean and deduplicate
        effects = [e.strip() for e in effects if e.strip() and len(e.strip()) > 2]
        return list(set(effects))[:20]  # Limit to top 20 to avoid noise

    def _calculate_set_metrics(self, predicted: List[str], actual: List[str]) -> tuple:
        """Calculate precision, recall, F1 for sets of effects"""
        if not actual:
            return 0.0, 0.0, 0.0

        if not predicted:
            return 0.0, 0.0, 0.0

        # Convert to lowercase sets for comparison
        pred_set = set([p.lower().strip() for p in predicted])
        actual_set = set([a.lower().strip() for a in actual])

        # Calculate intersection
        intersection = pred_set.intersection(actual_set)

        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(actual_set) if actual_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _evaluate_query_success(self, predicted_effects: List[str], ground_truth: List[str], threshold: float = 0.5) -> int:
        """
        Evaluate if a complex query was successful as a binary classification.

        Args:
            predicted_effects: List of predicted effects
            ground_truth: List of expected effects
            threshold: F1 threshold above which we consider the query successful

        Returns:
            int: 1 if successful (F1 > threshold), 0 if not
        """
        if not ground_truth:
            # If no ground truth, success = finding nothing
            return 1 if not predicted_effects else 0

        precision, recall, f1 = self._calculate_set_metrics(predicted_effects, ground_truth)
        return 1 if f1 >= threshold else 0

    def save_results(self, results: List[Dict], output_file: str):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    def run_comprehensive_evaluation(self, architecture: str, test_size: int = 50):
        """Run evaluation on all complex query types"""
        model = architecture.split('_')[-1]  # Extract model from architecture name
        arch_name = '_'.join(architecture.split('_')[:-1])

        logger.info(f"🚀 Starting comprehensive complex query evaluation: {architecture}")

        all_results = []

        # 1. Organ-specific queries
        organ_results = self.evaluate_organ_specific_queries(architecture, model, test_size)
        all_results.extend(organ_results)

        # 2. Severity-filtered queries
        severity_results = self.evaluate_severity_filtered_queries(architecture, model, test_size)
        all_results.extend(severity_results)

        # 3. Drug comparison queries
        comparison_results = self.evaluate_drug_comparison_queries(architecture, model, test_size)
        all_results.extend(comparison_results)

        # 4. Reverse lookup queries
        reverse_results = self.evaluate_reverse_lookup_queries(architecture, model, test_size)
        all_results.extend(reverse_results)

        # Save results
        output_file = f"results_complex_{architecture}.json"
        self.save_results(all_results, output_file)

        # Calculate summary metrics
        self._print_summary(all_results, architecture)

        return all_results

    def _print_summary(self, results: List[Dict], architecture: str):
        """Print evaluation summary with both set-based and binary classification metrics"""
        successful_results = [r for r in results if 'error' not in r]

        if not successful_results:
            logger.warning("No successful evaluations")
            return

        # Group by query type
        by_type = {}
        for result in successful_results:
            query_type = result['query_type']
            if query_type not in by_type:
                by_type[query_type] = []
            by_type[query_type].append(result)

        print(f"\n📊 COMPLEX QUERY EVALUATION SUMMARY - {architecture}")
        print("=" * 80)

        # Calculate binary classification metrics across all queries
        all_query_success = [r.get('query_success', 0) for r in successful_results]
        all_ground_truth_binary = [1] * len(all_query_success)  # All queries should ideally succeed

        if all_query_success:
            binary_metrics = calculate_binary_classification_metrics(
                all_ground_truth_binary, all_query_success
            )

        for query_type, type_results in by_type.items():
            if not type_results:
                continue

            # Set-based metrics (existing)
            avg_precision = np.mean([r['precision'] for r in type_results])
            avg_recall = np.mean([r['recall'] for r in type_results])
            avg_f1 = np.mean([r['f1_score'] for r in type_results])
            avg_confidence = np.mean([r['confidence'] for r in type_results])
            avg_time = np.mean([r['response_time'] for r in type_results])

            # Binary query success metrics
            type_query_success = [r.get('query_success', 0) for r in type_results]
            type_success_rate = np.mean(type_query_success) if type_query_success else 0.0

            print(f"\n{query_type.upper()} QUERIES ({len(type_results)} samples):")
            print(f"  Set-based Metrics:")
            print(f"    Precision: {avg_precision:.3f}")
            print(f"    Recall: {avg_recall:.3f}")
            print(f"    F1-Score: {avg_f1:.3f}")
            print(f"  Binary Query Success Rate: {type_success_rate:.3f}")
            print(f"  Confidence: {avg_confidence:.3f}")
            print(f"  Avg Response Time: {avg_time:.3f}s")

        # Overall metrics
        overall_precision = np.mean([r['precision'] for r in successful_results])
        overall_recall = np.mean([r['recall'] for r in successful_results])
        overall_f1 = np.mean([r['f1_score'] for r in successful_results])
        overall_confidence = np.mean([r['confidence'] for r in successful_results])
        overall_time = np.mean([r['response_time'] for r in successful_results])
        overall_success_rate = np.mean(all_query_success) if all_query_success else 0.0

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Set-based Metrics:")
        print(f"    Precision: {overall_precision:.3f}")
        print(f"    Recall: {overall_recall:.3f}")
        print(f"    F1-Score: {overall_f1:.3f}")
        print(f"  Binary Query Success Rate: {overall_success_rate:.3f}")
        print(f"  Confidence: {overall_confidence:.3f}")
        print(f"  Avg Response Time: {overall_time:.3f}s")
        print(f"  Processing Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")

        # Display comprehensive binary classification metrics for overall query success
        if all_query_success:
            print(f"\nBINARY CLASSIFICATION METRICS (Query Success):")
            print(f"  Accuracy:    {binary_metrics['accuracy']:.4f}")
            print(f"  F1 Score:    {binary_metrics['f1_score']:.4f}")
            print(f"  Precision:   {binary_metrics['precision']:.4f}")
            print(f"  Sensitivity: {binary_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {binary_metrics['specificity']:.4f}")
            print(f"  Confusion Matrix: TP={binary_metrics['tp']}, TN={binary_metrics['tn']}, FP={binary_metrics['fp']}, FN={binary_metrics['fn']}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate complex queries across all architectures')
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['format_a_qwen', 'format_a_llama3',
                               'format_b_qwen', 'format_b_llama3',
                               'graphrag_qwen', 'graphrag_llama3',
                               'enhanced_format_b_qwen', 'enhanced_format_b_llama3',
                               'pure_llm_qwen', 'pure_llm_llama3'],
                       help='Architecture to evaluate')
    parser.add_argument('--test_size', type=int, default=50,
                       help='Number of queries to test per type')
    parser.add_argument('--query_type', type=str, default='all',
                       choices=['all', 'organ_specific', 'severity_filtered',
                               'drug_comparison', 'reverse_lookup'],
                       help='Type of queries to evaluate')

    args = parser.parse_args()

    evaluator = ComplexQueryEvaluator()

    if args.query_type == 'all':
        evaluator.run_comprehensive_evaluation(args.architecture, args.test_size)
    else:
        # Run specific query type
        model = args.architecture.split('_')[-1]

        if args.query_type == 'organ_specific':
            results = evaluator.evaluate_organ_specific_queries(args.architecture, model, args.test_size)
        elif args.query_type == 'severity_filtered':
            results = evaluator.evaluate_severity_filtered_queries(args.architecture, model, args.test_size)
        elif args.query_type == 'drug_comparison':
            results = evaluator.evaluate_drug_comparison_queries(args.architecture, model, args.test_size)
        elif args.query_type == 'reverse_lookup':
            results = evaluator.evaluate_reverse_lookup_queries(args.architecture, model, args.test_size)

        output_file = f"results_complex_{args.query_type}_{args.architecture}.json"
        evaluator.save_results(results, output_file)
        evaluator._print_summary(results, args.architecture)

if __name__ == "__main__":
    main()