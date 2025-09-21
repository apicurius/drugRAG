#!/usr/bin/env python3
"""
Enhanced Complex Query Evaluation Pipeline
Uses advanced architectures with chain-of-thought reasoning and semantic metrics
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
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced architectures
from src.architectures.enhanced_graphrag import EnhancedGraphRAG
from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
from src.architectures.advanced_rag_format_b import AdvancedRAGFormatB
from src.evaluation.advanced_metrics import AdvancedMetrics
# Data is already in Pinecone and Neo4j - no local processor needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedComplexQueryEvaluator:
    """Evaluate enhanced architectures on complex drug queries"""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize evaluator with configuration

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Data is already in Pinecone and Neo4j databases

        # Initialize advanced metrics
        self.metrics = AdvancedMetrics()

        # Initialize architectures (lazy loading)
        self.architectures = {}

        logger.info("✅ Enhanced Complex Query Evaluator initialized")

    def initialize_architecture(self, architecture: str, model: str = "qwen") -> Any:
        """
        Initialize specified architecture

        Args:
            architecture: Architecture name
            model: Model to use (qwen or llama3)

        Returns:
            Initialized architecture instance
        """
        key = f"{architecture}_{model}"

        if key in self.architectures:
            return self.architectures[key]

        logger.info(f"Initializing {architecture} with {model}...")

        if architecture == "enhanced_graphrag":
            arch = EnhancedGraphRAG(self.config_path, model)
        elif architecture == "enhanced_format_b":
            arch = EnhancedRAGFormatB(self.config_path, model)
        elif architecture == "advanced_rag_format_b":
            arch = AdvancedRAGFormatB(self.config_path, model)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.architectures[key] = arch
        return arch

    def generate_test_queries(self, query_type: str, num_queries: Optional[int] = 50) -> List[Dict[str, Any]]:
        """
        Load test queries from CSV files

        Args:
            query_type: Type of queries to generate
            num_queries: Number of queries (None for all available)

        Returns:
            List of test queries with ground truth
        """
        test_queries = []

        # Map query types to CSV files
        query_files = {
            'organ_specific': '../data/processed/comprehensive_organ_queries.csv',
            'severity_filtered': '../data/processed/comprehensive_severity_queries.csv',
            'drug_comparison': '../data/processed/comprehensive_comparison_queries.csv',
            'reverse_lookup': '../data/processed/reverse_queries.csv',
            'combination': '../data/processed/comprehensive_combination_queries.csv'
        }

        if query_type in query_files:
            # Load queries from CSV
            csv_path = query_files[query_type]
            try:
                df = pd.read_csv(csv_path)

                # Limit queries if specified
                if num_queries is not None:
                    df = df.head(num_queries)

                # Convert DataFrame to list of dicts
                for i, row in df.iterrows():
                    query_data = {
                        'query_id': f"{query_type}_{i}",
                        'query': row.get('query', row.get('question', '')),
                        'query_type': query_type
                    }

                    # Add expected answer or ground truth
                    ground_truth = None
                    for gt_col in ['ground_truth', 'expected_answer', 'expected_effects']:
                        if gt_col in row:
                            ground_truth = row[gt_col]
                            break

                    if ground_truth is not None:
                        if isinstance(ground_truth, str):
                            # Parse JSON-like string or comma-separated list
                            import ast
                            try:
                                query_data['ground_truth'] = ast.literal_eval(ground_truth)
                            except:
                                query_data['ground_truth'] = [s.strip() for s in ground_truth.split(',')]
                        else:
                            query_data['ground_truth'] = [ground_truth]
                    else:
                        query_data['ground_truth'] = []

                    # Add other metadata from CSV
                    for col in ['drug', 'drug1', 'drug2', 'organ', 'organ_filter', 'severity', 'severity_filter',
                                'side_effect', 'num_results', 'comparison_type']:
                        if col in row:
                            query_data[col] = row[col]

                    # Map filter columns to standard names for compatibility
                    if 'organ_filter' in query_data and 'organ' not in query_data:
                        query_data['organ'] = query_data['organ_filter']
                    if 'severity_filter' in query_data and 'severity' not in query_data:
                        query_data['severity'] = query_data['severity_filter']

                    test_queries.append(query_data)

                logger.info(f"Loaded {len(test_queries)} queries from {csv_path}")
            except Exception as e:
                logger.error(f"Error loading queries from {csv_path}: {e}")
                # Fall back to generated queries
                return self._generate_mock_queries(query_type, num_queries)

        else:
            # Unknown query type - generate mock queries
            return self._generate_mock_queries(query_type, num_queries)

        return test_queries

    def _generate_mock_queries(self, query_type: str, num_queries: Optional[int] = 50) -> List[Dict[str, Any]]:
        """Generate mock queries for testing"""
        test_queries = []

        if query_type == "drug_comparison":
            # Drug pairs for comparison
            drug_pairs = [
                ('aspirin', 'ibuprofen'),
                ('metformin', 'glipizide'),
                ('lisinopril', 'losartan'),
                ('atorvastatin', 'simvastatin'),
                ('warfarin', 'rivaroxaban'),
                ('omeprazole', 'ranitidine'),
                ('metoprolol', 'atenolol'),
                ('gabapentin', 'pregabalin'),
                ('sertraline', 'fluoxetine'),
                ('prednisone', 'methylprednisolone')
            ]

            for i, (drug1, drug2) in enumerate(drug_pairs[:num_queries]):
                ground_truth = self._get_comparison_ground_truth(drug1, drug2)

                test_queries.append({
                    'query_id': f"compare_{i}",
                    'query': f"Compare the side effects of {drug1} and {drug2}",
                    'query_type': 'drug_comparison',
                    'drug1': drug1,
                    'drug2': drug2,
                    'ground_truth': ground_truth
                })

        if query_type == "severity_filtered":
            # Severity-based queries
            drugs = ['warfarin', 'methotrexate', 'chemotherapy', 'insulin', 'digoxin']
            severities = ['severe', 'life-threatening', 'serious']

            for i in range(min(num_queries, len(drugs) * len(severities))):
                drug = drugs[i % len(drugs)]
                severity = severities[(i // len(drugs)) % len(severities)]

                ground_truth = self._get_severity_filtered_ground_truth(drug, severity)

                test_queries.append({
                    'query_id': f"severity_{i}",
                    'query': f"What are the {severity} side effects of {drug}?",
                    'query_type': 'severity_filtered',
                    'drug': drug,
                    'severity': severity,
                    'ground_truth': ground_truth
                })

        return test_queries

    def evaluate_architecture(self, architecture: str, model: str, test_queries: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate a single architecture on test queries

        Args:
            architecture: Architecture name
            model: Model name (qwen/llama3)
            test_queries: List of test queries

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {architecture} with {model} on {len(test_queries)} queries...")

        # Initialize architecture
        arch = self.initialize_architecture(architecture, model)

        results = []
        processing_times = []

        for query_data in tqdm(test_queries, desc=f"{architecture}_{model}"):
            start_time = time.time()

            try:
                # Process query based on type
                if query_data['query_type'] == 'organ_specific':
                    if architecture == "enhanced_graphrag":
                        response = arch.enhanced_organ_specific_query(
                            query_data['drug'],
                            query_data['organ']
                        )
                    elif architecture == "enhanced_format_b":
                        response = arch.complex_query_organ_specific(
                            query_data['drug'],
                            query_data['organ']
                        )
                    else:  # advanced_rag_format_b
                        response = arch.enhanced_complex_organ_query(
                            query_data['drug'],
                            query_data['organ']
                        )

                elif query_data['query_type'] == 'drug_comparison':
                    if architecture == "enhanced_graphrag":
                        response = arch.enhanced_drug_comparison(
                            query_data['drug1'],
                            query_data['drug2']
                        )
                    elif architecture == "enhanced_format_b":
                        response = arch.complex_query_drug_comparison(
                            query_data['drug1'],
                            query_data['drug2']
                        )
                    else:  # advanced_rag_format_b
                        response = arch.enhanced_complex_comparison(
                            query_data['drug1'],
                            query_data['drug2']
                        )

                elif query_data['query_type'] == 'severity_filtered':
                    if architecture == "enhanced_format_b":
                        response = arch.complex_query_severity_analysis(
                            query_data['drug'],
                            query_data['severity']
                        )
                    else:
                        # Use general complex query handler for others
                        response = arch.process_complex_query(query_data['query'])

                elif query_data['query_type'] == 'reverse_lookup':
                    if architecture == "enhanced_format_b":
                        response = arch.complex_query_reverse_lookup(
                            query_data.get('side_effect', query_data['query'])
                        )
                    else:
                        # Use general complex query handler for others
                        response = arch.process_complex_query(query_data['query'])

                else:
                    # Use general complex query handler
                    if architecture == "enhanced_format_b":
                        # EnhancedRAGFormatB doesn't have process_complex_query
                        # Fall back to binary query or skip
                        response = {'error': 'Complex query type not supported'}
                    else:
                        response = arch.process_complex_query(query_data['query'])

                # Extract predicted effects from response
                predicted_effects = self._extract_effects_from_response(response)

                # Calculate metrics
                eval_metrics = self.metrics.comprehensive_evaluation(
                    predicted_effects,
                    query_data['ground_truth'],
                    query_data['query_type']
                )

                # Record results
                results.append({
                    'query_id': query_data['query_id'],
                    'query_type': query_data['query_type'],
                    'metrics': eval_metrics,
                    'response': response,
                    'predicted_count': len(predicted_effects),
                    'ground_truth_count': len(query_data['ground_truth'])
                })

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

            except Exception as e:
                logger.error(f"Error processing query {query_data['query_id']}: {e}")
                results.append({
                    'query_id': query_data['query_id'],
                    'query_type': query_data['query_type'],
                    'error': str(e),
                    'metrics': self.metrics.comprehensive_evaluation([], query_data['ground_truth'])
                })
                processing_times.append(0)

        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(results)
        aggregated_metrics['avg_processing_time'] = np.mean(processing_times) if processing_times else 0
        aggregated_metrics['total_queries'] = len(test_queries)
        aggregated_metrics['successful_queries'] = sum(1 for r in results if 'error' not in r)

        return {
            'architecture': architecture,
            'model': model,
            'aggregated_metrics': aggregated_metrics,
            'detailed_results': results
        }

    def run_comprehensive_evaluation(self, architectures: List[str], models: List[str],
                                    query_types: List[str], queries_per_type: Optional[int] = 20):
        """
        Run comprehensive evaluation across architectures and query types

        Args:
            architectures: List of architectures to evaluate
            models: List of models to use
            query_types: List of query types to test
            queries_per_type: Number of queries per type (None for all available)

        Returns:
            Complete evaluation results
        """
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'architectures': architectures,
                'models': models,
                'query_types': query_types,
                'queries_per_type': queries_per_type
            },
            'results': []
        }

        # Generate test queries for all types
        all_test_queries = []
        for query_type in query_types:
            test_queries = self.generate_test_queries(query_type, queries_per_type)
            all_test_queries.extend(test_queries)

        logger.info(f"Generated {len(all_test_queries)} test queries across {len(query_types)} types")

        # Evaluate each architecture-model combination
        for architecture in architectures:
            for model in models:
                logger.info(f"\n{'='*50}")
                logger.info(f"Evaluating {architecture} with {model}")
                logger.info(f"{'='*50}")

                eval_results = self.evaluate_architecture(
                    architecture,
                    model,
                    all_test_queries
                )

                all_results['results'].append(eval_results)

                # Print summary
                self._print_evaluation_summary(eval_results)

        return all_results

    def _extract_effects_from_response(self, response: Dict[str, Any]) -> List[str]:
        """Extract predicted effects from architecture response"""
        effects = []

        # Try different response formats
        if 'effects' in response:
            effects = response['effects']
            if isinstance(effects, dict):
                # Handle categorized effects
                for category, effect_list in effects.items():
                    if isinstance(effect_list, list):
                        effects.extend(effect_list)
            elif isinstance(effects, list):
                # Handle direct list
                if effects and isinstance(effects[0], dict):
                    effects = [e.get('effect', e.get('name', str(e))) for e in effects]

        elif 'side_effects_found' in response:
            effects = response['side_effects_found']

        elif 'common_effects' in response:
            effects = response['common_effects']
            if isinstance(effects[0], dict):
                effects = [e.get('name', str(e)) for e in effects]

        elif 'effects_by_severity' in response:
            for severity, effect_list in response['effects_by_severity'].items():
                effects.extend(effect_list)

        # Ensure all effects are strings
        return [str(e) for e in effects if e]

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all queries"""
        if not results:
            return {}

        # Extract all metrics
        all_metrics = [r['metrics'] for r in results if 'metrics' in r]

        if not all_metrics:
            return {}

        # Aggregate different metric types
        aggregated = {
            'avg_precision': np.mean([m['basic_metrics']['precision'] for m in all_metrics]),
            'avg_recall': np.mean([m['basic_metrics']['recall'] for m in all_metrics]),
            'avg_f1': np.mean([m['basic_metrics']['f1'] for m in all_metrics]),
            'avg_semantic_similarity': np.mean([m['semantic_metrics']['similarity'] for m in all_metrics]),
            'avg_ndcg': np.mean([m['ranking_metrics']['ndcg@10'] for m in all_metrics]),
            'avg_map': np.mean([m['ranking_metrics']['map'] for m in all_metrics]),
            'avg_safety_score': np.mean([m['clinical_metrics']['safety_score'] for m in all_metrics]),
            'avg_completeness': np.mean([m['clinical_metrics']['completeness'] for m in all_metrics]),
            'avg_aggregate_score': np.mean([m['aggregate_score'] for m in all_metrics])
        }

        # Add standard deviations for key metrics
        aggregated['std_f1'] = np.std([m['basic_metrics']['f1'] for m in all_metrics])
        aggregated['std_aggregate_score'] = np.std([m['aggregate_score'] for m in all_metrics])

        return aggregated

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print formatted evaluation summary"""
        metrics = results['aggregated_metrics']

        print(f"\n📊 Evaluation Summary: {results['architecture']} with {results['model']}")
        print("=" * 60)
        print(f"Total Queries: {metrics.get('total_queries', 0)}")
        print(f"Successful: {metrics.get('successful_queries', 0)}")
        print(f"Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
        print("\n📈 Performance Metrics:")
        print(f"  Precision: {metrics.get('avg_precision', 0):.3f}")
        print(f"  Recall: {metrics.get('avg_recall', 0):.3f}")
        print(f"  F1 Score: {metrics.get('avg_f1', 0):.3f} (±{metrics.get('std_f1', 0):.3f})")
        print(f"  Semantic Similarity: {metrics.get('avg_semantic_similarity', 0):.3f}")
        print(f"  NDCG@10: {metrics.get('avg_ndcg', 0):.3f}")
        print(f"  MAP: {metrics.get('avg_map', 0):.3f}")
        print(f"  Safety Score: {metrics.get('avg_safety_score', 0):.3f}")
        print(f"  Completeness: {metrics.get('avg_completeness', 0):.3f}")
        print(f"  📊 Aggregate Score: {metrics.get('avg_aggregate_score', 0):.3f}")

    def _get_organ_specific_ground_truth(self, drug: str, organ: str) -> List[str]:
        """Get ground truth for organ-specific query (mock implementation)"""
        # In production, load from database
        # This is a simplified mock
        organ_effects = {
            'heart': ['arrhythmia', 'tachycardia', 'chest pain', 'hypertension'],
            'liver': ['hepatotoxicity', 'elevated liver enzymes', 'jaundice'],
            'kidney': ['renal impairment', 'acute kidney injury', 'nephrotoxicity'],
            'lung': ['dyspnea', 'respiratory depression', 'pulmonary edema'],
            'stomach': ['nausea', 'vomiting', 'abdominal pain', 'diarrhea']
        }
        return organ_effects.get(organ, [])

    def _get_comparison_ground_truth(self, drug1: str, drug2: str) -> List[str]:
        """Get ground truth for comparison query (mock implementation)"""
        # In production, load from database
        return ['headache', 'dizziness', 'nausea', 'fatigue']

    def _get_severity_filtered_ground_truth(self, drug: str, severity: str) -> List[str]:
        """Get ground truth for severity-filtered query (mock implementation)"""
        # In production, load from database
        if severity in ['severe', 'life-threatening']:
            return ['anaphylaxis', 'stevens-johnson syndrome', 'hepatic failure']
        return ['mild headache', 'slight nausea']

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")


def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced Complex Query Evaluation')
    parser.add_argument('--architectures', nargs='+',
                       default=['enhanced_format_b', 'enhanced_graphrag', 'advanced_rag_format_b'],
                       help='Architectures to evaluate (enhanced_format_b, enhanced_graphrag, advanced_rag_format_b)')
    parser.add_argument('--models', nargs='+',
                       default=['qwen', 'llama3'],
                       help='Models to use')
    parser.add_argument('--query_types', nargs='+',
                       default=['organ_specific', 'drug_comparison', 'severity_filtered'],
                       help='Query types to test')
    parser.add_argument('--queries_per_type', type=str, default='10',
                       help='Number of queries per type (integer or "all")')
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path')
    parser.add_argument('--output', default='enhanced_evaluation_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Parse queries_per_type
    if args.queries_per_type.lower() == 'all':
        queries_per_type = None  # Will process all available queries
    else:
        queries_per_type = int(args.queries_per_type)

    # Initialize evaluator
    evaluator = EnhancedComplexQueryEvaluator(args.config)

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        architectures=args.architectures,
        models=args.models,
        query_types=args.query_types,
        queries_per_type=queries_per_type
    )

    # Save results
    evaluator.save_results(results, args.output)

    print("\n" + "="*60)
    print("🎯 EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()