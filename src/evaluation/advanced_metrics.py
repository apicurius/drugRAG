#!/usr/bin/env python3
"""
Advanced Evaluation Metrics for Complex Queries
Includes semantic similarity, ranking metrics, and clinical relevance scoring
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import ndcg_score
import logging
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Advanced metrics for evaluating complex query responses"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with sentence transformer for semantic similarity

        Args:
            model_name: Name of sentence transformer model
        """
        try:
            self.semantic_model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded semantic model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load semantic model: {e}. Falling back to string matching.")
            self.semantic_model = None

        # Medical term importance weights
        self.severity_weights = {
            'fatal': 5.0,
            'life-threatening': 4.5,
            'severe': 4.0,
            'serious': 3.5,
            'moderate': 2.0,
            'mild': 1.0,
            'minor': 0.5
        }

        # Organ system criticality scores
        self.organ_criticality = {
            'cardiac': 5.0,
            'heart': 5.0,
            'respiratory': 4.5,
            'lung': 4.5,
            'renal': 4.0,
            'kidney': 4.0,
            'hepatic': 4.0,
            'liver': 4.0,
            'neurological': 3.5,
            'brain': 3.5,
            'gastrointestinal': 2.5,
            'skin': 1.5
        }

    def semantic_similarity_score(self, predicted: List[str], ground_truth: List[str]) -> float:
        """
        Calculate semantic similarity between predicted and ground truth lists

        Args:
            predicted: List of predicted effects
            ground_truth: List of ground truth effects

        Returns:
            Semantic similarity score (0-1)
        """
        if not predicted or not ground_truth:
            return 0.0

        if not self.semantic_model:
            # Fallback to Jaccard similarity
            return self._jaccard_similarity(predicted, ground_truth)

        try:
            # Encode all effects
            pred_embeddings = self.semantic_model.encode(predicted)
            truth_embeddings = self.semantic_model.encode(ground_truth)

            # Calculate pairwise similarities
            similarity_matrix = np.zeros((len(predicted), len(ground_truth)))

            for i, pred_emb in enumerate(pred_embeddings):
                for j, truth_emb in enumerate(truth_embeddings):
                    similarity = 1 - cosine(pred_emb, truth_emb)
                    similarity_matrix[i, j] = similarity

            # For each predicted, find best match in ground truth
            pred_matches = np.max(similarity_matrix, axis=1)
            # For each ground truth, find best match in predicted
            truth_matches = np.max(similarity_matrix, axis=0)

            # Calculate F1-like score
            precision = np.mean(pred_matches) if len(pred_matches) > 0 else 0
            recall = np.mean(truth_matches) if len(truth_matches) > 0 else 0

            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            return float(f1)

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return self._jaccard_similarity(predicted, ground_truth)

    def hierarchical_precision_recall(self, predicted: List[str], ground_truth: List[str],
                                     similarity_threshold: float = 0.7) -> Tuple[float, float, float]:
        """
        Calculate precision/recall giving partial credit for semantically similar terms

        Args:
            predicted: List of predicted effects
            ground_truth: List of ground truth effects
            similarity_threshold: Minimum similarity for partial credit

        Returns:
            Tuple of (precision, recall, f1)
        """
        if not predicted and not ground_truth:
            return 1.0, 1.0, 1.0

        if not predicted or not ground_truth:
            return 0.0, 0.0, 0.0

        # Create similarity matrix
        matches_matrix = self._calculate_similarity_matrix(predicted, ground_truth)

        # Apply threshold
        matches_matrix[matches_matrix < similarity_threshold] = 0

        # Calculate hierarchical scores
        # Precision: For each predicted, what's the best match in ground truth?
        pred_scores = np.max(matches_matrix, axis=1)
        precision = np.mean(pred_scores) if len(pred_scores) > 0 else 0

        # Recall: For each ground truth, what's the best match in predicted?
        truth_scores = np.max(matches_matrix, axis=0)
        recall = np.mean(truth_scores) if len(truth_scores) > 0 else 0

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return float(precision), float(recall), float(f1)

    def ndcg_at_k(self, predicted: List[str], ground_truth: List[str],
                  relevance_scores: Optional[Dict[str, float]] = None, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k

        Args:
            predicted: Ranked list of predicted effects
            ground_truth: List of ground truth effects
            relevance_scores: Optional relevance scores for each effect
            k: Cutoff position

        Returns:
            NDCG@k score
        """
        if not predicted or not ground_truth:
            return 0.0

        # Limit to top k
        predicted_k = predicted[:k]

        # Create relevance scores if not provided
        if not relevance_scores:
            relevance_scores = {}
            for effect in ground_truth:
                effect_lower = effect.lower()
                # Assign higher scores to more severe effects
                base_score = 1.0
                for term, weight in self.severity_weights.items():
                    if term in effect_lower:
                        base_score = weight
                        break
                relevance_scores[effect] = base_score

        # Build relevance vector for predicted items
        y_true = []
        for effect in predicted_k:
            if effect in relevance_scores:
                y_true.append(relevance_scores[effect])
            else:
                # Check for semantic similarity
                max_sim = 0
                for gt_effect in ground_truth:
                    sim = self._string_similarity(effect, gt_effect)
                    if sim > max_sim:
                        max_sim = sim
                y_true.append(max_sim * relevance_scores.get(gt_effect, 1.0) if max_sim > 0.7 else 0)

        # Ideal ranking (sorted by relevance)
        y_ideal = sorted(y_true, reverse=True)

        if sum(y_ideal) == 0:
            return 0.0

        # Calculate NDCG
        try:
            ndcg = ndcg_score([y_ideal], [y_true])
            return float(ndcg)
        except Exception as e:
            logger.error(f"NDCG calculation failed: {e}")
            return 0.0

    def mean_average_precision(self, predicted: List[str], ground_truth: List[str]) -> float:
        """
        Calculate Mean Average Precision for ranked list

        Args:
            predicted: Ranked list of predicted effects
            ground_truth: List of ground truth effects

        Returns:
            MAP score
        """
        if not ground_truth:
            return 0.0

        ground_truth_set = set([gt.lower() for gt in ground_truth])
        num_relevant = 0
        sum_precision = 0

        for i, pred in enumerate(predicted, 1):
            if pred.lower() in ground_truth_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precision += precision_at_i

        if num_relevant == 0:
            return 0.0

        return sum_precision / len(ground_truth_set)

    def clinical_relevance_score(self, predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate clinical relevance metrics with severity weighting

        Args:
            predicted: List of predicted effects
            ground_truth: List of ground truth effects with severity

        Returns:
            Dictionary of clinical metrics
        """
        # Separate effects by severity
        severe_truth = [e for e in ground_truth if any(s in e.lower() for s in ['severe', 'serious', 'fatal'])]
        moderate_truth = [e for e in ground_truth if any(s in e.lower() for s in ['moderate', 'significant'])]
        mild_truth = [e for e in ground_truth if any(s in e.lower() for s in ['mild', 'minor'])]

        severe_pred = [e for e in predicted if any(s in e.lower() for s in ['severe', 'serious', 'fatal'])]
        moderate_pred = [e for e in predicted if any(s in e.lower() for s in ['moderate', 'significant'])]
        mild_pred = [e for e in predicted if any(s in e.lower() for s in ['mild', 'minor'])]

        # Calculate weighted accuracy
        severe_score = self._calculate_overlap(severe_pred, severe_truth) * 3.0
        moderate_score = self._calculate_overlap(moderate_pred, moderate_truth) * 2.0
        mild_score = self._calculate_overlap(mild_pred, mild_truth) * 1.0

        total_weight = 3.0 * len(severe_truth) + 2.0 * len(moderate_truth) + 1.0 * len(mild_truth)
        weighted_score = (severe_score + moderate_score + mild_score) / total_weight if total_weight > 0 else 0

        # Safety score (penalty for missing severe effects)
        missed_severe = len(severe_truth) - len(set(severe_pred) & set(severe_truth))
        safety_penalty = missed_severe * 0.2  # 20% penalty per missed severe effect
        safety_score = max(0, 1.0 - safety_penalty)

        # Completeness score
        completeness = len(set(predicted) & set(ground_truth)) / len(ground_truth) if ground_truth else 0

        return {
            'weighted_accuracy': weighted_score,
            'safety_score': safety_score,
            'completeness': completeness,
            'severe_recall': self._calculate_overlap(severe_pred, severe_truth) if severe_truth else 1.0,
            'moderate_recall': self._calculate_overlap(moderate_pred, moderate_truth) if moderate_truth else 1.0,
            'mild_recall': self._calculate_overlap(mild_pred, mild_truth) if mild_truth else 1.0
        }

    def ranking_correlation(self, predicted_ranking: List[str], ground_truth_ranking: List[str]) -> float:
        """
        Calculate Spearman rank correlation between predicted and ground truth rankings

        Args:
            predicted_ranking: Ranked list of predicted effects
            ground_truth_ranking: Ranked list of ground truth effects

        Returns:
            Spearman correlation (-1 to 1)
        """
        if not predicted_ranking or not ground_truth_ranking:
            return 0.0

        # Find common elements
        common = set(predicted_ranking) & set(ground_truth_ranking)
        if len(common) < 2:
            return 0.0

        # Get ranks for common elements
        pred_ranks = {item: i for i, item in enumerate(predicted_ranking)}
        truth_ranks = {item: i for i, item in enumerate(ground_truth_ranking)}

        # Calculate Spearman correlation
        n = len(common)
        sum_d_squared = sum((pred_ranks[item] - truth_ranks[item]) ** 2 for item in common)
        correlation = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))

        return float(correlation)

    def comprehensive_evaluation(self, predicted: List[str], ground_truth: List[str],
                                query_type: str = "general") -> Dict[str, Any]:
        """
        Perform comprehensive evaluation with all metrics

        Args:
            predicted: List of predicted effects
            ground_truth: List of ground truth effects
            query_type: Type of query for context

        Returns:
            Dictionary with all evaluation metrics
        """
        # Basic metrics
        precision, recall, f1 = self.hierarchical_precision_recall(predicted, ground_truth)

        # Semantic metrics
        semantic_sim = self.semantic_similarity_score(predicted, ground_truth)

        # Ranking metrics
        ndcg = self.ndcg_at_k(predicted, ground_truth)
        map_score = self.mean_average_precision(predicted, ground_truth)

        # Clinical metrics
        clinical_scores = self.clinical_relevance_score(predicted, ground_truth)

        # Aggregate score (weighted combination)
        aggregate_score = (
            f1 * 0.3 +
            semantic_sim * 0.2 +
            ndcg * 0.2 +
            clinical_scores['safety_score'] * 0.2 +
            clinical_scores['completeness'] * 0.1
        )

        return {
            'query_type': query_type,
            'basic_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'semantic_metrics': {
                'similarity': semantic_sim
            },
            'ranking_metrics': {
                'ndcg@10': ndcg,
                'map': map_score
            },
            'clinical_metrics': clinical_scores,
            'aggregate_score': aggregate_score,
            'interpretation': self._interpret_scores(aggregate_score)
        }

    def _calculate_similarity_matrix(self, list1: List[str], list2: List[str]) -> np.ndarray:
        """Calculate pairwise similarity matrix between two lists"""
        matrix = np.zeros((len(list1), len(list2)))

        for i, item1 in enumerate(list1):
            for j, item2 in enumerate(list2):
                matrix[i, j] = self._string_similarity(item1, item2)

        return matrix

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simple Jaccard for words)"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _jaccard_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate Jaccard similarity between two lists"""
        set1 = set([item.lower() for item in list1])
        set2 = set([item.lower() for item in list2])

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_overlap(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate overlap ratio"""
        if not ground_truth:
            return 1.0 if not predicted else 0.0

        pred_set = set([p.lower() for p in predicted])
        truth_set = set([g.lower() for g in ground_truth])

        return len(pred_set & truth_set) / len(truth_set)

    def _interpret_scores(self, score: float) -> str:
        """Interpret aggregate score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"