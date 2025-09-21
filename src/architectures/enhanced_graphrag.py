#!/usr/bin/env python3
"""
Enhanced GraphRAG with Advanced Cypher Queries and Chain-of-Thought Reasoning
Implements multi-hop traversal, path analysis, and sophisticated graph algorithms
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase
import sys
import os

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.binary_parser import parse_binary_response
from src.utils.query_understanding import QueryUnderstanding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGraphRAG:
    """Enhanced GraphRAG with advanced querying and reasoning capabilities"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize Enhanced GraphRAG with advanced features

        Args:
            config_path: Path to configuration file
            model: vLLM model to use ("qwen" or "llama3")
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Neo4j connection
        try:
            # Use bolt:// protocol which works, instead of neo4j+s:// which fails
            neo4j_host = self.config['neo4j_uri'].replace('neo4j+s://', '').replace('neo4j://', '')
            bolt_uri = f'bolt://{neo4j_host}:7687'

            self.driver = GraphDatabase.driver(
                bolt_uri,
                auth=(self.config['neo4j_username'], self.config['neo4j_password']),
                encrypted=True,
                trust='TRUST_ALL_CERTIFICATES'
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j connection established")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            self.driver = None

        # Initialize vLLM for reasoning
        self.model = model
        if model == "qwen":
            from src.models.vllm_model import VLLMQwenModel
            self.llm = VLLMQwenModel(config_path)
        elif model == "llama3":
            from src.models.vllm_model import VLLMLLAMA3Model
            self.llm = VLLMLLAMA3Model(config_path)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Initialize query understanding module
        self.query_understanding = QueryUnderstanding()

        logger.info(f"✅ Enhanced GraphRAG initialized with {model} and advanced querying")

    def escape_special_characters(self, input_string: str) -> str:
        """Escape special characters for Cypher queries"""
        return re.sub(r"(['\\])", r"\\\1", input_string)

    def enhanced_organ_specific_query(self, drug: str, organ: str) -> Dict[str, Any]:
        """
        Advanced organ-specific query with multi-hop analysis and severity ranking

        Args:
            drug: Drug name
            organ: Organ system

        Returns:
            Comprehensive organ-specific side effects with severity and frequency
        """
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug_escaped = self.escape_special_characters(drug.lower())

        # Get expanded organ terms from query understanding
        organ_terms = self.query_understanding.expand_medical_terms(organ)
        organ_conditions = " OR ".join([f"toLower(t.name) CONTAINS '{term}'" for term in organ_terms])

        # Advanced Cypher query with severity analysis and path traversal
        cypher = f"""
        // Direct effects on organ system
        MATCH (s:Drug)-[r:HAS_SIDE_EFFECT]->(t:SideEffect)
        WHERE s.name = '{drug_escaped}' AND ({organ_conditions})
        WITH t, r, 1 as path_length, 'direct' as effect_type

        UNION

        // Secondary effects that may impact organ system (2-hop)
        MATCH path = (s:Drug)-[:HAS_SIDE_EFFECT*1..2]->(t:SideEffect)
        WHERE s.name = '{drug_escaped}'
        AND ({organ_conditions})
        AND length(path) = 2
        WITH t, relationships(path)[0] as r, length(path) as path_length, 'indirect' as effect_type

        // Aggregate and rank by importance
        WITH t.name as effect_name,
             CASE
                WHEN t.severity = 'severe' THEN 3
                WHEN t.severity = 'moderate' THEN 2
                ELSE 1
             END as severity_score,
             CASE
                WHEN t.frequency = 'common' THEN 3
                WHEN t.frequency = 'uncommon' THEN 2
                ELSE 1
             END as frequency_score,
             effect_type,
             path_length,
             COALESCE(t.clinical_significance, 'standard') as clinical_significance

        // Calculate importance score
        WITH effect_name,
             severity_score,
             frequency_score,
             effect_type,
             path_length,
             clinical_significance,
             (severity_score * 2 + frequency_score) / path_length as importance_score

        ORDER BY importance_score DESC, severity_score DESC
        RETURN effect_name,
               severity_score,
               frequency_score,
               effect_type,
               path_length,
               clinical_significance,
               importance_score
        LIMIT 25
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                effects = []
                for record in result:
                    effects.append({
                        'effect': record['effect_name'],
                        'severity_score': record['severity_score'],
                        'frequency_score': record['frequency_score'],
                        'type': record['effect_type'],
                        'distance': record['path_length'],
                        'clinical_significance': record['clinical_significance'],
                        'importance': round(record['importance_score'], 2)
                    })

                # Use chain-of-thought reasoning to analyze results
                if effects:
                    cot_response = self._apply_chain_of_thought_organ(drug, organ, effects)
                else:
                    cot_response = {'analysis': 'No organ-specific effects found', 'confidence': 'LOW'}

                return {
                    'drug': drug,
                    'organ': organ,
                    'effects': effects[:15],  # Top 15 effects
                    'total_found': len(effects),
                    'analysis': cot_response,
                    'architecture': 'enhanced_graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'enhanced_organ_specific'
                }

        except Exception as e:
            logger.error(f"Enhanced organ query error: {e}")
            return {
                'error': str(e),
                'architecture': 'enhanced_graphrag',
                'model': f'vllm_{self.model}'
            }

    def enhanced_drug_comparison(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """
        Advanced drug comparison with pattern analysis and interaction detection

        Args:
            drug1: First drug name
            drug2: Second drug name

        Returns:
            Detailed comparison with common, unique, and interaction patterns
        """
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug1_escaped = self.escape_special_characters(drug1.lower())
        drug2_escaped = self.escape_special_characters(drug2.lower())

        # Sophisticated comparison query with multiple analyses
        cypher = f"""
        // Get all effects for drug1
        MATCH (d1:Drug)-[:HAS_SIDE_EFFECT]->(e1:SideEffect)
        WHERE d1.name = '{drug1_escaped}'
        WITH collect(DISTINCT e1) as drug1_effects

        // Get all effects for drug2
        MATCH (d2:Drug)-[:HAS_SIDE_EFFECT]->(e2:SideEffect)
        WHERE d2.name = '{drug2_escaped}'
        WITH drug1_effects, collect(DISTINCT e2) as drug2_effects

        // Find common effects
        WITH drug1_effects, drug2_effects,
             [e IN drug1_effects WHERE e IN drug2_effects] as common_effects

        // Find unique effects for each drug
        WITH drug1_effects, drug2_effects, common_effects,
             [e IN drug1_effects WHERE NOT e IN drug2_effects] as unique_to_drug1,
             [e IN drug2_effects WHERE NOT e IN drug1_effects] as unique_to_drug2

        // Analyze common effects with severity
        UNWIND common_effects as common
        WITH drug1_effects, drug2_effects, unique_to_drug1, unique_to_drug2,
             collect({{
                name: common.name,
                severity: COALESCE(common.severity, 'unknown'),
                frequency: COALESCE(common.frequency, 'unknown')
             }}) as common_detailed

        // Return comprehensive comparison
        RETURN
            size(drug1_effects) as drug1_total,
            size(drug2_effects) as drug2_total,
            size(common_detailed) as common_count,
            common_detailed[0..10] as top_common_effects,
            [e IN unique_to_drug1 | e.name][0..10] as drug1_unique_sample,
            [e IN unique_to_drug2 | e.name][0..10] as drug2_unique_sample,
            size(unique_to_drug1) as drug1_unique_count,
            size(unique_to_drug2) as drug2_unique_count
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                record = result.single()

                if record:
                    comparison_data = {
                        'drug1': drug1,
                        'drug2': drug2,
                        'drug1_total_effects': record['drug1_total'],
                        'drug2_total_effects': record['drug2_total'],
                        'common_effects': record['top_common_effects'] or [],
                        'common_count': record['common_count'],
                        'drug1_unique': record['drug1_unique_sample'] or [],
                        'drug1_unique_count': record['drug1_unique_count'],
                        'drug2_unique': record['drug2_unique_sample'] or [],
                        'drug2_unique_count': record['drug2_unique_count'],
                        'similarity_score': self._calculate_similarity_score(
                            record['common_count'],
                            record['drug1_total'],
                            record['drug2_total']
                        )
                    }

                    # Apply chain-of-thought analysis
                    cot_analysis = self._apply_chain_of_thought_comparison(comparison_data)
                    comparison_data['analysis'] = cot_analysis

                    return {
                        **comparison_data,
                        'architecture': 'enhanced_graphrag',
                        'model': f'vllm_{self.model}',
                        'query_type': 'enhanced_comparison'
                    }

                return {
                    'error': 'No data found for comparison',
                    'architecture': 'enhanced_graphrag',
                    'model': f'vllm_{self.model}'
                }

        except Exception as e:
            logger.error(f"Enhanced comparison error: {e}")
            return {
                'error': str(e),
                'architecture': 'enhanced_graphrag',
                'model': f'vllm_{self.model}'
            }

    def multi_drug_interaction_pattern(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Analyze interaction patterns across multiple drugs

        Args:
            drugs: List of drug names

        Returns:
            Interaction patterns and risk analysis
        """
        if not self.driver or len(drugs) < 2:
            return {'error': 'Need at least 2 drugs and Neo4j connection'}

        drug_list_escaped = [self.escape_special_characters(d.lower()) for d in drugs]
        drug_conditions = ", ".join([f"'{d}'" for d in drug_list_escaped])

        cypher = f"""
        // Find effects shared by multiple drugs
        MATCH (d:Drug)-[:HAS_SIDE_EFFECT]->(e:SideEffect)
        WHERE d.name IN [{drug_conditions}]
        WITH e, collect(DISTINCT d.name) as drugs_with_effect, count(DISTINCT d) as drug_count
        WHERE drug_count >= 2

        // Calculate risk score based on severity and drug count
        WITH e.name as effect,
             drugs_with_effect,
             drug_count,
             CASE
                WHEN e.severity = 'severe' THEN drug_count * 3
                WHEN e.severity = 'moderate' THEN drug_count * 2
                ELSE drug_count
             END as risk_score

        ORDER BY risk_score DESC, drug_count DESC
        RETURN effect,
               drugs_with_effect,
               drug_count,
               risk_score
        LIMIT 20
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                interaction_patterns = []

                for record in result:
                    interaction_patterns.append({
                        'effect': record['effect'],
                        'drugs_affected': record['drugs_with_effect'],
                        'drug_count': record['drug_count'],
                        'risk_score': record['risk_score'],
                        'interaction_level': self._classify_interaction_level(
                            record['drug_count'], len(drugs)
                        )
                    })

                return {
                    'drugs_analyzed': drugs,
                    'interaction_patterns': interaction_patterns,
                    'high_risk_count': sum(1 for p in interaction_patterns if p['risk_score'] >= 6),
                    'architecture': 'enhanced_graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'multi_drug_interaction'
                }

        except Exception as e:
            logger.error(f"Multi-drug interaction error: {e}")
            return {'error': str(e)}

    def _calculate_similarity_score(self, common: int, total1: int, total2: int) -> float:
        """Calculate Jaccard similarity between drug effect profiles"""
        if total1 == 0 or total2 == 0:
            return 0.0
        union = total1 + total2 - common
        return round(common / union if union > 0 else 0, 3)

    def _classify_interaction_level(self, affected_drugs: int, total_drugs: int) -> str:
        """Classify interaction level based on drug overlap"""
        ratio = affected_drugs / total_drugs
        if ratio >= 0.75:
            return 'high'
        elif ratio >= 0.5:
            return 'moderate'
        else:
            return 'low'

    def _apply_chain_of_thought_organ(self, drug: str, organ: str, effects: List[Dict]) -> Dict[str, Any]:
        """Apply chain-of-thought reasoning to organ-specific results"""

        # Separate effects by severity
        severe_effects = [e for e in effects if e['severity_score'] == 3]
        moderate_effects = [e for e in effects if e['severity_score'] == 2]
        direct_effects = [e for e in effects if e['type'] == 'direct']

        prompt = f"""Analyze organ-specific side effects using medical reasoning:

Drug: {drug}
Organ System: {organ}

Evidence Summary:
- Total effects found: {len(effects)}
- Severe effects: {len(severe_effects)}
- Direct effects: {len(direct_effects)}
- Top effects by importance: {', '.join([e['effect'] for e in effects[:5]])}

Step-by-step Analysis:
1. Clinical Significance: Which effects are most clinically relevant for {organ} system?
2. Risk Assessment: What is the overall risk to {organ} function?
3. Monitoring Recommendations: What should be monitored?

Provide a concise medical assessment:"""

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            response = self.llm.generate_response(prompt, max_tokens=200, temperature=0.1)
            return {
                'assessment': response,
                'confidence': 'HIGH' if len(effects) > 5 else 'MEDIUM',
                'severity_profile': {
                    'severe': len(severe_effects),
                    'moderate': len(moderate_effects),
                    'mild': len(effects) - len(severe_effects) - len(moderate_effects)
                }
            }
        except Exception as e:
            logger.error(f"Chain-of-thought reasoning error: {e}")
            return {'assessment': 'Analysis failed', 'confidence': 'LOW'}

    def _apply_chain_of_thought_comparison(self, comparison_data: Dict) -> Dict[str, Any]:
        """Apply chain-of-thought reasoning to drug comparison"""

        prompt = f"""Analyze drug comparison using systematic reasoning:

Comparing: {comparison_data['drug1']} vs {comparison_data['drug2']}

Statistical Summary:
- {comparison_data['drug1']}: {comparison_data['drug1_total_effects']} total effects
- {comparison_data['drug2']}: {comparison_data['drug2_total_effects']} total effects
- Common effects: {comparison_data['common_count']} ({comparison_data.get('similarity_score', 0)*100:.1f}% similarity)

Common Severe Effects:
{', '.join([e['name'] for e in comparison_data['common_effects'] if e.get('severity') == 'severe'][:5]) or 'None identified'}

Analysis Framework:
1. Safety Profile Comparison: Which drug has a better safety profile?
2. Shared Risks: What are the major shared risks?
3. Unique Concerns: What unique risks does each drug pose?
4. Clinical Decision: Key factors for choosing between these drugs?

Provide clinical insights:"""

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            response = self.llm.generate_response(prompt, max_tokens=250, temperature=0.1)
            return {
                'clinical_comparison': response,
                'similarity_interpretation': self._interpret_similarity(
                    comparison_data.get('similarity_score', 0)
                ),
                'recommendation_confidence': 'HIGH' if comparison_data['common_count'] > 10 else 'MEDIUM'
            }
        except Exception as e:
            logger.error(f"Comparison analysis error: {e}")
            return {'clinical_comparison': 'Analysis failed', 'recommendation_confidence': 'LOW'}

    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score"""
        if score > 0.7:
            return 'Very similar side effect profiles'
        elif score > 0.4:
            return 'Moderately similar profiles'
        elif score > 0.2:
            return 'Some overlap in effects'
        else:
            return 'Distinct side effect profiles'

    def process_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Process any complex query using query understanding and appropriate strategy

        Args:
            query: Natural language complex query

        Returns:
            Comprehensive response based on query type
        """
        # Analyze the query
        analysis = self.query_understanding.analyze_query(query)
        query_type = analysis['query_type']
        entities = analysis['entities']

        logger.info(f"Processing {query_type} query with entities: {entities}")

        # Route to appropriate handler based on query type
        if query_type == 'drug_comparison_common' and len(entities['drugs']) >= 2:
            return self.enhanced_drug_comparison(entities['drugs'][0], entities['drugs'][1])

        elif query_type == 'organ_specific' and entities['drugs'] and entities['organs']:
            return self.enhanced_organ_specific_query(entities['drugs'][0], entities['organs'][0])

        elif query_type == 'drug_comparison' and len(entities['drugs']) >= 2:
            return self.multi_drug_interaction_pattern(entities['drugs'])

        else:
            # Fallback to general complex processing
            return {
                'query': query,
                'analysis': analysis,
                'error': 'Query type not fully supported yet',
                'suggestion': 'Please use specific comparison or organ-specific format'
            }