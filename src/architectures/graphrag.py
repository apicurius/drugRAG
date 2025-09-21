#!/usr/bin/env python3
"""
GraphRAG Implementation with vLLM
Uses exact Cypher query from notebook: May_Cause_Side_Effect relationship
Supports both binary and complex queries
"""

import json
import logging
import re
from typing import Dict, Any, List
from neo4j import GraphDatabase
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.binary_parser import parse_binary_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAG:
    """GraphRAG using Neo4j + vLLM for reasoning"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize GraphRAG with Neo4j and vLLM

        Args:
            config_path: Path to configuration file
            model: vLLM model ("qwen" or "llama3")
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Neo4j driver with direct bolt connection (bypass routing)
        try:
            # Use direct bolt:// connection to bypass routing discovery issues
            self.driver = GraphDatabase.driver(
                "bolt://9d0e641a.databases.neo4j.io:7687",
                auth=(self.config['neo4j_username'], self.config['neo4j_password']),
                encrypted=True,
                trust="TRUST_ALL_CERTIFICATES",
                max_connection_pool_size=1,
                connection_timeout=30
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            logger.info("✅ Neo4j connection established (direct bolt)")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            raise

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

        logger.info(f"✅ GraphRAG initialized with {model} via vLLM")

    def escape_special_characters(self, input_string: str) -> str:
        """Escape special characters for Neo4j Cypher queries"""
        return re.sub(r"(['\\])", r"\\\1", input_string)

    def binary_query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """
        Binary query using exact Cypher from notebook
        EXACT QUERY: MATCH (s)-[r:May_Cause_Side_Effect]->(t) WHERE s.name = 'drug' AND t.name = 'side_effect' RETURN s, r, t
        """
        if not self.driver:
            return {'answer': 'ERROR', 'confidence': 0.0, 'error': 'No Neo4j connection'}

        # Escape and normalize names (lowercase as per notebook)
        drug_escaped = self.escape_special_characters(drug.lower())
        side_effect_escaped = self.escape_special_characters(side_effect.lower())

        # Cypher query using actual relationship in database
        cypher = f"""
        MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
        WHERE s.name = '{drug_escaped}' AND t.name = '{side_effect_escaped}'
        RETURN s, r, t
        """

        try:
            with self.driver.session() as session:
                cypher_result = session.run(cypher)

                # Check if any results found
                records = list(cypher_result)

                if len(records) > 0:
                    # Direct answer from graph - relationship exists
                    graph_result = f"Yes, the side effect {side_effect} is listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
                    direct_answer = "YES"
                else:
                    # No relationship found
                    graph_result = f"No, the side effect {side_effect} is not listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
                    direct_answer = "NO"

                # Use vLLM for reasoning (following notebook pattern)
                prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the GraphRAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the GraphRAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {side_effect} an adverse effect of {drug}?

### GraphRAG Results:

{graph_result}

FINAL ANSWER: [YES or NO]"""

                # Use temperature=0.1 for RAG deterministic outputs
                llm_response = self.llm.generate_response(prompt, max_tokens=150, temperature=0.1)

                # Use standardized binary parser (notebook-compatible)
                parsed_answer = parse_binary_response(llm_response)

                # Fallback to direct graph result if UNKNOWN
                final_answer = parsed_answer if parsed_answer != 'UNKNOWN' else direct_answer

                return {
                    'answer': final_answer,
                    'confidence': 1.0 if final_answer == direct_answer else 0.8,
                    'drug': drug,
                    'side_effect': side_effect,
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'cypher_query': cypher,
                    'graph_result': graph_result,
                    'llm_reasoning': llm_response[:200]
                }

        except Exception as e:
            logger.error(f"GraphRAG query error: {e}")
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_organ_specific(self, drug: str, organ: str) -> Dict[str, Any]:
        """Complex query: Find all side effects of a drug affecting specific organ system"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug_escaped = self.escape_special_characters(drug.lower())

        # Cypher query to find all side effects of drug affecting specific organ
        cypher = f"""
        MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
        WHERE s.name = '{drug_escaped}' AND (
            toLower(t.name) CONTAINS '{organ.lower()}' OR
            toLower(t.name) CONTAINS 'cardiac' AND '{organ.lower()}' = 'heart' OR
            toLower(t.name) CONTAINS 'hepatic' AND '{organ.lower()}' = 'liver' OR
            toLower(t.name) CONTAINS 'renal' AND '{organ.lower()}' = 'kidney' OR
            toLower(t.name) CONTAINS 'pulmonary' AND '{organ.lower()}' = 'lung' OR
            toLower(t.name) CONTAINS 'gastrointestinal' AND '{organ.lower()}' = 'stomach'
        )
        RETURN t.name AS side_effect
        LIMIT 20
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                side_effects = [record['side_effect'] for record in result]

                return {
                    'drug': drug,
                    'organ': organ,
                    'side_effects_found': side_effects,
                    'count': len(side_effects),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'organ_specific'
                }

        except Exception as e:
            logger.error(f"Complex query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_drug_comparison(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """Complex query: Find common side effects between two drugs"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug1_escaped = self.escape_special_characters(drug1.lower())
        drug2_escaped = self.escape_special_characters(drug2.lower())

        cypher = f"""
        MATCH (s1)-[r1:HAS_SIDE_EFFECT]->(t)<-[r2:HAS_SIDE_EFFECT]-(s2)
        WHERE s1.name = '{drug1_escaped}' AND s2.name = '{drug2_escaped}'
        RETURN t.name AS common_side_effect
        LIMIT 20
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                common_effects = [record['common_side_effect'] for record in result]

                return {
                    'drug1': drug1,
                    'drug2': drug2,
                    'common_effects': common_effects,
                    'count': len(common_effects),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'drug_comparison'
                }

        except Exception as e:
            logger.error(f"Complex query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_reverse_lookup(self, side_effect: str) -> Dict[str, Any]:
        """Complex query: Find all drugs that cause a specific side effect"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        side_effect_escaped = self.escape_special_characters(side_effect.lower())

        cypher = f"""
        MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
        WHERE t.name = '{side_effect_escaped}'
        RETURN s.name AS drug_name
        LIMIT 20
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                drugs = [record['drug_name'] for record in result]

                return {
                    'side_effect': side_effect,
                    'drugs_found': drugs,
                    'count': len(drugs),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'reverse_lookup'
                }

        except Exception as e:
            logger.error(f"Complex query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_severity_filter(self, drug: str, severity: str) -> Dict[str, Any]:
        """Complex query: Find side effects of drug by severity level"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug_escaped = self.escape_special_characters(drug.lower())

        # Define severity keywords
        severity_keywords = {
            'severe': ['severe', 'life threatening', 'fatal', 'death', 'cardiac arrest', 'anaphylactic', 'toxic'],
            'moderate': ['moderate', 'serious', 'significant', 'major'],
            'mild': ['mild', 'minor', 'slight', 'light']
        }

        keywords = severity_keywords.get(severity.lower(), [severity.lower()])

        # Create Cypher query with severity filtering
        cypher = f"""
        MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
        WHERE s.name = '{drug_escaped}' AND (
            {' OR '.join([f"toLower(t.name) CONTAINS '{keyword}'" for keyword in keywords])}
        )
        RETURN t.name AS side_effect
        LIMIT 50
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                side_effects = [record['side_effect'] for record in result]

                return {
                    'drug': drug,
                    'severity': severity,
                    'side_effects_found': side_effects,
                    'count': len(side_effects),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'severity_filtered',
                    'cypher_query': cypher
                }

        except Exception as e:
            logger.error(f"Severity filter query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_all_effects(self, drug: str) -> Dict[str, Any]:
        """Complex query: Find all side effects of a drug (reverse lookup)"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug_escaped = self.escape_special_characters(drug.lower())

        cypher = f"""
        MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
        WHERE s.name = '{drug_escaped}'
        RETURN t.name AS side_effect
        ORDER BY t.name
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                side_effects = [record['side_effect'] for record in result]

                return {
                    'drug': drug,
                    'side_effects_found': side_effects,
                    'count': len(side_effects),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'reverse_lookup',
                    'cypher_query': cypher
                }

        except Exception as e:
            logger.error(f"All effects query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def complex_query_drug_unique(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """Complex query: Find side effects unique to drug1 compared to drug2"""
        if not self.driver:
            return {'error': 'No Neo4j connection'}

        drug1_escaped = self.escape_special_characters(drug1.lower())
        drug2_escaped = self.escape_special_characters(drug2.lower())

        cypher = f"""
        MATCH (s1)-[r1:HAS_SIDE_EFFECT]->(t)
        WHERE s1.name = '{drug1_escaped}'
        AND NOT EXISTS {{
            MATCH (s2)-[r2:HAS_SIDE_EFFECT]->(t)
            WHERE s2.name = '{drug2_escaped}'
        }}
        RETURN t.name AS unique_side_effect
        LIMIT 50
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                unique_effects = [record['unique_side_effect'] for record in result]

                return {
                    'drug1': drug1,
                    'drug2': drug2,
                    'unique_effects': unique_effects,
                    'count': len(unique_effects),
                    'architecture': 'graphrag',
                    'model': f'vllm_{self.model}',
                    'query_type': 'drug_unique_comparison',
                    'cypher_query': cypher
                }

        except Exception as e:
            logger.error(f"Unique comparison query error: {e}")
            return {
                'error': str(e),
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}'
            }

    def query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """Main query method for binary queries (for compatibility)"""
        return self.binary_query(drug, side_effect)

    def query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple binary queries in batch with OPTIMIZED PARALLEL PROCESSING
        Uses concurrent Neo4j queries and batch vLLM inference

        Args:
            queries: List of dicts with 'drug' and 'side_effect' keys

        Returns:
            List of query results
        """
        if not queries:
            return []

        if not self.driver:
            return [{'answer': 'ERROR', 'confidence': 0.0, 'error': 'No Neo4j connection'} for _ in queries]

        logger.info(f"🚀 GRAPHRAG BATCH PROCESSING: {len(queries)} queries with parallel Neo4j + batch vLLM")

        # Step 1: Concurrent Neo4j queries
        logger.info(f"🔍 Running {len(queries)} Neo4j Cypher queries in parallel...")

        def process_single_neo4j_query(idx_query):
            """Process a single Neo4j query"""
            idx, query = idx_query
            drug = query['drug']
            side_effect = query['side_effect']

            # Escape and normalize names
            drug_escaped = self.escape_special_characters(drug.lower())
            side_effect_escaped = self.escape_special_characters(side_effect.lower())

            cypher = f"""
            MATCH (s)-[r:HAS_SIDE_EFFECT]->(t)
            WHERE s.name = '{drug_escaped}' AND t.name = '{side_effect_escaped}'
            RETURN s, r, t
            """

            try:
                with self.driver.session() as session:
                    cypher_result = session.run(cypher)
                    records = list(cypher_result)

                    if len(records) > 0:
                        graph_result = f"Yes, the side effect {side_effect} is listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
                        direct_answer = "YES"
                    else:
                        graph_result = f"No, the side effect {side_effect} is not listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
                        direct_answer = "NO"

                    return idx, {
                        'drug': drug,
                        'side_effect': side_effect,
                        'graph_result': graph_result,
                        'direct_answer': direct_answer,
                        'cypher_query': cypher
                    }

            except Exception as e:
                logger.error(f"Neo4j query failed for {drug}-{side_effect}: {e}")
                return idx, {
                    'drug': drug,
                    'side_effect': side_effect,
                    'graph_result': f"Error querying Neo4j: {e}",
                    'direct_answer': "ERROR",
                    'cypher_query': cypher
                }

        # Use ThreadPoolExecutor for concurrent Neo4j queries
        max_workers = min(10, len(queries))  # Limit concurrent connections to Neo4j
        neo4j_results = [None] * len(queries)  # Pre-allocate to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all Neo4j queries
            future_to_idx = {}
            for i, query in enumerate(queries):
                future = executor.submit(process_single_neo4j_query, (i, query))
                future_to_idx[future] = i

            # Collect Neo4j results with progress bar
            with tqdm(total=len(queries), desc="🔍 Neo4j", unit="query", ncols=100) as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, result = future.result(timeout=30)
                        neo4j_results[idx] = result
                        pbar.update(1)
                    except Exception as e:
                        idx = future_to_idx[future]
                        logger.error(f"Query {idx} failed: {e}")
                        neo4j_results[idx] = {
                            'drug': queries[idx]['drug'],
                            'side_effect': queries[idx]['side_effect'],
                            'graph_result': "Error querying Neo4j",
                            'direct_answer': "ERROR",
                            'cypher_query': ""
                        }
                        pbar.update(1)

        # Step 2: Prepare prompts for batch vLLM processing
        logger.info(f"🧠 Preparing {len(queries)} prompts for batch vLLM inference...")
        prompts = []

        for neo4j_result in neo4j_results:
            prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the GraphRAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the GraphRAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {neo4j_result['side_effect']} an adverse effect of {neo4j_result['drug']}?

### GraphRAG Results:

{neo4j_result['graph_result']}

FINAL ANSWER: [YES or NO]"""
            prompts.append(prompt)

        # Step 3: Batch vLLM inference
        logger.info(f"⚡ Running batch vLLM inference...")
        try:
            # Use temperature=0.1 for RAG deterministic outputs
            responses = self.llm.generate_batch(prompts, max_tokens=150, temperature=0.1)
        except Exception as e:
            logger.error(f"Batch vLLM failed: {e}. Falling back to individual processing.")
            # Fallback to individual processing
            responses = []
            for prompt in prompts:
                try:
                    response = self.llm.generate_response(prompt, max_tokens=150, temperature=0.1)
                    responses.append(response)
                except Exception:
                    responses.append("UNKNOWN")

        # Step 4: Parse results and combine with Neo4j data
        results = []
        for query, neo4j_result, response in zip(queries, neo4j_results, responses):
            # Use standardized binary parser
            parsed_answer = parse_binary_response(response)

            # Fallback to direct graph result if UNKNOWN
            final_answer = parsed_answer if parsed_answer != 'UNKNOWN' else neo4j_result['direct_answer']

            results.append({
                'answer': final_answer,
                'confidence': 1.0 if final_answer == neo4j_result['direct_answer'] else 0.8,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'architecture': 'graphrag',
                'model': f'vllm_{self.model}_batch_optimized',
                'cypher_query': neo4j_result['cypher_query'],
                'graph_result': neo4j_result['graph_result'],
                'llm_reasoning': response[:200],
                'full_response': response
            })

        success_rate = sum(1 for r in results if r['answer'] not in ['ERROR', 'UNKNOWN']) / len(results) * 100
        logger.info(f"✅ GRAPHRAG BATCH COMPLETE: {success_rate:.1f}% successful, {len(results)} total results")

        return results

    def binary_query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Alias for query_batch for consistency"""
        return self.query_batch(queries)

    def __del__(self):
        """Close Neo4j driver on cleanup"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()