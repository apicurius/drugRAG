#!/usr/bin/env python3
"""
Advanced RAG Format B with Hierarchical Retrieval and Chain-of-Thought Reasoning
Implements multi-stage retrieval, query decomposition, and sophisticated prompting
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pinecone import Pinecone
import numpy as np
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.embedding_client import create_embedding_client
from src.utils.token_manager import create_token_manager
from src.utils.binary_parser import parse_binary_response
from src.utils.query_understanding import QueryUnderstanding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRAGFormatB:
    """Advanced Format B with hierarchical retrieval and enhanced reasoning"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize Advanced RAG Format B

        Args:
            config_path: Path to configuration file
            model: vLLM model ("qwen" or "llama3")
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Pinecone for retrieval
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.namespace = "drug-side-effects-formatB"

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
            raise ValueError(f"Unknown model: {model}")

        # Initialize token manager for context truncation
        self.token_manager = create_token_manager(model_type=model)

        # Initialize query understanding module
        self.query_understanding = QueryUnderstanding()

        logger.info(f"✅ Advanced RAG Format B initialized with {model} and hierarchical retrieval")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using robust client"""
        return self.embedding_client.get_embedding(text)

    def hierarchical_retrieval(self, query: str, drug: str, organ: str = None,
                             severity: str = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Perform hierarchical retrieval with coarse-to-fine strategy

        Args:
            query: Search query
            drug: Drug name
            organ: Optional organ system filter
            severity: Optional severity filter

        Returns:
            Tuple of (retrieved documents, retrieval metadata)
        """
        retrieval_metadata = {
            'stages': [],
            'total_retrieved': 0,
            'filtering_applied': []
        }

        # Stage 1: Broad retrieval
        stage1_query = f"{drug} side effects adverse events"
        stage1_embedding = self.get_embedding(stage1_query)

        if not stage1_embedding:
            return [], {'error': 'Failed to generate embedding'}

        stage1_results = self.index.query(
            vector=stage1_embedding,
            top_k=50,  # Retrieve more initially
            namespace=self.namespace,
            include_metadata=True
        )

        retrieval_metadata['stages'].append({
            'stage': 1,
            'query': stage1_query,
            'retrieved': len(stage1_results.matches)
        })

        # Stage 2: Filter by drug name (exact match)
        stage2_results = []
        for match in stage1_results.matches:
            if match.metadata and match.score > 0.5:
                pair_drug = match.metadata.get('drug', '').lower()
                if drug.lower() in pair_drug or pair_drug in drug.lower():
                    stage2_results.append(match)

        retrieval_metadata['stages'].append({
            'stage': 2,
            'filter': 'drug_name',
            'retained': len(stage2_results)
        })

        # Stage 3: Apply organ system filter if specified
        if organ:
            organ_terms = self.query_understanding.expand_medical_terms(organ)
            stage3_results = []
            for match in stage2_results:
                effect = match.metadata.get('side_effect', '').lower()
                if any(term in effect for term in organ_terms):
                    stage3_results.append(match)

            retrieval_metadata['filtering_applied'].append(f'organ:{organ}')
            retrieval_metadata['stages'].append({
                'stage': 3,
                'filter': 'organ_system',
                'retained': len(stage3_results)
            })
        else:
            stage3_results = stage2_results

        # Stage 4: Apply severity filter if specified
        if severity:
            severity_terms = {
                'severe': ['severe', 'serious', 'life-threatening', 'fatal', 'critical'],
                'moderate': ['moderate', 'significant'],
                'mild': ['mild', 'minor', 'slight']
            }

            final_results = []
            for match in stage3_results:
                effect = match.metadata.get('side_effect', '').lower()
                if any(term in effect for term in severity_terms.get(severity, [])):
                    final_results.append(match)

            retrieval_metadata['filtering_applied'].append(f'severity:{severity}')
            retrieval_metadata['stages'].append({
                'stage': 4,
                'filter': 'severity',
                'retained': len(final_results)
            })
        else:
            final_results = stage3_results

        # Convert to document format
        documents = []
        for match in final_results[:20]:  # Limit final results
            documents.append({
                'drug': match.metadata.get('drug', ''),
                'side_effect': match.metadata.get('side_effect', ''),
                'score': match.score,
                'text': match.metadata.get('text', '')
            })

        retrieval_metadata['total_retrieved'] = len(documents)
        return documents, retrieval_metadata

    def multi_query_retrieval(self, queries: List[str]) -> List[Dict]:
        """
        Perform parallel retrieval for multiple sub-queries and merge results

        Args:
            queries: List of sub-queries

        Returns:
            Merged and deduplicated results
        """
        all_results = []
        seen_pairs = set()

        # Parallel retrieval
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            future_to_query = {
                executor.submit(self._single_query_retrieval, q): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    for result in results:
                        # Deduplicate by drug-effect pair
                        pair_key = f"{result.get('drug', '')}_{result.get('side_effect', '')}"
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            all_results.append(result)
                except Exception as e:
                    logger.error(f"Query '{query}' failed: {e}")

        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:30]  # Return top 30 unique results

    def _single_query_retrieval(self, query: str) -> List[Dict]:
        """Helper for single query retrieval"""
        embedding = self.get_embedding(query)
        if not embedding:
            return []

        results = self.index.query(
            vector=embedding,
            top_k=10,
            namespace=self.namespace,
            include_metadata=True
        )

        documents = []
        for match in results.matches:
            if match.metadata and match.score > 0.5:
                documents.append({
                    'drug': match.metadata.get('drug', ''),
                    'side_effect': match.metadata.get('side_effect', ''),
                    'score': match.score,
                    'text': match.metadata.get('text', '')
                })

        return documents

    def enhanced_complex_organ_query(self, drug: str, organ: str) -> Dict[str, Any]:
        """
        Enhanced organ-specific query with hierarchical retrieval and CoT reasoning

        Args:
            drug: Drug name
            organ: Organ system

        Returns:
            Comprehensive organ-specific analysis
        """
        # Analyze query for better understanding
        query_analysis = self.query_understanding.analyze_query(
            f"What are the {organ} side effects of {drug}?"
        )

        # Perform hierarchical retrieval
        documents, retrieval_metadata = self.hierarchical_retrieval(
            query=f"{drug} {organ} effects",
            drug=drug,
            organ=organ
        )

        # If hierarchical retrieval yields few results, try sub-queries
        if len(documents) < 5:
            sub_queries = query_analysis['sub_queries']
            additional_docs = self.multi_query_retrieval(sub_queries)
            documents.extend(additional_docs)

        # Extract and categorize effects
        organ_effects = self._categorize_effects_by_severity(documents)

        # Generate chain-of-thought analysis
        cot_prompt = self._generate_cot_prompt_organ(drug, organ, organ_effects, documents)

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            llm_response = self.llm.generate_response(cot_prompt, max_tokens=300, temperature=0.1)

            # Parse structured response
            structured_response = self._parse_structured_response(llm_response)

            return {
                'drug': drug,
                'organ': organ,
                'effects_by_severity': organ_effects,
                'total_effects': sum(len(effects) for effects in organ_effects.values()),
                'analysis': structured_response,
                'retrieval_metadata': retrieval_metadata,
                'confidence': self._calculate_confidence(len(documents)),
                'architecture': 'advanced_rag_format_b',
                'model': f'vllm_{self.model}',
                'query_type': 'hierarchical_organ_specific'
            }

        except Exception as e:
            logger.error(f"Enhanced organ query error: {e}")
            return {
                'drug': drug,
                'organ': organ,
                'error': str(e),
                'architecture': 'advanced_rag_format_b',
                'model': f'vllm_{self.model}'
            }

    def enhanced_complex_comparison(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """
        Enhanced drug comparison with multi-query retrieval and CoT reasoning

        Args:
            drug1: First drug
            drug2: Second drug

        Returns:
            Detailed comparison with clinical insights
        """
        # Generate sub-queries for comprehensive retrieval
        sub_queries = [
            f"{drug1} side effects",
            f"{drug2} side effects",
            f"{drug1} severe adverse events",
            f"{drug2} severe adverse events",
            f"{drug1} {drug2} common effects",
            f"{drug1} {drug2} drug interaction"
        ]

        # Multi-query retrieval
        all_documents = self.multi_query_retrieval(sub_queries)

        # Separate effects by drug
        drug1_effects = set()
        drug2_effects = set()

        for doc in all_documents:
            drug = doc.get('drug', '').lower()
            effect = doc.get('side_effect', '').lower()

            if drug1.lower() in drug:
                drug1_effects.add(effect)
            if drug2.lower() in drug:
                drug2_effects.add(effect)

        # Calculate comparison metrics
        common_effects = list(drug1_effects.intersection(drug2_effects))
        drug1_unique = list(drug1_effects - drug2_effects)
        drug2_unique = list(drug2_effects - drug1_effects)

        # Generate chain-of-thought comparison
        cot_prompt = self._generate_cot_prompt_comparison(
            drug1, drug2, common_effects, drug1_unique, drug2_unique
        )

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            llm_response = self.llm.generate_response(cot_prompt, max_tokens=350, temperature=0.1)

            return {
                'drug1': drug1,
                'drug2': drug2,
                'common_effects': common_effects[:15],
                'common_count': len(common_effects),
                'drug1_unique': drug1_unique[:10],
                'drug1_unique_count': len(drug1_unique),
                'drug2_unique': drug2_unique[:10],
                'drug2_unique_count': len(drug2_unique),
                'similarity_score': len(common_effects) / (len(drug1_effects) + len(drug2_effects) - len(common_effects))
                                  if (drug1_effects or drug2_effects) else 0,
                'clinical_analysis': llm_response,
                'confidence': self._calculate_confidence(len(all_documents)),
                'architecture': 'advanced_rag_format_b',
                'model': f'vllm_{self.model}',
                'query_type': 'multi_query_comparison'
            }

        except Exception as e:
            logger.error(f"Enhanced comparison error: {e}")
            return {
                'error': str(e),
                'architecture': 'advanced_rag_format_b',
                'model': f'vllm_{self.model}'
            }

    def _categorize_effects_by_severity(self, documents: List[Dict]) -> Dict[str, List[str]]:
        """Categorize effects by severity level"""
        categories = {
            'severe': [],
            'moderate': [],
            'mild': [],
            'unclassified': []
        }

        severity_keywords = {
            'severe': ['severe', 'serious', 'life-threatening', 'fatal', 'critical', 'emergency'],
            'moderate': ['moderate', 'significant', 'persistent', 'bothersome'],
            'mild': ['mild', 'minor', 'slight', 'temporary', 'transient']
        }

        for doc in documents:
            effect = doc.get('side_effect', '')
            effect_lower = effect.lower()

            categorized = False
            for severity, keywords in severity_keywords.items():
                if any(keyword in effect_lower for keyword in keywords):
                    if effect not in categories[severity]:
                        categories[severity].append(effect)
                    categorized = True
                    break

            if not categorized and effect not in categories['unclassified']:
                categories['unclassified'].append(effect)

        return categories

    def _generate_cot_prompt_organ(self, drug: str, organ: str,
                                  effects: Dict, documents: List[Dict]) -> str:
        """Generate chain-of-thought prompt for organ-specific query"""

        context = "\n".join([f"• {doc['side_effect']}" for doc in documents[:15]])

        return f"""Analyze {organ} system effects of {drug} using systematic medical reasoning:

RETRIEVED EVIDENCE:
{context}

CATEGORIZED EFFECTS:
- Severe: {', '.join(effects.get('severe', [])) or 'None identified'}
- Moderate: {', '.join(effects.get('moderate', [])) or 'None identified'}
- Mild: {', '.join(effects.get('mild', [])) or 'None identified'}

STEP-BY-STEP ANALYSIS:

1. ORGAN SYSTEM IMPACT
   - Direct effects on {organ} function
   - Secondary/systemic effects affecting {organ}
   - Mechanism of action relevance

2. CLINICAL SIGNIFICANCE
   - Most concerning effects for patients
   - Frequency and reversibility
   - Monitoring requirements

3. RISK STRATIFICATION
   - High-risk patient populations
   - Contraindications related to {organ} disease
   - Drug-drug interactions affecting {organ}

4. MANAGEMENT RECOMMENDATIONS
   - Preventive measures
   - Early detection strategies
   - When to discontinue therapy

STRUCTURED SUMMARY:
Provide a concise clinical assessment focusing on actionable information."""

    def _generate_cot_prompt_comparison(self, drug1: str, drug2: str,
                                       common: List[str], unique1: List[str],
                                       unique2: List[str]) -> str:
        """Generate chain-of-thought prompt for drug comparison"""

        return f"""Compare safety profiles of {drug1} vs {drug2} using systematic analysis:

EVIDENCE SUMMARY:
- Common effects ({len(common)}): {', '.join(common[:8]) if common else 'None'}
- {drug1} unique ({len(unique1)}): {', '.join(unique1[:5]) if unique1 else 'None'}
- {drug2} unique ({len(unique2)}): {', '.join(unique2[:5]) if unique2 else 'None'}

COMPARATIVE ANALYSIS FRAMEWORK:

1. SHARED RISK PROFILE
   - Major risks common to both drugs
   - Severity of shared effects
   - Class effects vs drug-specific

2. DISTINCTIVE SAFETY CONCERNS
   - Key differences in side effect profiles
   - Unique severe effects for each drug
   - Organ system preferences

3. PATIENT SELECTION CRITERIA
   - Which patients better suited for {drug1}?
   - Which patients better suited for {drug2}?
   - Absolute contraindications

4. CLINICAL DECISION FACTORS
   - Primary safety considerations
   - Risk-benefit for specific conditions
   - Monitoring requirements comparison

RECOMMENDATION:
Provide evidence-based guidance for drug selection based on safety profiles."""

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        # Simple parsing - in production, use more sophisticated parsing
        lines = response.split('\n')

        structured = {
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'confidence': 'MEDIUM'
        }

        current_section = 'summary'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if 'FINDING' in line.upper() or 'KEY' in line.upper():
                current_section = 'findings'
            elif 'RECOMMEND' in line.upper():
                current_section = 'recommendations'
            elif line.startswith('•') or line.startswith('-'):
                if current_section == 'findings':
                    structured['key_findings'].append(line[1:].strip())
                elif current_section == 'recommendations':
                    structured['recommendations'].append(line[1:].strip())
            else:
                if current_section == 'summary' and not structured['summary']:
                    structured['summary'] = line

        return structured

    def _calculate_confidence(self, doc_count: int) -> str:
        """Calculate confidence based on evidence"""
        if doc_count >= 15:
            return 'HIGH'
        elif doc_count >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def process_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Process any complex query using advanced retrieval and reasoning

        Args:
            query: Natural language query

        Returns:
            Comprehensive response
        """
        # Analyze query
        analysis = self.query_understanding.analyze_query(query)
        entities = analysis['entities']

        # Route to appropriate handler
        if analysis['query_type'] == 'organ_specific' and entities['drugs'] and entities['organs']:
            return self.enhanced_complex_organ_query(
                entities['drugs'][0],
                entities['organs'][0]
            )

        elif 'comparison' in analysis['query_type'] and len(entities['drugs']) >= 2:
            return self.enhanced_complex_comparison(
                entities['drugs'][0],
                entities['drugs'][1]
            )

        else:
            # Fallback to general complex query handling
            return {
                'query': query,
                'analysis': analysis,
                'message': 'Query processed with standard retrieval',
                'architecture': 'advanced_rag_format_b'
            }