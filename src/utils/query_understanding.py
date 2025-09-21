#!/usr/bin/env python3
"""
Query Understanding Module for Complex Queries
Provides query decomposition, entity extraction, and semantic expansion
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QueryUnderstanding:
    """Understand and decompose complex drug-related queries"""

    def __init__(self):
        """Initialize with medical term mappings"""
        # Organ system synonyms and related terms
        self.organ_mappings = {
            'heart': ['cardiac', 'cardiovascular', 'coronary', 'myocardial', 'atrial', 'ventricular'],
            'liver': ['hepatic', 'hepato', 'hepatobiliary', 'portal'],
            'kidney': ['renal', 'nephro', 'urinary', 'glomerular'],
            'lung': ['pulmonary', 'respiratory', 'bronchial', 'alveolar'],
            'stomach': ['gastrointestinal', 'gastric', 'GI', 'digestive', 'intestinal'],
            'brain': ['neurological', 'cerebral', 'CNS', 'cognitive', 'mental'],
            'skin': ['dermatological', 'cutaneous', 'epidermal', 'dermal'],
            'blood': ['hematological', 'vascular', 'thrombotic', 'hemorrhagic'],
            'muscle': ['muscular', 'musculoskeletal', 'myopathy'],
            'eye': ['ocular', 'ophthalmic', 'visual', 'retinal']
        }

        # Severity terms
        self.severity_terms = {
            'severe': ['severe', 'serious', 'critical', 'life-threatening', 'fatal', 'dangerous'],
            'moderate': ['moderate', 'significant', 'notable', 'important'],
            'mild': ['mild', 'minor', 'slight', 'minimal', 'trivial']
        }

        # Query type patterns
        self.query_patterns = {
            'comparison': r'compare|versus|vs|between|difference',
            'organ_specific': r'(heart|liver|kidney|lung|stomach|brain|skin|blood|muscle|eye|cardiac|hepatic|renal|pulmonary)',
            'severity': r'(severe|moderate|mild|serious|critical|life-threatening)',
            'common': r'common|shared|both|mutual',
            'unique': r'unique|exclusive|only|specific',
            'all': r'all|every|complete|full|list',
            'count': r'how many|number of|count'
        }

    def classify_query(self, query: str) -> str:
        """
        Classify the query type

        Returns:
            Query type: comparison, organ_specific, severity_filtered, reverse_lookup, etc.
        """
        query_lower = query.lower()

        # Check for comparison queries
        if re.search(self.query_patterns['comparison'], query_lower):
            if re.search(self.query_patterns['common'], query_lower):
                return 'drug_comparison_common'
            elif re.search(self.query_patterns['unique'], query_lower):
                return 'drug_comparison_unique'
            else:
                return 'drug_comparison'

        # Check for organ-specific queries
        if re.search(self.query_patterns['organ_specific'], query_lower):
            if re.search(self.query_patterns['severity'], query_lower):
                return 'organ_severity_combined'
            return 'organ_specific'

        # Check for severity queries
        if re.search(self.query_patterns['severity'], query_lower):
            return 'severity_filtered'

        # Check for reverse lookup (all effects)
        if re.search(self.query_patterns['all'], query_lower) and 'effect' in query_lower:
            return 'reverse_lookup'

        return 'general'

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from the query

        Returns:
            Dictionary with drugs, organs, severity levels, and effects
        """
        entities = {
            'drugs': [],
            'organs': [],
            'severity': [],
            'effects': []
        }

        query_lower = query.lower()

        # Extract drug names (simplified - in production, use NER model)
        # Look for capitalized words or known drug patterns
        drug_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        potential_drugs = re.findall(drug_pattern, query)
        entities['drugs'] = [d for d in potential_drugs if len(d) > 3]

        # Extract organ systems
        for organ, terms in self.organ_mappings.items():
            if organ in query_lower or any(term in query_lower for term in terms):
                entities['organs'].append(organ)

        # Extract severity
        for severity, terms in self.severity_terms.items():
            if any(term in query_lower for term in terms):
                entities['severity'].append(severity)

        # Extract specific effects mentioned (between quotes or after "effect")
        effect_pattern = r'"([^"]+)"|effect[s]?\s+(?:like|such as|including)\s+([^,\.]+)'
        effect_matches = re.findall(effect_pattern, query_lower)
        for match in effect_matches:
            effect = match[0] or match[1]
            if effect:
                entities['effects'].append(effect.strip())

        return entities

    def decompose_query(self, query: str, query_type: str) -> List[str]:
        """
        Decompose complex query into sub-queries for better retrieval

        Returns:
            List of sub-queries
        """
        sub_queries = []
        entities = self.extract_entities(query)

        if query_type == 'drug_comparison_common':
            # Generate queries for each drug + common patterns
            for drug in entities['drugs']:
                sub_queries.append(f"{drug} side effects adverse events")
            if len(entities['drugs']) >= 2:
                sub_queries.append(f"{entities['drugs'][0]} {entities['drugs'][1]} common effects")

        elif query_type == 'organ_specific':
            for drug in entities['drugs']:
                for organ in entities['organs']:
                    sub_queries.append(f"{drug} {organ} effects")
                    # Add synonyms
                    for synonym in self.organ_mappings.get(organ, [])[:3]:
                        sub_queries.append(f"{drug} {synonym} adverse events")

        elif query_type == 'severity_filtered':
            for drug in entities['drugs']:
                for severity in entities['severity']:
                    sub_queries.append(f"{drug} {severity} adverse effects")
                    for term in self.severity_terms.get(severity, [])[:2]:
                        sub_queries.append(f"{drug} {term} side effects")

        else:
            # Default: create basic sub-queries
            for drug in entities['drugs']:
                sub_queries.append(f"{drug} side effects")
                sub_queries.append(f"{drug} adverse events")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in sub_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        return unique_queries[:10]  # Limit to 10 sub-queries

    def expand_medical_terms(self, term: str) -> List[str]:
        """
        Expand medical term with synonyms and related terms

        Returns:
            List of expanded terms including original
        """
        expanded = [term]
        term_lower = term.lower()

        # Check organ mappings
        for organ, synonyms in self.organ_mappings.items():
            if organ == term_lower or term_lower in synonyms:
                expanded.extend([organ] + synonyms[:3])
                break

        # Check severity mappings
        for severity, synonyms in self.severity_terms.items():
            if severity == term_lower or term_lower in synonyms:
                expanded.extend(synonyms[:3])
                break

        # Common medical abbreviation expansions
        abbreviations = {
            'mi': ['myocardial infarction', 'heart attack'],
            'htn': ['hypertension', 'high blood pressure'],
            'dm': ['diabetes mellitus', 'diabetes'],
            'chf': ['congestive heart failure', 'heart failure'],
            'copd': ['chronic obstructive pulmonary disease'],
            'gi': ['gastrointestinal', 'digestive'],
            'cns': ['central nervous system', 'neurological'],
            'cvs': ['cardiovascular system', 'cardiac']
        }

        if term_lower in abbreviations:
            expanded.extend(abbreviations[term_lower])

        # Remove duplicates
        return list(set(expanded))

    def analyze_query(self, query: str) -> Dict[str, any]:
        """
        Complete query analysis combining all components

        Returns:
            Dictionary with query type, entities, sub-queries, and expansions
        """
        query_type = self.classify_query(query)
        entities = self.extract_entities(query)
        sub_queries = self.decompose_query(query, query_type)

        # Expand key terms
        expanded_terms = {}
        for organ in entities.get('organs', []):
            expanded_terms[organ] = self.expand_medical_terms(organ)

        return {
            'original_query': query,
            'query_type': query_type,
            'entities': entities,
            'sub_queries': sub_queries,
            'expanded_terms': expanded_terms,
            'search_strategy': self._suggest_search_strategy(query_type)
        }

    def _suggest_search_strategy(self, query_type: str) -> str:
        """Suggest optimal search strategy based on query type"""
        strategies = {
            'drug_comparison_common': 'parallel_retrieval_intersection',
            'drug_comparison_unique': 'parallel_retrieval_difference',
            'organ_specific': 'hierarchical_filtering',
            'severity_filtered': 'severity_weighted_ranking',
            'reverse_lookup': 'exhaustive_retrieval',
            'organ_severity_combined': 'multi_stage_filtering',
            'general': 'standard_retrieval'
        }
        return strategies.get(query_type, 'standard_retrieval')

    def generate_enhanced_prompt(self, query_analysis: Dict, retrieved_context: str) -> str:
        """
        Generate enhanced prompt with chain-of-thought reasoning

        Args:
            query_analysis: Output from analyze_query
            retrieved_context: Retrieved documents/evidence

        Returns:
            Enhanced prompt for LLM
        """
        query_type = query_analysis['query_type']
        entities = query_analysis['entities']

        # Base chain-of-thought template
        prompt = f"""Complex Medical Query Analysis

Original Query: {query_analysis['original_query']}
Query Type: {query_type}

Step 1: Understanding the Request
- Drugs involved: {', '.join(entities.get('drugs', ['None identified']))}
- Organ systems: {', '.join(entities.get('organs', ['Not specified']))}
- Severity filter: {', '.join(entities.get('severity', ['Any severity']))}
- Specific effects: {', '.join(entities.get('effects', ['All effects']))}

Step 2: Retrieved Medical Evidence
{retrieved_context}

Step 3: Analysis Approach
"""

        # Add query-type specific reasoning steps
        if 'comparison' in query_type:
            prompt += """
- Identify effects for each drug separately
- Find overlapping effects (common to both drugs)
- Identify unique effects (specific to each drug)
- Rank by clinical significance
"""
        elif query_type == 'organ_specific':
            prompt += f"""
- Filter effects related to {', '.join(entities.get('organs', []))} system
- Consider both direct organ effects and systemic effects affecting the organ
- Include related medical terminology: {', '.join([term for organ in entities.get('organs', []) for term in self.organ_mappings.get(organ, [])[:3]])}
- Rank by frequency and severity within organ system
"""
        elif query_type == 'severity_filtered':
            prompt += f"""
- Focus on {', '.join(entities.get('severity', []))} effects only
- Consider FDA black box warnings if severity is severe
- Exclude mild or transient effects if severity filter is moderate/severe
- Rank by clinical impact
"""

        prompt += """

Step 4: Generate Comprehensive Answer
Based on the evidence and analysis above, provide a structured response:

FINDINGS:
[List the relevant effects/findings based on the query]

CONFIDENCE:
[Rate confidence as HIGH/MEDIUM/LOW based on evidence strength]

LIMITATIONS:
[Note any gaps in the evidence or uncertainty]

"""

        return prompt