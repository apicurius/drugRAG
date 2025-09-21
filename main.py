#!/usr/bin/env python3
"""
DrugRAG: Unified Drug Side Effect Retrieval System
Main entry point for all architectures
"""

import json
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.architectures.rag_format_a import FormatARAG
from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG
from src.architectures.enhanced_graphrag import MicrosoftGraphRAGUltimate as EnhancedGraphRAG
from src.architectures.enhanced_rag_format_b import EnhancedFormatBRAG
from src.models.llama import LLAMA3Model
from src.models.qwen import QwenModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DrugRAG:
    """
    Unified interface for all DrugRAG architectures
    Supports 7 architectures for binary queries and 4 for complex queries
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize all architectures"""
        logger.info("="*80)
        logger.info("INITIALIZING DRUGRAG UNIFIED SYSTEM")
        logger.info("="*80)
        
        self.config_path = config_path
        self.architectures = {}
        
        # Initialize architectures lazily to avoid connection issues
        self.architecture_classes = {
            'format_a': FormatARAG,
            'format_a_llama3': lambda cfg: FormatARAG(cfg, llm_model='llama3'),
            'format_a_qwen': lambda cfg: FormatARAG(cfg, llm_model='qwen'),
            'format_b': FormatBRAG,
            'format_b_llama3': lambda cfg: FormatBRAG(cfg, llm_model='llama3'),
            'format_b_qwen': lambda cfg: FormatBRAG(cfg, llm_model='qwen'),
            'graphrag': GraphRAG,
            'enhanced_graphrag': EnhancedGraphRAG,
            'enhanced_format_b': EnhancedFormatBRAG,
            'enhanced_format_b_llama3': lambda cfg: EnhancedFormatBRAG(cfg, llm_model='llama3'),
            'enhanced_format_b_qwen': lambda cfg: EnhancedFormatBRAG(cfg, llm_model='qwen'),
            'llama3': LLAMA3Model,
            'qwen': QwenModel
        }
        
        logger.info("✅ DrugRAG system ready")
    
    def get_architecture(self, name: str):
        """Get or initialize architecture on demand"""
        if name not in self.architectures:
            if name in self.architecture_classes:
                logger.info(f"Initializing {name}...")
                self.architectures[name] = self.architecture_classes[name](self.config_path)
            else:
                raise ValueError(f"Unknown architecture: {name}")
        return self.architectures[name]
    
    def binary_query(self, drug: str, side_effect: str, architecture: str = 'all') -> Dict[str, Any]:
        """
        Execute binary query on specified architecture(s)
        
        Args:
            drug: Drug name
            side_effect: Side effect to check
            architecture: 'all' or specific architecture name
        
        Returns:
            Query results from specified architecture(s)
        """
        if architecture == 'all':
            results = {}
            for arch_name in self.architecture_classes.keys():
                try:
                    arch = self.get_architecture(arch_name)
                    results[arch_name] = arch.query(drug, side_effect)
                    logger.info(f"✓ {arch_name}: {results[arch_name]['answer']}")
                except Exception as e:
                    logger.error(f"✗ {arch_name} failed: {e}")
                    results[arch_name] = {
                        'answer': 'ERROR',
                        'error': str(e)
                    }
            return results
        else:
            arch = self.get_architecture(architecture)
            return arch.query(drug, side_effect)
    
    def complex_query(self, query_type: str, architecture: str = 'enhanced_graphrag', **kwargs) -> Dict[str, Any]:
        """
        Execute complex query on capable architectures
        
        Args:
            query_type: Type of complex query (organ_specific, severity_filtered, etc.)
            architecture: Architecture to use ('all' for all capable ones)
            **kwargs: Query-specific parameters
        
        Returns:
            Query results
        """
        # Architectures that support complex queries
        complex_capable = ['enhanced_graphrag', 'enhanced_format_b', 'llama3', 'qwen']
        
        if architecture == 'all':
            results = {}
            for arch_name in complex_capable:
                try:
                    arch = self.get_architecture(arch_name)
                    method_name = f'complex_query_{query_type}'
                    if hasattr(arch, method_name):
                        method = getattr(arch, method_name)
                        results[arch_name] = method(**kwargs)
                        logger.info(f"✓ {arch_name} completed {query_type}")
                    else:
                        results[arch_name] = {
                            'error': f'{arch_name} does not support {query_type}'
                        }
                except Exception as e:
                    logger.error(f"✗ {arch_name} failed: {e}")
                    results[arch_name] = {'error': str(e)}
            return results
        else:
            arch = self.get_architecture(architecture)
            method_name = f'complex_query_{query_type}'
            if hasattr(arch, method_name):
                method = getattr(arch, method_name)
                return method(**kwargs)
            else:
                raise ValueError(f"{architecture} does not support {query_type}")
    
    def list_architectures(self) -> Dict[str, List[str]]:
        """List all available architectures and their capabilities"""
        return {
            'binary_query': list(self.architecture_classes.keys()),
            'complex_query': ['enhanced_graphrag', 'enhanced_format_b', 'llama3', 'qwen'],
            'rag_architectures': ['format_a', 'format_b', 'graphrag', 'enhanced_graphrag', 'enhanced_format_b'],
            'llm_baselines': ['llama3', 'qwen']
        }
    
    def close(self):
        """Close all connections"""
        for arch in self.architectures.values():
            if hasattr(arch, 'close'):
                arch.close()


def main():
    """Example usage of DrugRAG system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DrugRAG Unified System')
    parser.add_argument('--drug', type=str, help='Drug name')
    parser.add_argument('--side-effect', type=str, help='Side effect')
    parser.add_argument('--architecture', type=str, default='all', 
                       help='Architecture to use (default: all)')
    parser.add_argument('--query-type', type=str, default='binary',
                       help='Query type: binary, organ_specific, severity_filtered, etc.')
    parser.add_argument('--organ', type=str, help='Organ system for organ_specific query')
    parser.add_argument('--severity', type=str, help='Severity level for severity_filtered query')
    
    args = parser.parse_args()
    
    # Initialize system
    drugrag = DrugRAG()
    
    try:
        if args.query_type == 'binary':
            if not args.drug or not args.side_effect:
                print("Error: --drug and --side-effect required for binary query")
                return
            
            print(f"\n🔍 Binary Query: {args.drug} → {args.side_effect}")
            print("-" * 60)
            
            results = drugrag.binary_query(args.drug, args.side_effect, args.architecture)
            
            if args.architecture == 'all':
                print("\n📊 Results from all architectures:")
                for arch, result in results.items():
                    answer = result.get('answer', 'ERROR')
                    confidence = result.get('confidence', 0.0)
                    print(f"  {arch:20s}: {answer:8s} (confidence: {confidence:.2f})")
            else:
                print(f"\n📊 Result from {args.architecture}:")
                print(f"  Answer: {results.get('answer', 'ERROR')}")
                print(f"  Confidence: {results.get('confidence', 0.0):.2f}")
        
        elif args.query_type == 'organ_specific':
            if not args.drug or not args.organ:
                print("Error: --drug and --organ required for organ_specific query")
                return
            
            print(f"\n🔍 Organ-Specific Query: {args.drug} → {args.organ}")
            print("-" * 60)
            
            results = drugrag.complex_query('organ_specific', 
                                           architecture=args.architecture,
                                           drug=args.drug,
                                           organ=args.organ)
            
            if args.architecture == 'all':
                for arch, result in results.items():
                    effects = result.get('side_effects_found', [])
                    print(f"\n{arch}:")
                    print(f"  Found {len(effects)} effects: {', '.join(effects[:5])}...")
            else:
                effects = results.get('side_effects_found', [])
                print(f"Found {len(effects)} effects: {', '.join(effects)}")
        
        elif args.query_type == 'list':
            print("\n📋 Available Architectures:")
            print("-" * 60)
            capabilities = drugrag.list_architectures()
            for category, archs in capabilities.items():
                print(f"\n{category}:")
                for arch in archs:
                    print(f"  - {arch}")
        
        else:
            print(f"Query type '{args.query_type}' not yet implemented in CLI")
    
    finally:
        drugrag.close()


if __name__ == "__main__":
    main()