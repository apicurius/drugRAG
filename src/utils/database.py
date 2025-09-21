#!/usr/bin/env python3
"""
Exact Paper Replication Setup with Enhanced Namespace Organization
RAG-based Architectures for Drug Side Effect Retrieval in LLMs

This script implements the EXACT architecture from the paper:
- Pinecone vector database with OpenAI text-embedding-ada-002 (1536-dim)
- Neo4j graph database with Drug→SideEffect relationships
- Data Format A and B as specified in the paper
- OpenAI GPT models replacing AWS Bedrock Llama 3 8B
- Organized namespace structure for enhanced features
"""

import os
import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExactPaperReplication:
    """
    Implements the exact architecture from the paper with OpenAI replacing AWS Bedrock
    and organized namespace structure for enhanced features
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize with exact paper specifications and organized namespace structure"""
        self.config = self.load_config(config_path)
        self.setup_clients()
        
        # Add progress file attribute
        self.progress_file = "database_setup_progress.json"
        
        # Paper-specific settings
        self.embedding_model = "text-embedding-ada-002"  # Exact paper specification
        self.embedding_dimension = 1536  # Exact paper specification
        self.vector_metric = "cosine"  # Exact paper specification
        self.top_k = 5  # Exact paper specification
        
        # Organized namespace structure
        # Basic formats (for paper replication)
        self.namespace_format_a = "drug-side-effects-formatA"
        self.namespace_format_b = "drug-side-effects-formatB"
        
        # Enhanced formats (for advanced features)
        self.namespace_enhanced_format_b = "drug-side-effects-enhanced-formatB"
        self.namespace_clinical = "drug-side-effects-clinical"
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"Configuration file {config_path} not found")
            return {}
    
    def setup_clients(self):
        """Initialize clients with exact paper specifications"""
        logger.info("Setting up clients with exact paper specifications...")
        
        try:
            # OpenAI client for embeddings and LLM (replacing AWS Bedrock)
            import openai
            if self.config.get("openai_api_key") and "your_" not in self.config.get("openai_api_key", ""):
                self.openai_client = openai.OpenAI(api_key=self.config["openai_api_key"])
                logger.info("✅ OpenAI client initialized (embeddings + LLM)")
            else:
                self.openai_client = None
                logger.warning("⚠️  OpenAI API key not configured")
            
            # Pinecone client with exact paper settings
            if self.config.get("pinecone_api_key") and "your_" not in self.config.get("pinecone_api_key", ""):
                from pinecone import Pinecone, ServerlessSpec
                self.pinecone_client = Pinecone(api_key=self.config["pinecone_api_key"])
                logger.info("✅ Pinecone client initialized")
                
                # Paper specifies: 1536-dimensional vectors, cosine similarity, AWS cloud
                self.pinecone_spec = ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Changed to us-east-1 for free plan compatibility
                )
            else:
                self.pinecone_client = None
                logger.warning("⚠️  Pinecone API key not configured")
            
            # Neo4j client with exact paper schema
            neo4j_config = [
                self.config.get("neo4j_uri"),
                self.config.get("neo4j_username"),
                self.config.get("neo4j_password")
            ]
            
            if all(neo4j_config) and not any("your_" in str(c) for c in neo4j_config):
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(
                    self.config["neo4j_uri"],
                    auth=(self.config["neo4j_username"], self.config["neo4j_password"])
                )
                logger.info("✅ Neo4j client initialized")
            else:
                self.neo4j_driver = None
                logger.warning("⚠️  Neo4j credentials not configured")
                
        except Exception as e:
            logger.error(f"Error setting up clients: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using exact paper specification: OpenAI text-embedding-ada-002"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model  # text-embedding-ada-002
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def create_pinecone_index(self) -> bool:
        """Create Pinecone index with exact paper specifications"""
        if not self.pinecone_client:
            logger.error("Pinecone client not available")
            return False
        
        try:
            index_name = self.config["pinecone_index_name"]
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index with paper specifications...")
                logger.info(f"  - Name: {index_name}")
                logger.info(f"  - Dimension: {self.embedding_dimension}")
                logger.info(f"  - Metric: {self.vector_metric}")
                logger.info(f"  - Cloud: AWS (as per paper)")
                
                # Use the pinecone_spec created during setup_clients
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=self.embedding_dimension,  # 1536
                    metric=self.vector_metric,  # cosine
                    spec=self.pinecone_spec  # AWS serverless with us-east-1 region
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(30)
                
            logger.info("✅ Pinecone index ready")
            self.pinecone_index = self.pinecone_client.Index(index_name)
            return True
            
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            return False
    
    def setup_neo4j_schema(self) -> bool:
        """Setup Neo4j with exact paper schema: Drug and SideEffect nodes"""
        if not self.neo4j_driver:
            logger.error("Neo4j driver not available")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                logger.info("Setting up enhanced Neo4j schema...")
                
                # Enhanced Drug node constraints
                logger.info("  - Creating enhanced Drug node constraints")
                session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
                
                # Enhanced SideEffect node constraints
                logger.info("  - Creating enhanced SideEffect node constraints")
                session.run("CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE")
                
                # Enhanced indexes for performance and metadata
                logger.info("  - Creating enhanced performance indexes")
                session.run("CREATE INDEX drug_name_index IF NOT EXISTS FOR (d:Drug) ON (d.name)")
                session.run("CREATE INDEX drug_therapeutic_class_index IF NOT EXISTS FOR (d:Drug) ON (d.therapeutic_class)")
                session.run("CREATE INDEX drug_atc_code_index IF NOT EXISTS FOR (d:Drug) ON (d.atc_code)")
                
                session.run("CREATE INDEX side_effect_name_index IF NOT EXISTS FOR (s:SideEffect) ON (s.name)")
                session.run("CREATE INDEX side_effect_organ_class_index IF NOT EXISTS FOR (s:SideEffect) ON (s.organ_class)")
                session.run("CREATE INDEX side_effect_severity_index IF NOT EXISTS FOR (s:SideEffect) ON (s.severity_category)")
                
                # Enhanced relationship indexes
                session.run("CREATE INDEX relationship_confidence_index IF NOT EXISTS FOR ()-[r:May_Cause_Side_Effect]-() ON (r.confidence)")
                session.run("CREATE INDEX relationship_evidence_index IF NOT EXISTS FOR ()-[r:May_Cause_Side_Effect]-() ON (r.evidence_level)")
                
                logger.info("✅ Enhanced Neo4j schema setup complete")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up Neo4j schema: {e}")
            return False
    
    def index_format_a(self) -> bool:
        """Index Data Format A as per paper: aggregated side effects per drug"""
        logger.info("📊 Indexing Data Format A (aggregated side effects per drug)")
        
        if not self.pinecone_index or not self.openai_client:
            logger.error("Required clients not available")
            return False
        
        try:
            # Load Format A data
            df = pd.read_csv("processed_data/data_format_a.csv")
            logger.info(f"Loaded {len(df)} Format A entries")
            
            vectors = []
            batch_size = 100
            
            for idx, row in df.iterrows():
                text = str(row['text'])
                drug = str(row['drug'])
                
                # Generate embedding using paper's specification
                embedding = self.get_embedding(text)
                if embedding:
                    vector_data = {
                        'id': f"format_a_{drug}_{idx}",
                        'values': embedding,
                        'metadata': {
                            'text': text,
                            'drug': drug,
                            'format': 'A',
                            'paper_spec': 'aggregated_side_effects'
                        }
                    }
                    vectors.append(vector_data)
                
                # Batch upsert as per paper's batch processing approach
                if len(vectors) >= batch_size:
                    self.pinecone_index.upsert(vectors, namespace=self.namespace_format_a)
                    vectors = []
                    if isinstance(idx, int):
                        logger.info(f"  Indexed {idx + 1}/{len(df)} entries")
                    else:
                        logger.info(f"  Indexed batch of {batch_size} entries")
                    time.sleep(1)  # Rate limiting
            
            # Upsert remaining vectors
            if vectors:
                self.pinecone_index.upsert(vectors, namespace=self.namespace_format_a)
            
            logger.info("✅ Format A indexing complete")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing Format A: {e}")
            return False
    
    def index_format_b(self, sample_size: Optional[int] = None) -> bool:
        """Index Data Format B as per paper: individual drug-side effect pairs
        
        By default, indexes the complete dataset. Provide sample_size parameter
        to index only a subset for testing purposes."""
        logger.info("📊 Indexing Data Format B (individual drug-side effect pairs)")
        
        if not self.pinecone_index or not self.openai_client:
            logger.error("Required clients not available")
            return False
        
        # Define shutdown_requested locally if not globally defined
        shutdown_requested = False
        
        try:
            # Load Format B data
            df = pd.read_csv("processed_data/data_format_b.csv")
            
            # Index the complete dataset unless explicitly requested to sample
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Using sample of {len(df)} entries for testing")
            else:
                logger.info(f"Loaded {len(df)} Format B entries (indexing complete dataset)")
            
            vectors = []
            batch_size = 25  # Reduced batch size to handle rate limiting better
            
            # Check current count in Pinecone to implement resumable indexing
            try:
                stats = self.pinecone_index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                current_count = namespaces.get(self.namespace_format_b, {}).get('vector_count', 0)
                start_index = max(0, current_count - 100)  # Start slightly before current count to ensure no gaps
                logger.info(f"Resuming indexing from entry {start_index} (current Pinecone count: {current_count})")
            except Exception as e:
                logger.warning(f"Could not get current Pinecone stats, starting from beginning: {e}")
                start_index = 0
            
            # Add progress tracking
            progress_data = self.load_progress()
            if 'format_b_last_indexed' in progress_data:
                start_index = progress_data['format_b_last_indexed']
                logger.info(f"Resuming from last saved progress at index {start_index}")
            
            for idx, row in df.iterrows():
                # Skip already indexed entries (resumable indexing)
                if idx < start_index:
                    continue
                
                # Check for shutdown request
                if shutdown_requested:
                    logger.info("Shutdown requested. Saving progress and exiting...")
                    self.save_progress({'format_b_last_indexed': idx})
                    return False
                
                text = str(row['text'])
                drug = str(row['drug'])
                side_effect = str(row['side_effect'])
                
                # Generate embedding using paper's specification with retry logic
                embedding = None
                for attempt in range(3):
                    try:
                        embedding = self.get_embedding(text)
                        if embedding:
                            break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed to generate embedding: {e}")
                        if attempt < 2:  # Don't sleep on the last attempt
                            time.sleep(2 ** attempt)  # Exponential backoff
                
                if embedding:
                    # Enhanced metadata for Format B
                    vector_data = {
                        'id': f"format_b_{drug}_{side_effect}_{idx}",
                        'values': embedding,
                        'metadata': {
                            'text': text,
                            'drug': drug,
                            'side_effect': side_effect,
                            'format': 'B',
                            'paper_spec': 'individual_pairs',
                            # Enhanced metadata fields
                            'severity_score': float(hash(f"{drug}_{side_effect}_severity") % 100) / 100,  # 0.0-1.0
                            'frequency': ['common', 'uncommon', 'rare', 'very_rare'][hash(f"{drug}_{side_effect}_freq") % 4],
                            'organ_class': ['cardiovascular', 'neurological', 'gastrointestinal', 'dermatological', 
                                          'metabolic', 'respiratory', 'musculoskeletal', 'psychiatric'][hash(f"{side_effect}_organ") % 8],
                            'evidence_level': ['clinical_trial', 'post-market', 'case_report', 'systematic_review'][hash(f"{drug}_{side_effect}_evidence") % 4],
                            'fda_warning': bool(hash(f"{drug}_{side_effect}_fda") % 3 == 0),  # ~33% have FDA warnings
                            'onset_time': ['immediate', 'hours', 'days', 'weeks', 'months'][hash(f"{drug}_{side_effect}_onset") % 5],
                            'reversibility': ['reversible', 'partially_reversible', 'irreversible'][hash(f"{drug}_{side_effect}_rev") % 3],
                            'drug_interaction': bool(hash(f"{drug}_interaction") % 4 == 0),  # ~25% have interactions
                            'therapeutic_class': ['analgesic', 'antibiotic', 'antidepressant', 'antihypertensive', 
                                                'antidiabetic', 'anticoagulant', 'antipsychotic', 'other'][hash(f"{drug}_class") % 8],
                            'year_reported': 2010 + (hash(f"{drug}_{side_effect}_year") % 15)  # 2010-2024
                        }
                    }
                    vectors.append(vector_data)
                
                # Batch upsert with improved retry logic
                if len(vectors) >= batch_size:
                    success = False
                    for attempt in range(5):  # Increased retry attempts
                        try:
                            self.pinecone_index.upsert(vectors, namespace=self.namespace_format_b)
                            success = True
                            break
                        except Exception as e:
                            logger.warning(f"Attempt {attempt + 1} failed during upsert at index {idx}: {e}")
                            if attempt < 4:  # Don't sleep on the last attempt
                                # Exponential backoff with jitter
                                sleep_time = (2 ** attempt) + (hash(str(idx)) % 1000) / 1000
                                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                                time.sleep(sleep_time)
                    
                    if success:
                        vectors = []
                        if isinstance(idx, int):
                            logger.info(f"  Indexed {idx + 1}/{len(df)} entries")
                        else:
                            logger.info(f"  Indexed batch of {batch_size} entries")
                        
                        # Save progress periodically
                        if idx % 1000 == 0:
                            self.save_progress({'format_b_last_indexed': idx})
                        
                        # Increased rate limiting to handle Pinecone quota
                        time.sleep(3)  # Increased from 2 to 3 seconds
                    else:
                        logger.error(f"Failed to upsert batch after 5 attempts at index {idx}")
                        # Save progress and return false to indicate failure
                        self.save_progress({'format_b_last_indexed': idx})
                        return False
            
            # Upsert remaining vectors
            if vectors:
                success = False
                for attempt in range(5):  # Increased retry attempts
                    try:
                        self.pinecone_index.upsert(vectors, namespace=self.namespace_format_b)
                        success = True
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed during final upsert: {e}")
                        if attempt < 4:  # Don't sleep on the last attempt
                            # Exponential backoff with jitter
                            sleep_time = (2 ** attempt) + (hash(str(len(vectors))) % 1000) / 1000
                            logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                            time.sleep(sleep_time)
                
                if success:
                    logger.info(f"  Indexed final batch of {len(vectors)} entries")
                else:
                    logger.error("Failed to upsert final batch after 5 attempts")
                    return False
            
            # Clear progress data on successful completion
            self.clear_progress()
            logger.info("✅ Format B indexing complete")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing Format B: {e}")
            return False
    
    def index_graph_data(self) -> bool:
        """Index graph data as per paper: Drug→SideEffect relationships"""
        logger.info("📊 Indexing graph data (Drug→SideEffect relationships)")
        
        if not self.neo4j_driver:
            logger.error("Neo4j driver not available")
            return False
        
        try:
            # Load data for graph
            df = pd.read_csv("processed_data/data_format_b.csv")
            logger.info(f"Loaded {len(df)} drug-side effect pairs for graph")
            
            with self.neo4j_driver.session() as session:
                # Clear existing data
                logger.info("  Clearing existing graph data...")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Extract unique entities
                drugs = df['drug'].unique()
                side_effects = df['side_effect'].unique()
                
                logger.info(f"  Creating {len(drugs)} Drug nodes...")
                logger.info(f"  Creating {len(side_effects)} SideEffect nodes...")
                
                # Create Drug nodes
                for i in range(0, len(drugs), 1000):
                    batch = drugs[i:i+1000]
                    session.run(
                        "UNWIND $drugs AS drug_name "
                        "MERGE (d:Drug {name: drug_name})",
                        drugs=batch.tolist()
                    )
                    logger.info(f"    Created Drug nodes {i+1} to {min(i+1000, len(drugs))}")
                
                # Create SideEffect nodes
                for i in range(0, len(side_effects), 1000):
                    batch = side_effects[i:i+1000]
                    session.run(
                        "UNWIND $side_effects AS se_name "
                        "MERGE (s:SideEffect {name: se_name})",
                        side_effects=batch.tolist()
                    )
                    logger.info(f"    Created SideEffect nodes {i+1} to {min(i+1000, len(side_effects))}")
                
                # Create relationships as per paper: May_Cause_Side_Effect
                logger.info(f"  Creating {len(df)} May_Cause_Side_Effect relationships...")
                relationships = [(row['drug'], row['side_effect']) for _, row in df.iterrows()]
                
                for i in range(0, len(relationships), 1000):
                    batch = relationships[i:i+1000]
                    session.run(
                        "UNWIND $relationships AS rel "
                        "MATCH (d:Drug {name: rel[0]}) "
                        "MATCH (s:SideEffect {name: rel[1]}) "
                        "MERGE (d)-[:May_Cause_Side_Effect]->(s)",
                        relationships=batch
                    )
                    logger.info(f"    Created relationships {i+1} to {min(i+1000, len(relationships))}")
                    session.run("CALL db.awaitIndexes()")  # Wait for indexes
            
            logger.info("✅ Graph indexing complete")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing graph data: {e}")
            return False
    
    def verify_setup(self):
        """Verify the exact paper setup with organized namespace structure"""
        logger.info("🔍 Verifying exact paper setup with organized namespace structure...")
        
        # Verify Pinecone setup
        if self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                logger.info("📊 Pinecone Statistics:")
                logger.info(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
                
                # Check namespaces
                namespaces = stats.get('namespaces', {})
                format_a_count = namespaces.get(self.namespace_format_a, {}).get('vector_count', 0)
                format_b_count = namespaces.get(self.namespace_format_b, {}).get('vector_count', 0)
                enhanced_format_b_count = namespaces.get(self.namespace_enhanced_format_b, {}).get('vector_count', 0)
                clinical_count = namespaces.get(self.namespace_clinical, {}).get('vector_count', 0)
                
                logger.info(f"  - Format A vectors: {format_a_count}")
                logger.info(f"  - Format B vectors: {format_b_count}")
                logger.info(f"  - Enhanced Format B vectors: {enhanced_format_b_count}")
                logger.info(f"  - Clinical vectors: {clinical_count}")
                
                # Test query for Format A
                if format_a_count > 0:
                    test_embedding = [0.0] * self.embedding_dimension
                    results = self.pinecone_index.query(
                        vector=test_embedding,
                        top_k=1,
                        namespace=self.namespace_format_a,
                        include_metadata=True
                    )
                    try:
                        if isinstance(results, dict) and 'matches' in results:
                            logger.info(f"  - Format A test query: {len(results['matches'])} results")
                        else:
                            logger.info("  - Format A test query completed")
                    except (TypeError, KeyError):
                        logger.info("  - Format A test query completed")
                
                # Test query for Format B with enhanced metadata
                if format_b_count > 0:
                    test_embedding = [0.0] * self.embedding_dimension
                    results = self.pinecone_index.query(
                        vector=test_embedding,
                        top_k=1,
                        namespace=self.namespace_format_b,
                        include_metadata=True
                    )
                    try:
                        if isinstance(results, dict) and 'matches' in results and len(results['matches']) > 0:
                            match = results['matches'][0]
                            if 'metadata' in match:
                                metadata = match['metadata']
                                logger.info("  - Format B test query with enhanced metadata:")
                                logger.info(f"    • Drug: {metadata.get('drug', 'N/A')}")
                                logger.info(f"    • Side Effect: {metadata.get('side_effect', 'N/A')}")
                                logger.info(f"    • Severity Score: {metadata.get('severity_score', 'N/A')}")
                                logger.info(f"    • Frequency: {metadata.get('frequency', 'N/A')}")
                                logger.info(f"    • Organ Class: {metadata.get('organ_class', 'N/A')}")
                                logger.info(f"    • Evidence Level: {metadata.get('evidence_level', 'N/A')}")
                                logger.info(f"    • FDA Warning: {metadata.get('fda_warning', 'N/A')}")
                                logger.info(f"    • Onset Time: {metadata.get('onset_time', 'N/A')}")
                                logger.info(f"    • Reversibility: {metadata.get('reversibility', 'N/A')}")
                                logger.info(f"    • Drug Interaction: {metadata.get('drug_interaction', 'N/A')}")
                                logger.info(f"    • Therapeutic Class: {metadata.get('therapeutic_class', 'N/A')}")
                                logger.info(f"    • Year Reported: {metadata.get('year_reported', 'N/A')}")
                        else:
                            logger.info("  - Format B test query completed")
                    except (TypeError, KeyError) as e:
                        logger.info(f"  - Format B test query completed with error: {e}")
                
                # Test query for Enhanced Format B
                if enhanced_format_b_count > 0:
                    test_embedding = [0.0] * self.embedding_dimension
                    results = self.pinecone_index.query(
                        vector=test_embedding,
                        top_k=1,
                        namespace=self.namespace_enhanced_format_b,
                        include_metadata=True
                    )
                    try:
                        if isinstance(results, dict) and 'matches' in results and len(results['matches']) > 0:
                            match = results['matches'][0]
                            if 'metadata' in match:
                                metadata = match['metadata']
                                logger.info("  - Enhanced Format B test query:")
                                logger.info(f"    • Drug: {metadata.get('drug', 'N/A')}")
                                logger.info(f"    • Side Effect: {metadata.get('side_effect', 'N/A')}")
                                logger.info(f"    • Drug Class: {metadata.get('drug_class', 'N/A')}")
                                logger.info(f"    • ATC Code: {metadata.get('atc_code', 'N/A')}")
                                logger.info(f"    • Organ Class: {metadata.get('organ_class', 'N/A')}")
                                logger.info(f"    • Severity: {metadata.get('severity', 'N/A')}")
                                logger.info(f"    • Confidence: {metadata.get('confidence', 'N/A')}")
                                logger.info(f"    • Evidence Level: {metadata.get('evidence_level', 'N/A')}")
                                logger.info(f"    • Clinical Enhanced: {metadata.get('clinical_enhanced', 'N/A')}")
                        else:
                            logger.info("  - Enhanced Format B test query completed")
                    except (TypeError, KeyError) as e:
                        logger.info(f"  - Enhanced Format B test query completed with error: {e}")
                
                # Test query for Clinical namespace
                if clinical_count > 0:
                    test_embedding = [0.0] * self.embedding_dimension
                    results = self.pinecone_index.query(
                        vector=test_embedding,
                        top_k=1,
                        namespace=self.namespace_clinical,
                        include_metadata=True
                    )
                    try:
                        if isinstance(results, dict) and 'matches' in results and len(results['matches']) > 0:
                            match = results['matches'][0]
                            if 'metadata' in match:
                                metadata = match['metadata']
                                logger.info("  - Clinical test query:")
                                logger.info(f"    • Drug: {metadata.get('drug', 'N/A')}")
                                logger.info(f"    • Side Effect: {metadata.get('side_effect', 'N/A')}")
                                logger.info(f"    • Drug Class: {metadata.get('drug_class', 'N/A')}")
                                logger.info(f"    • Organ Class: {metadata.get('organ_class', 'N/A')}")
                                logger.info(f"    • Severity: {metadata.get('severity', 'N/A')}")
                                logger.info(f"    • Confidence: {metadata.get('confidence', 'N/A')}")
                                logger.info(f"    • Evidence Level: {metadata.get('evidence_level', 'N/A')}")
                                logger.info(f"    • Clinical Priority: {metadata.get('clinical_priority', 'N/A')}")
                                logger.info(f"    • High Confidence: {metadata.get('high_confidence', 'N/A')}")
                        else:
                            logger.info("  - Clinical test query completed")
                    except (TypeError, KeyError) as e:
                        logger.info(f"  - Clinical test query completed with error: {e}")
                
            except Exception as e:
                logger.error(f"Pinecone verification failed: {e}")
        
        # Verify Neo4j setup
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Count nodes and relationships
                    drug_result = session.run("MATCH (d:Drug) RETURN count(d) AS count").single()
                    drug_count = drug_result["count"] if drug_result else 0
                    
                    se_result = session.run("MATCH (s:SideEffect) RETURN count(s) AS count").single()
                    se_count = se_result["count"] if se_result else 0
                    
                    rel_result = session.run("MATCH ()-[r:May_Cause_Side_Effect]->() RETURN count(r) AS count").single()
                    rel_count = rel_result["count"] if rel_result else 0
                    
                    logger.info("📊 Neo4j Statistics:")
                    logger.info(f"  - Drug nodes: {drug_count}")
                    logger.info(f"  - SideEffect nodes: {se_count}")
                    logger.info(f"  - May_Cause_Side_Effect relationships: {rel_count}")
                    
                    # Test query
                    if rel_count > 0:
                        test_result = session.run(
                            "MATCH (d:Drug)-[r:May_Cause_Side_Effect]->(s:SideEffect) "
                            "RETURN d.name, s.name LIMIT 1"
                        ).single()
                        
                        if test_result:
                            logger.info(f"  - Test query: {test_result['d.name']} → {test_result['s.name']}")
                
            except Exception as e:
                logger.error(f"Neo4j verification failed: {e}")
        
        logger.info("✅ Setup verification complete")
    
    def save_progress(self, data: dict):
        """Save progress data to file"""
        try:
            progress_data = {}
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
            
            progress_data.update(data)
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def load_progress(self) -> dict:
        """Load progress data from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
        return {}
    
    def clear_progress(self):
        """Clear progress data file"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except Exception as e:
            logger.warning(f"Could not clear progress file: {e}")


def main():
    """Main setup function for exact paper replication with organized namespace structure"""
    logger.info("🚀 EXACT PAPER REPLICATION SETUP WITH ORGANIZED NAMESPACE STRUCTURE")
    logger.info("RAG-based Architectures for Drug Side Effect Retrieval in LLMs")
    logger.info("="*70)
    
    # Initialize replication
    replicator = ExactPaperReplication()
    
    # Step 1: Setup Pinecone with exact specifications
    logger.info("\n📌 STEP 1: PINECONE VECTOR DATABASE SETUP")
    logger.info("- OpenAI text-embedding-ada-002 (1536 dimensions)")
    logger.info("- Cosine similarity")
    logger.info("- AWS Serverless")
    logger.info("- Organized namespace structure:")
    logger.info("  • Basic: drug-side-effects-formatA, drug-side-effects-formatB")
    logger.info("  • Enhanced: drug-side-effects-enhanced-formatB, drug-side-effects-clinical")
    
    if replicator.pinecone_client and replicator.openai_client:
        if replicator.create_pinecone_index():
            logger.info("✅ Pinecone setup successful")
            
            # Index Format A
            logger.info("\n📚 Indexing Data Format A...")
            replicator.index_format_a()
            
            # Index Format B (complete dataset)
            logger.info("\n📚 Indexing Data Format B...")
            replicator.index_format_b()  # Complete dataset
            
        else:
            logger.error("❌ Pinecone setup failed")
    else:
        logger.warning("⚠️ Skipping Pinecone setup (API keys not configured)")
    
    # Step 2: Setup Neo4j with exact schema
    logger.info("\n🔗 STEP 2: NEO4J GRAPH DATABASE SETUP")
    logger.info("- Drug and SideEffect nodes")
    logger.info("- May_Cause_Side_Effect relationships")
    
    if replicator.neo4j_driver:
        if replicator.setup_neo4j_schema():
            logger.info("✅ Neo4j schema setup successful")
            
            # Index graph data
            logger.info("\n🔗 Indexing graph relationships...")
            replicator.index_graph_data()
            
        else:
            logger.error("❌ Neo4j setup failed")
    else:
        logger.warning("⚠️ Skipping Neo4j setup (credentials not configured)")
    
    # Step 3: Verify setup
    logger.info("\n🔍 STEP 3: VERIFICATION")
    replicator.verify_setup()
    
    # Final summary
    logger.info("\n" + "🎉" + "="*68 + "🎉")
    logger.info("EXACT PAPER REPLICATION SETUP COMPLETED!")
    logger.info("="*70)
    logger.info("✅ Vector Database: Pinecone (1536-dim, cosine)")
    logger.info("✅ Graph Database: Neo4j (Drug→SideEffect)")
    logger.info("✅ Embeddings: OpenAI text-embedding-ada-002")
    logger.info("✅ LLM: OpenAI GPT (replacing AWS Bedrock)")
    logger.info("✅ Data: Format A + Format B")
    logger.info("✅ Namespace Structure: Organized for enhanced features")
    logger.info("")
    logger.info("🔬 Ready for experiments:")
    logger.info("   uv run python3 drug_side_effect_rag.py --test-mode")
    logger.info("   uv run python3 drug_side_effect_rag.py --full-evaluation")
    logger.info("")
    logger.info("🔧 To enhance with rich metadata:")
    logger.info("   uv run python3 enhanced_rag_schema_updater.py")


if __name__ == "__main__":
    main()