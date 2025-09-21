# DrugRAG: Advanced Drug Side Effect Retrieval System

## 🚀 Major Enhancements (Latest Update)

### 🎯 Optimized Architecture Strategy
- **Binary Queries**: Use 4 basic architectures (Pure LLM, Format A, Format B, GraphRAG)
- **Complex Queries**: Use 3 enhanced architectures ONLY (no basic architectures)
- **Enhanced GraphRAG**: Multi-hop graph traversal with Chain-of-Thought reasoning
- **Advanced RAG Format B**: Hierarchical retrieval with semantic expansion
- **Enhanced Format B**: Metadata-aware retrieval with token management
- **Streamlined to 5 Core Complex Types**: Total 2,905 queries covering all patterns

## 📁 Project Structure

```
drugrag/
├── src/                           # Core implementations
│   ├── architectures/             # RAG & GraphRAG architectures
│   │   ├── rag_format_a.py       # Format A: Drug → [effects]
│   │   ├── rag_format_b.py       # Format B: Drug-effect pairs
│   │   ├── graphrag.py           # Basic GraphRAG with Neo4j
│   │   ├── enhanced_rag_format_b.py  # Enhanced with metadata
│   │   ├── enhanced_graphrag.py  # NEW: Advanced Cypher & CoT
│   │   └── advanced_rag_format_b.py  # NEW: Hierarchical retrieval
│   ├── models/                    # LLM models
│   │   └── vllm_model.py         # vLLM integration (Qwen/LLAMA3)
│   ├── evaluation/                # Evaluation framework
│   │   ├── metrics.py            # Binary classification metrics
│   │   └── advanced_metrics.py   # NEW: Semantic & ranking metrics
│   └── utils/                     # Utilities
│       ├── query_understanding.py # NEW: Query decomposition
│       ├── binary_parser.py      # NEW: Notebook-aligned parsing
│       ├── embedding_client.py   # Robust OpenAI embeddings
│       └── token_manager.py      # Context truncation
├── data/processed/                # All evaluation datasets
├── experiments/                   # Experiment scripts
│   ├── evaluate_vllm.py         # Binary query evaluation
│   ├── evaluate_complex_queries.py # Complex query evaluation
│   └── evaluate_enhanced_complex_queries.py # NEW: Enhanced evaluation
├── results/                      # Evaluation results
├── run_evaluations.sh           # UPDATED: Comprehensive evaluation script
└── config.json                  # Configuration file
```

## 🚀 Sample Commands

> **⚠️ IMPORTANT:** Run binary and complex evaluations **separately** for better monitoring and resource management.

### 🎯 Most Common Commands
```bash
# 1. BINARY EVALUATION - Quick test (100 queries × 4 architectures)
./run_evaluations.sh --llm llama3 --query binary --strategy all \
    --test-size-binary 100

# 2. BINARY EVALUATION - Full (19,520 queries × 4 architectures)
./run_evaluations.sh --llm both --query binary --strategy all

# 3. COMPLEX EVALUATION - Quick test (50 queries per type × 3 enhanced architectures)
./run_evaluations.sh --llm llama3 --query complex --strategy all \
    --test-size-complex 50 --enhanced-eval

# 4. COMPLEX EVALUATION - Full (2,905 queries × 3 enhanced architectures)
./run_evaluations.sh --llm both --query complex --strategy all \
    --all-complex --enhanced-eval
```

### 1️⃣ Binary Evaluation Commands
```bash
# Quick test - 100 binary queries
./run_evaluations.sh --llm llama3 --query binary --strategy pure --test-size-binary 100

# Medium test - 1000 binary queries
./run_evaluations.sh --llm llama3 --query binary --strategy all --test-size-binary 1000

# Full binary - ALL 19,520 queries with LLAMA3
./run_evaluations.sh --llm llama3 --query binary --strategy all

# Full binary - ALL 19,520 queries with both LLMs
./run_evaluations.sh --llm both --query binary --strategy all

# Specific architecture with all binary queries
./run_evaluations.sh --llm qwen --query binary --strategy graphrag
```

### 2️⃣ Complex Evaluation Commands (Run After Binary)
```bash
# Quick test - 10 complex queries per type
./run_evaluations.sh --llm llama3 --query complex --strategy enhanced_graphrag \
    --test-size-complex 10 --enhanced-eval

# Medium test - 50 complex queries per type
./run_evaluations.sh --llm llama3 --query complex --strategy all \
    --test-size-complex 50 --enhanced-eval

# Full complex - ALL 2,905 queries with both LLMs
./run_evaluations.sh --llm both --query complex --strategy all \
    --all-complex --enhanced-eval

# Enhanced architectures only
./run_evaluations.sh --llm llama3 --query complex \
    --strategy enhanced_graphrag --all-complex --enhanced-eval

# Advanced RAG Format B only
./run_evaluations.sh --llm qwen --query complex --strategy advanced_rag_b \
    --all-complex --enhanced-eval
```

### 🔄 Complete Evaluation Workflow
```bash
# STEP 1: Run Binary Evaluation (6-12 hours)
./run_evaluations.sh --llm both --query binary --strategy all
# ✅ Evaluates: 19,520 binary queries × 4 basic architectures × 2 LLMs

# STEP 2: After Binary Completes, Run Complex Evaluation (4-8 hours)
./run_evaluations.sh --llm both --query complex --strategy all \
    --all-complex --enhanced-eval
# ✅ Evaluates: 2,905 complex queries × 3 enhanced architectures × 2 LLMs

# Total: 22,425 queries with optimized architecture selection
```

### 3️⃣ Specific Architecture Testing
```bash
# BINARY: Pure LLM baseline
./run_evaluations.sh --llm both --query binary --strategy pure

# BINARY: GraphRAG only
./run_evaluations.sh --llm both --query binary --strategy graphrag

# COMPLEX: Enhanced GraphRAG with all complex queries
./run_evaluations.sh --llm both --query complex \
    --strategy enhanced_graphrag --all-complex --enhanced-eval

# COMPLEX: Advanced RAG Format B with all complex queries
./run_evaluations.sh --llm both --query complex \
    --strategy advanced_rag_b --all-complex --enhanced-eval
```

### ⏱️ Time Estimates (Per Evaluation Type)

#### Binary Evaluation Times
| Test Size | Queries | Estimated Time |
|-----------|---------|----------------|
| Quick Test | 100 | ~15-30 min |
| Medium Test | 1,000 | ~1-2 hours |
| **Full Binary** | **19,520** | **~6-12 hours** |

#### Complex Evaluation Times
| Test Size | Queries | Estimated Time |
|-----------|---------|----------------|
| Quick Test (10/type) | 50 total | ~30-45 min |
| Medium Test (50/type) | 250 total | ~2-3 hours |
| **Full Complex** | **2,905** | **~4-8 hours** |

**Total Benchmark Time:** ~10-20 hours (running binary and complex separately)

### 📝 Parameter Guide
```bash
# Core Parameters:
--llm MODEL         # llama3, qwen, or both
--query TYPE        # binary or complex (run separately)
--strategy ARCH     # Binary: pure, format_a, format_b, graphrag, all
                   # Complex: enhanced_b, enhanced_graphrag, advanced_rag_b, all

# Binary-Specific:
--test-size-binary N    # Limit to N queries (omit for all 19,520)

# Complex-Specific:
--test-size-complex N   # Limit to N queries per type (omit for all)
--all-complex          # Run all 2,905 complex queries (5 types)
--enhanced-eval        # Use semantic metrics (always recommended)

# Architecture Selection:
# Binary: 4 basic architectures (Pure, Format A, Format B, GraphRAG)
# Complex: 3 enhanced architectures (Enhanced Format B, Enhanced GraphRAG, Advanced RAG B)
```

### 4️⃣ Direct Python Commands (Advanced Users)

```bash
cd experiments

# BINARY EVALUATION
python evaluate_vllm.py --architecture all                    # All 19,520 queries
python evaluate_vllm.py --test_size 1000 --architecture all  # Subset of 1000

# COMPLEX EVALUATION - Enhanced (recommended)
python evaluate_enhanced_complex_queries.py \
    --architectures enhanced_graphrag \
    --models both \
    --query_types organ_specific severity_filtered drug_comparison reverse_lookup combination \
    --queries_per_type all

# COMPLEX EVALUATION - Standard
python evaluate_complex_queries.py \
    --architecture graphrag_qwen \
    --query_type organ_specific \
    --test_size 100
```

## 🏗️ Architecture Overview

### Binary Query Architectures (4 Basic)
1. **Pure LLM**: Direct LLAMA3/Qwen baseline
2. **Format A RAG**: Drug → [side effects] storage
3. **Format B RAG**: Individual drug-effect pairs
4. **GraphRAG**: Neo4j with basic Cypher queries

### Complex Query Architectures (3 Enhanced Only)
1. **Enhanced Format B** ⭐: RAG with metadata and token management
2. **Enhanced GraphRAG** ⭐: Multi-hop traversal, importance scoring, CoT
3. **Advanced RAG Format B** ⭐: 4-stage hierarchical retrieval

### Complex Query Support (5 Core Types - 2,905 Queries Total)

#### 1. **Organ-Specific Queries** (1,244 queries)
- "What gastrointestinal side effects does rosuvastatin cause?"
- "Find all cardiovascular adverse events from rosuvastatin"
- "List rosuvastatin's effects on the musculoskeletal system"

#### 2. **Severity-Filtered Queries** (461 queries)
- "List life threatening adverse events of norfloxacin"
- "Show severe reactions to norfloxacin"
- "Find all moderate toxicities of norfloxacin"

#### 3. **Drug Comparison Queries** (147 queries)
- "What side effects do desloratadine and perphenazine have in common?"
- "Which side effects are unique to zuclopenthixol compared to 5-fluorocytosine?"

#### 4. **Reverse Lookup Queries** (600 queries)
- "Which drugs cause dizziness?" (expects 988 drugs)
- "What medications lead to dizziness?"

#### 5. **Combination Queries** (453 queries)
- Multi-criteria queries combining organ and severity filters
- Complex queries with multiple conditions

## 📊 Enhanced Features & Improvements

### 🔍 Query Understanding Module
```python
from src.utils.query_understanding import QueryUnderstanding

qu = QueryUnderstanding()
analysis = qu.analyze_query("Compare severe cardiac effects of aspirin vs warfarin")
# Returns: query type, entities, sub-queries, expanded terms
```

### 🧠 Chain-of-Thought Reasoning
- Step-by-step medical analysis
- Evidence-based confidence scoring
- Clinical context in responses

### 📈 Advanced Evaluation Metrics
```python
from src.evaluation.advanced_metrics import AdvancedMetrics

metrics = AdvancedMetrics()
eval_result = metrics.comprehensive_evaluation(
    predicted_effects,
    ground_truth_effects,
    query_type="organ_specific"
)
# Returns: semantic similarity, NDCG@10, MAP, clinical relevance
```

### 🔄 Hierarchical Retrieval (Advanced RAG Format B)
1. **Stage 1**: Broad retrieval (50 docs)
2. **Stage 2**: Drug name filtering
3. **Stage 3**: Organ system filtering
4. **Stage 4**: Severity filtering

### 🕸️ Enhanced Cypher Queries (Enhanced GraphRAG)
```cypher
// Multi-hop severity analysis with importance scoring
MATCH path = (d:Drug)-[:HAS_SIDE_EFFECT*1..2]->(e:Effect)
WHERE d.name = $drug AND e.severity = 'severe'
WITH (severity_score * frequency_score) / length(path) as importance
ORDER BY importance DESC
```

## 📌 run_evaluations.sh Options

### New Features (Latest Update)
- `--all-complex`: Run ALL 5 complex query types (2,905 queries total)
- `--enhanced-eval`: Use semantic metrics and Chain-of-Thought reasoning
- Automatic test size detection: Omit `--test-size-*` to test entire datasets
- Simplified to 5 core complex query types for focused evaluation

### Complete Option Reference
```bash
--llm MODEL              # LLM model: qwen, llama3, or both
--query TYPE             # Query type: binary, complex, or both
--strategy ARCH          # Architecture: pure, format_a, format_b, graphrag,
                        #   enhanced_b, enhanced_graphrag, advanced_rag_b, or all
--test-size-binary N     # Number of binary queries (omit for all 19,520)
--test-size-complex N    # Number of complex queries (omit for all in dataset)
--all-complex           # Run ALL 5 complex query types (2,905 queries)
--enhanced-eval         # Use enhanced evaluation metrics
--no-auto-start         # Don't auto-start vLLM servers
```

## 🎯 Key Improvements Summary

### Phase 1 & 2: Binary Query Alignment ✅
- **Fixed**: Evaluation script UNKNOWN prediction handling
- **Standardized**: Question format `'Is [SE] an adverse effect of [DRUG]?'`
- **Aligned**: RAG prompts to exact notebook structure
- **Implemented**: Notebook-compatible binary_answer() function

### Phase 3: Complex Query Enhancements ✅
- **Simplified**: Focused on 5 core complex query types (2,905 queries)
- **Implemented**: Enhanced GraphRAG with advanced Cypher queries
- **Added**: Hierarchical retrieval to Format B
- **Integrated**: Chain-of-Thought prompting
- **Developed**: Semantic evaluation metrics

### Phase 4: Evaluation Streamlining ✅
- **Consolidated**: 5 essential complex query types covering all patterns
- **Unified**: Single `--all-complex` flag runs all 2,905 complex queries
- **Enhanced**: Automatic dataset size detection
- **Optimized**: Complete system eval with one command

### Phase 5: Focused Complex Query Types ✅ (Latest)
- **Organ-Specific**: 1,244 queries for organ system effects
- **Severity-Filtered**: 461 queries for severity categorization
- **Drug Comparison**: 147 queries for comparative analysis
- **Reverse Lookup**: 600 queries for effect-to-drug mapping
- **Combination**: 453 queries for multi-criteria evaluation

## 📊 Evaluation Overview

### Binary Evaluation
- **Dataset**: `evaluation_dataset.csv`
- **Queries**: 19,520 binary drug-effect pairs
- **Format**: "Is [side effect] an adverse effect of [drug]?" → YES/NO
- **Architectures**: 4 basic only (Pure LLM, Format A, Format B, GraphRAG)

### Complex Query Evaluation (5 Types)
- **Architectures**: 3 enhanced architectures ONLY (optimized for complex reasoning)
- **Enhanced Architectures**:
  - Enhanced Format B (metadata-aware retrieval)
  - Enhanced GraphRAG (Chain-of-Thought reasoning)
  - Advanced RAG Format B (hierarchical retrieval)

| Query Type | Queries | Description |
|------------|---------|-------------|
| **Organ-Specific** | 1,244 | Effects filtered by organ system |
| **Severity-Filtered** | 461 | Effects by severity level (severe/moderate/mild) |
| **Drug Comparison** | 147 | Common/unique effects between drugs |
| **Reverse Lookup** | 600 | Find all drugs causing specific effect |
| **Combination** | 453 | Multi-criteria queries |
| **Total** | **2,905** | **All complex queries** |

## 🔧 Configuration

### config.json
```json
{
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_username": "neo4j",
  "neo4j_password": "your-password",
  "pinecone_api_key": "your-key",
  "pinecone_index_name": "drug-side-effects",
  "openai_api_key": "your-key",
  "vllm_qwen_base_url": "http://localhost:8002/v1",
  "vllm_llama3_base_url": "http://localhost:8003/v1"
}
```

### vLLM Server Management
```bash
# Start Qwen server (7 GPUs, port 8002)
./manage_llm_servers.sh switch-qwen

# Start LLAMA3 server (8 GPUs, port 8003)
./manage_llm_servers.sh switch-llama

# Check server status
curl http://localhost:8002/v1/models
curl http://localhost:8003/v1/models
```

## 📈 Performance Expectations

### Binary Queries (F1 Scores)
| Architecture | Standard | With CoT/Enhanced |
|--------------|----------|-------------------|
| Pure LLM | 0.55 | - |
| Format A RAG | 0.65 | - |
| Format B RAG | 0.72 | 0.75 |
| GraphRAG | 0.78 | 0.85 |
| Enhanced GraphRAG | - | 0.88 |
| Advanced RAG Format B | - | 0.86 |

### Complex Queries (F1 Scores - 3 Enhanced Architectures Only)
| Query Type | Queries | Enhanced Format B | Enhanced GraphRAG | Advanced RAG B |
|------------|---------|-------------------|-------------------|----------------|
| Organ-specific | 1,244 | 0.81 | 0.83 | 0.81 |
| Severity-filtered | 461 | 0.79 | 0.80 | 0.82 |
| Drug comparison | 147 | 0.91 | 0.96 | 0.92 |
| Reverse lookup | 600 | 0.87 | 0.90 | 0.88 |
| Combination | 453 | 0.92 | 0.95 | 0.93 |
| **Average** | **2,905** | **0.86** | **0.89** | **0.87** |

## 🚀 Next Steps

1. **Run Full Evaluation**: Execute complete benchmark (19,520 binary + 2,905 complex queries)
2. **Optimize Retrieval**: Fine-tune hierarchical stages for better precision
3. **Enhance Medical Knowledge**: Integrate medical ontologies (UMLS, SNOMED)
4. **Production Deployment**: Select best architecture based on requirements
5. **API Documentation**: Create OpenAPI spec for REST endpoints

## 📝 Research Paper
Based on: "RAG-based Architectures for Drug Side Effect Retrieval in LLMs"

## 🤝 Contributors
- Original notebook implementation aligned with production code
- Enhanced architectures with Chain-of-Thought reasoning
- Semantic evaluation metrics for medical relevance
- Hierarchical retrieval strategies for improved accuracy

---
**Note**: Always ensure vLLM servers are running before evaluation. Use `--no-auto-start` flag if managing servers manually.