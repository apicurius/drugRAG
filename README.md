# DrugRAG: Drug Side Effect Retrieval System

A production-ready RAG system for identifying drug side effects using multiple retrieval architectures. This repository contains the November 2025 experiments evaluating Pure LLM (baseline), GraphRAG, Format A RAG, and Format B RAG on reverse queries, binary classification, and misspelling robustness.

## Key Features

- **Pure LLM Baseline**: Direct LLM inference without retrieval augmentation for baseline comparison
- **GraphRAG**: Graph-based retrieval using Neo4j for structured drug-side effect relationships
- **Format A RAG**: Vector-based retrieval with Pinecone for semantic search
- **Format B RAG**: Chunked retrieval strategy optimized for large result sets
- **vLLM Integration**: Ultra-fast inference using tensor parallelism (Qwen and Llama3 models)
- **Misspelling Robustness**: Evaluation of system performance with misspelled drug names
- **Spell Correction Recovery**: Assessment of spell correction impact on retrieval accuracy

## Installation

### Prerequisites
- Python 3.9+
- Neo4j (for GraphRAG)
- vLLM server (for inference)
- Pinecone account (for vector storage)
- OpenAI API key (for embeddings)

### Install Dependencies

Using pip:
```bash
pip install -e .
```

Using uv (recommended):
```bash
uv sync
```

### Configuration

Create a `config.json` file in the `experiments/` directory:
```json
{
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_password": "your_password",
  "pinecone_api_key": "your_pinecone_key",
  "pinecone_index_name": "drug-side-effects",
  "openai_api_key": "your_openai_key"
}
```

## Usage

### 1. Reverse Query Benchmark
Evaluates the system's ability to list all drugs causing a specific side effect.

```bash
./run_benchmark.sh reverse
```

### 2. Binary Query Benchmark (vLLM)
Evaluates "Does Drug X cause Side Effect Y?" using vLLM for fast inference.

```bash
# GraphRAG with Qwen
./run_benchmark.sh binary --architecture graphrag_qwen --test_size 100

# Format B with Llama3
./run_benchmark.sh binary --architecture format_b_llama3 --test_size 100

# Pure LLM baseline
./run_benchmark.sh binary --architecture pure_llm_qwen --test_size 100
```

### 3. Misspelling Evaluation
Tests robustness against misspelled drug names.

```bash
./run_benchmark.sh misspelling
```

## November 2025 Experiment Results

### Reverse Query Performance

Comparison of architectures on the reverse query task (side effect → list of drugs).

> **Source:** `experiments/results_reverse_query_benchmark_20251128_235601.json`

| Architecture | Avg Recall | Avg Precision | Avg F1 | Avg Latency | Queries |
|-------------|-----------|--------------|--------|-------------|---------|
| GraphRAG | 100.00% | 100.00% | 100.00% | 0.09s | 121 |
| Format B (Chunked) | 98.59% | 99.93% | 99.18% | 84.63s | 121 |
| Format A | 7.97% | 81.03% | 11.79% | 23.32s | 121 |

**Key Findings:**
- **GraphRAG achieves perfect performance** (100% recall, 100% precision) with ultra-fast 0.09s latency
- Format B achieves excellent 98.59% recall with chunked retrieval strategy
- **Format A fails on reverse queries** (only 7.97% recall) - not suitable for this task
- GraphRAG is ~900x faster than Format B for reverse queries (0.09s vs 84.6s)


### Binary Classification Performance (19,520 Queries)

Performance on binary queries ("Is [side effect] an adverse effect of [drug]?") using the full evaluation dataset.

> **Source:** `experiments/results_vllm_*_19520_*.json` files (Nov 25-28, 2025)

| Architecture | Model | Accuracy | F1 Score | Precision | Sensitivity | Specificity |
|-------------|-------|----------|----------|-----------|-------------|-------------|
| GraphRAG | Qwen | 99.95% | 0.999 | 0.999 | 0.999 | 0.999 |
| GraphRAG | Llama3 | 99.96% | 1.000 | 0.999 | 1.000 | 0.999 |
| Format B | Qwen | 99.71% | 0.997 | 0.997 | 0.997 | 0.997 |
| Format B | Llama3 | 98.36% | 0.983 | 0.998 | 0.970 | 0.998 |
| Format A | Qwen | 90.86% | 0.900 | 0.997 | 0.820 | 0.997 |
| Format A | Llama3 | 86.58% | 0.845 | 0.998 | 0.733 | 0.998 |
| Pure LLM | Qwen | 62.93% | 0.494 | 0.777 | 0.363 | 0.896 |
| Pure LLM | Llama3 | 63.19% | 0.535 | 0.726 | 0.423 | 0.840 |

**Key Findings:**
- **GraphRAG achieves near-perfect accuracy** with both models (99.95% Qwen, 99.96% Llama3)
- **Format B achieves excellent performance** (99.71% Qwen, 98.36% Llama3)
- Format A shows good accuracy (90.86% Qwen, 86.58% Llama3)
- **Pure LLM baseline performs at 62-63% accuracy**, demonstrating the critical need for retrieval augmentation
- RAG architectures provide **37-38% absolute improvement** over Pure LLM baseline

### Misspelling Robustness (180-Query Test Set)

System performance degradation with misspelled drug names.

> **Source:** `results/misspelling_experiment/summary_report_20251129_155639.txt`

| Architecture | Correct Spelling | Misspelled | Accuracy Drop | F1 Degradation |
|-------------|-----------------|------------|---------------|----------------|
| GraphRAG | 100.00% | 50.00% | -50.00% | -100.00% |
| Format B | 100.00% | 50.00% | -50.00% | -100.00% |
| Format A | 90.00% | 50.00% | -44.44% | -100.00% |
| Pure LLM | 60.56% | 62.78% | +3.67% | +8.66% |

**Key Findings:**
- RAG architectures fail to retrieve with misspelled drug names (drop to 50% - random guessing)
- **Pure LLM is robust to misspellings** (slight improvement from 60.56% to 62.78%) as it relies on learned knowledge rather than exact retrieval matching
- Misspelling severely impacts retrieval-based systems due to embedding/graph lookup failures
- This highlights critical need for spell correction in production RAG systems

### Spell Correction Recovery (180-Query Test Set)

Impact of LLM-based spell correction on misspelled queries.

> **Source:** `results/spell_correction_recovery/recovery_summary_20251129_163320.txt`

| Architecture | Perfect F1 | Raw Misspelled F1 | With Spell Correction F1 | Recovery Rate |
|-------------|---------|----------------|----------------------|---------------|
| GraphRAG | 1.000 | 0.000 | 0.875 | 87.5% |
| Format B | 1.000 | 0.000 | 0.875 | 87.5% |
| Format A | 0.889 | 0.000 | 0.732 | 82.4% |
| Pure LLM | 0.493 | 0.433 | 0.485 | 87.6% |

**Key Findings:**
- Spell correction recovers 82-88% of performance for RAG systems
- GraphRAG and Format B both achieve 87.5% F1 score after correction
- Format A recovers to 73.2% F1 (82.4% recovery rate)
- **Pure LLM maintains consistent performance** (~0.49 F1) regardless of spelling, as it doesn't rely on retrieval
- Spell correction is highly effective for production RAG deployment


## Repository Structure

```
drugRAG/
├── README.md                    # This file
├── pyproject.toml              # Dependencies and project metadata
├── run_evaluations.sh          # Unified benchmark runner
├── data/
│   └── processed/              # Datasets and ground truth
├── experiments/
│   ├── reverse_query_benchmark.py      # Reverse query evaluation
│   ├── evaluate_vllm.py                # Binary classification with vLLM
│   ├── evaluate_misspelling.py         # Misspelling robustness test
│   └── evaluate_spell_correction.py    # Spell correction recovery
├── results/
│   ├── misspelling_experiment/         # Misspelling results (JSON/CSV/TXT)
│   ├── spell_correction_experiment/    # Spell correction results
│   └── spell_correction_recovery/      # Recovery analysis results
├── scripts/
│   ├── generate_comprehensive_dataset.py         # Dataset generation
│   ├── generate_comprehensive_dataset_parallel.py
│   ├── generate_ground_truth_neo4j.py            # Neo4j ground truth
│   └── fix_dataset_case_sensitivity.py
└── src/
    ├── architectures/
    │   ├── graphrag.py          # GraphRAG implementation
    │   ├── rag_format_a.py      # Format A RAG
    │   └── rag_format_b.py      # Format B RAG (chunked)
    ├── models/
    │   └── vllm_model.py        # vLLM client for Qwen/Llama3
    ├── utils/
    │   ├── database.py          # Neo4j database utilities
    │   ├── embedding_client.py  # OpenAI embedding client
    │   ├── spell_corrector.py   # Spell correction utilities
    │   └── ...                  # Other utilities
    └── evaluation/
        ├── metrics.py           # Evaluation metrics
        └── advanced_metrics.py  # Advanced analysis
```

## Reproducibility

All experiments use the following settings for reproducibility:

- **Random Seed**: 42 (set in all benchmark scripts)
- **Dataset**: Comprehensive reverse queries (case-corrected, November 2025)
- **Ground Truth**: Neo4j-based ground truth with drug-side effect pairs
- **Binary Test Set**: 19,520 balanced queries (9,760 positive, 9,760 negative examples)
- **Misspelling Test Set**: 180 balanced queries (90 positive, 90 negative examples)
- **Models**: Qwen2.5-7B-Instruct and Llama3-8B via vLLM

### Expected Performance Metrics

Based on November 2025 experiments (latest results from Nov 25-28, 2025):

**Binary Classification (19,520 Queries):**
- GraphRAG Qwen: 99.95% accuracy, 0.999 F1 score
- GraphRAG Llama3: 99.96% accuracy, 1.000 F1 score
- Format B Qwen: 99.71% accuracy, 0.997 F1 score
- Format B Llama3: 98.36% accuracy, 0.983 F1 score
- Format A Qwen: 90.86% accuracy, 0.900 F1 score
- Format A Llama3: 86.58% accuracy, 0.845 F1 score
- Pure LLM Qwen: 62.93% accuracy, 0.494 F1 score
- Pure LLM Llama3: 63.19% accuracy, 0.535 F1 score

**Misspelling Robustness (180 Queries):**
- RAG systems drop to 50% accuracy (random guessing) with misspelled drug names
- Spell correction recovers 82-88% of original performance
- GraphRAG/Format B achieve 87.5% F1 after spell correction


## Citation

If you use this code or findings in your research, please cite:

```bibtex
@software{drugrag2025,
  title={DrugRAG: Drug Side Effect Retrieval with RAG and GraphRAG},
  author={DrugRAG Team},
  year={2025},
  month={November},
  url={https://github.com/apicurius/drugRAG}
}
```

## License

MIT License - see LICENSE file for details.
