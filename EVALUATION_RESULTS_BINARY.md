# DrugRAG Binary Query Evaluation Results
**Dataset**: 19,520 Drug-Side Effect Pairs
**Date**: September 20, 2025
**Hardware**: 4x NVIDIA RTX A4000 GPUs with Tensor Parallelism

## 📊 Performance Summary

### Qwen2.5-7B-Instruct Results

| Architecture | Accuracy | F1 Score | Precision | Sensitivity | Specificity | Status |
|--------------|----------|----------|-----------|-------------|-------------|---------|
| **Pure LLM** | 62.90% | 0.494 | 0.776 | 0.363 | 0.895 | ✅ Working |
| **Format A (Drug→Effects)** | 86.67% | 0.858 | 0.919 | 0.805 | 0.928 | ✅ Working |
| **Format B (Drug-Effect Pairs)** | 96.50% | 0.967 | 0.936 | 0.999 | 0.931 | ✅ Excellent |
| **GraphRAG (Neo4j)** | 100.00% | 1.000 | 1.000 | 1.000 | 1.000 | ✅ Perfect |

**Throughput**: ~45 queries/second (90x speedup vs single GPU)

### Llama-3.1-8B-Instruct Results (Fixed Configuration)

| Architecture | Accuracy | F1 Score | Precision | Sensitivity | Specificity | Status |
|--------------|----------|----------|-----------|-------------|-------------|---------|
| **Pure LLM** | 63.21% | 0.534 | 0.728 | 0.422 | 0.842 | ✅ Working |
| **Format A (Drug→Effects)** | 84.54% | 0.819 | 0.987 | 0.700 | 0.991 | ✅ Working |
| **Format B (Drug-Effect Pairs)** | 95.86% | 0.960 | 0.924 | 0.999 | 0.918 | ✅ Excellent |
| **GraphRAG (Neo4j)** | 99.96% | 1.000 | 1.000 | 1.000 | 1.000 | ✅ Perfect |

**Throughput**: ~100+ queries/second (stable performance)

## 🔍 Detailed Analysis

### Model Comparison (Both Working Now!)

| Architecture | Qwen2.5-7B | Llama-3.1-8B | Difference |
|--------------|------------|--------------|------------|
| **Pure LLM** | 62.90% | 63.21% | Llama +0.31% |
| **Format A** | 86.67% | 84.54% | Qwen +2.13% |
| **Format B** | 96.50% | 95.86% | Qwen +0.64% |
| **GraphRAG** | 100.00% | 99.96% | Qwen +0.04% |

### Key Findings
1. **Both models perform similarly** with fixed configurations
2. **GraphRAG**: Near-perfect for both models (99.96-100%)
3. **Format B RAG**: Excellent for both (95.86-96.50%)
4. **Format A RAG**: Good for both (84.54-86.67%)
5. **Pure LLM**: Baseline similar (~63%) for both models

## 🛠️ Server Configurations

### Qwen Server (Port 8002) - Stable ✅
```bash
--model Qwen/Qwen2.5-7B-Instruct
--tensor-parallel-size 4
--dtype float16
--max-model-len 4096
--gpu-memory-utilization 0.90
--enable-chunked-prefill
--max-num-batched-tokens 8192
--enforce-eager
--max-num-seqs 256
--distributed-executor-backend mp
```

### Llama Server (Port 8003) - Fixed Configuration
```bash
--model meta-llama/Llama-3.1-8B-Instruct
--tensor-parallel-size 4
--dtype float16
--max-model-len 4096  # Reduced from 8192
--gpu-memory-utilization 0.90
--enable-chunked-prefill
--max-num-batched-tokens 8192  # Reduced from 16384
--enforce-eager  # Added to prevent CUDA graph issues
--max-num-seqs 256
--distributed-executor-backend mp
# Removed: --swap-space 16 (caused crashes)
```

## 📈 Key Findings

### Architecture Rankings (Based on Qwen Results)
1. **GraphRAG**: 100% accuracy - Best overall
2. **Format B RAG**: 96.5% accuracy - Best practical solution
3. **Format A RAG**: 86.7% accuracy - Good alternative
4. **Pure LLM**: 62.9% accuracy - Baseline performance

### Performance Summary
- ✅ **Both models now working perfectly** after configuration fixes
- ✅ **GraphRAG achieves near-100% accuracy** on both models
- ✅ **Format B RAG exceeds 95% accuracy** on both models
- ✅ **Consistent performance** across Qwen and Llama models

## 🚀 Recommendations

1. **Production Use**: Deploy either Format B RAG (96% accuracy) or GraphRAG (100% accuracy)
2. **Model Choice**: Both Qwen and Llama perform similarly - choose based on licensing/deployment needs
3. **Best Balance**: Format B RAG offers excellent accuracy (96%) with simpler infrastructure than GraphRAG
4. **Maximum Accuracy**: GraphRAG with Neo4j achieves near-perfect results

## 📝 Test Commands

### Quick Test (100 samples)
```bash
# Qwen
./run_evaluations.sh --llm qwen --query binary --strategy all --test-size-binary 100

# Llama3 (with fixed config)
./run_evaluations.sh --llm llama3 --query binary --strategy all --test-size-binary 100
```

### Full Evaluation (19,520 samples)
```bash
# Both models, all architectures
./run_evaluations.sh --llm both --query binary --strategy all
```

## 📊 Confusion Matrix Details

### Best Performer: Qwen Format B
- **True Positives**: 9,747
- **True Negatives**: 9,089
- **False Positives**: 671
- **False Negatives**: 13
- **Sensitivity**: 99.9% (excellent at detecting true side effects)
- **Specificity**: 93.1% (good at rejecting false associations)

---
*Last Updated: September 20, 2025*
*Status: All evaluations complete and successful*