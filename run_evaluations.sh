#!/bin/bash

# Comprehensive DrugRAG Evaluation Script with Enhanced Architectures
# Supports all architectures including Enhanced GraphRAG and Advanced RAG Format B
# Features Chain-of-Thought reasoning and semantic evaluation metrics

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Default values
DEFAULT_LLM="llama3"
DEFAULT_QUERY="both"
DEFAULT_STRATEGY="all"
DEFAULT_TEST_SIZE_BINARY=50
DEFAULT_TEST_SIZE_COMPLEX=25
DEFAULT_COMPLEX_DATASET="complex_query_dataset.csv"

# Variables
LLM=""
QUERY_TYPE=""
STRATEGY=""
TEST_SIZE_BINARY=""
TEST_SIZE_COMPLEX=""
COMPLEX_DATASET=""
AUTO_START_SERVER=true
USE_ENHANCED_EVAL=false
RUN_ALL_COMPLEX=false
USE_BATCH_PROCESSING=false
BATCH_SIZE=50

# Function to show usage
show_usage() {
    echo "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        ENHANCED DRUGRAG EVALUATION                               ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --llm MODEL           LLM model to use (qwen|llama3|both) [default: $DEFAULT_LLM]"
    echo "  --query TYPE          Query type (binary|complex|both) [default: $DEFAULT_QUERY]"
    echo "  --strategy ARCH       Architecture strategy [default: $DEFAULT_STRATEGY]"
    echo "                        Binary: pure|format_a|format_b|graphrag|all"
    echo "                        Complex: all above + enhanced_b|enhanced_graphrag|advanced_rag_b"
    echo "  --test-size-binary N  Number of binary queries [default: $DEFAULT_TEST_SIZE_BINARY]"
    echo "                        Omit to test all 19,520 queries"
    echo "  --test-size-complex N Number of complex queries [default: $DEFAULT_TEST_SIZE_COMPLEX]"
    echo "                        Omit to test all queries in the dataset"
    echo "  --complex-dataset FILE Complex query dataset to use [default: $DEFAULT_COMPLEX_DATASET]"
    echo "  --all-complex         Run ALL 5 complex query types (3,022 queries total)"
    echo "  --batch               Enable batch processing for complex queries (4 GPU acceleration)"
    echo "  --batch-size N        Batch size for GPU processing [default: 50]"
    echo "  --no-auto-start       Don't automatically start vLLM servers"
    echo "  --enhanced-eval       Use enhanced evaluation with semantic metrics (for complex)"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Strategy Descriptions:"
    echo "  pure            : Pure LLM without RAG (binary only)"
    echo "  format_a        : RAG Format A - Drug → [side effects] (binary only)"
    echo "  format_b        : RAG Format B - Drug-effect pairs (binary only)"
    echo "  graphrag        : GraphRAG with Neo4j (binary only)"
    echo "  enhanced_b      : Enhanced Format B with metadata (complex only)"
    echo "  enhanced_graphrag : Enhanced GraphRAG with CoT reasoning (complex only)"
    echo "  advanced_rag_b  : Advanced RAG B with hierarchical retrieval (complex only)"
    echo "  all             : Binary: 4 basic architectures | Complex: 3 enhanced architectures"
    echo ""
    echo "Examples:"
    echo "  # Binary evaluation (4 basic architectures)"
    echo "  $0 --llm llama3 --query binary --strategy all --test-size-binary 100"
    echo "  $0 --llm both --query binary --strategy all  # All 19,520 queries"
    echo ""
    echo "  # Complex evaluation (3 enhanced architectures only)"
    echo "  $0 --llm llama3 --query complex --strategy enhanced_graphrag --test-size-complex 50"
    echo "  $0 --llm both --query complex --strategy all --all-complex --enhanced-eval"
    echo ""
    echo "  # NOTE: Binary uses basic architectures, Complex uses enhanced architectures"
    echo ""
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --llm)
                LLM="$2"
                shift 2
                ;;
            --query)
                QUERY_TYPE="$2"
                shift 2
                ;;
            --strategy)
                STRATEGY="$2"
                shift 2
                ;;
            --test-size-binary)
                TEST_SIZE_BINARY="$2"
                shift 2
                ;;
            --test-size-complex)
                TEST_SIZE_COMPLEX="$2"
                shift 2
                ;;
            --complex-dataset)
                COMPLEX_DATASET="$2"
                shift 2
                ;;
            --all-complex)
                RUN_ALL_COMPLEX=true
                shift
                ;;
            --no-auto-start)
                AUTO_START_SERVER=false
                shift
                ;;
            --enhanced-eval)
                USE_ENHANCED_EVAL=true
                shift
                ;;
            --batch)
                USE_BATCH_PROCESSING=true
                shift
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_color $RED "❌ Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done

    # Set defaults for unspecified parameters
    LLM=${LLM:-$DEFAULT_LLM}
    QUERY_TYPE=${QUERY_TYPE:-$DEFAULT_QUERY}
    STRATEGY=${STRATEGY:-$DEFAULT_STRATEGY}
    TEST_SIZE_BINARY=${TEST_SIZE_BINARY:-$DEFAULT_TEST_SIZE_BINARY}
    TEST_SIZE_COMPLEX=${TEST_SIZE_COMPLEX:-$DEFAULT_TEST_SIZE_COMPLEX}
    COMPLEX_DATASET=${COMPLEX_DATASET:-$DEFAULT_COMPLEX_DATASET}
}

# Function to validate arguments
validate_arguments() {
    # Validate LLM
    case $LLM in
        qwen|llama3|both) ;;
        *)
            print_color $RED "❌ Invalid LLM: $LLM. Must be: qwen, llama3, or both"
            exit 1
            ;;
    esac

    # Validate query type
    case $QUERY_TYPE in
        binary|complex|both) ;;
        *)
            print_color $RED "❌ Invalid query type: $QUERY_TYPE. Must be: binary, complex, or both"
            exit 1
            ;;
    esac

    # Validate strategy
    case $STRATEGY in
        pure|format_a|format_b|graphrag|enhanced_b|enhanced_graphrag|advanced_rag_b|all)
            # Check if enhanced architectures are used with binary queries
            if [[ "$QUERY_TYPE" == "binary" ]] && [[ "$STRATEGY" =~ ^(enhanced_b|enhanced_graphrag|advanced_rag_b)$ ]]; then
                print_color $RED "❌ Enhanced architectures ($STRATEGY) are only for complex queries."
                print_color $YELLOW "   For binary queries, use: pure, format_a, format_b, graphrag, or all"
                exit 1
            fi
            # Check if basic architectures are used with complex queries (except "all")
            if [[ "$QUERY_TYPE" == "complex" ]] && [[ "$STRATEGY" =~ ^(pure|format_a|format_b|graphrag)$ ]]; then
                print_color $RED "❌ Basic architectures ($STRATEGY) are not used for complex queries."
                print_color $YELLOW "   For complex queries, use: enhanced_b, enhanced_graphrag, advanced_rag_b, or all"
                exit 1
            fi
            ;;
        *)
            print_color $RED "❌ Invalid strategy: $STRATEGY. Must be: pure, format_a, format_b, graphrag, enhanced_b, enhanced_graphrag, advanced_rag_b, or all"
            exit 1
            ;;
    esac

    # Validate test sizes
    if ! [[ "$TEST_SIZE_BINARY" =~ ^[0-9]+$ ]] || [ "$TEST_SIZE_BINARY" -lt 1 ]; then
        print_color $RED "❌ Invalid binary test size: $TEST_SIZE_BINARY. Must be a positive integer"
        exit 1
    fi

    if ! [[ "$TEST_SIZE_COMPLEX" =~ ^[0-9]+$ ]] || [ "$TEST_SIZE_COMPLEX" -lt 1 ]; then
        print_color $RED "❌ Invalid complex test size: $TEST_SIZE_COMPLEX. Must be a positive integer"
        exit 1
    fi
}

# Function to start vLLM server if needed
ensure_server_running() {
    local model=$1
    local port=$2
    local server_name=$3

    if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
        print_color $GREEN "✅ $server_name server already running on port $port"
        return 0
    fi

    if [ "$AUTO_START_SERVER" = true ]; then
        print_color $BLUE "🚀 Starting $server_name server..."

        # Start server directly without killing existing ones
        case $model in
            qwen)
                if [ -f "./qwen.sh" ]; then
                    ./qwen.sh &
                else
                    print_color $RED "❌ qwen.sh not found"
                    exit 1
                fi
                ;;
            llama3)
                if [ -f "./llama.sh" ]; then
                    ./llama.sh &
                else
                    print_color $RED "❌ llama.sh not found"
                    exit 1
                fi
                ;;
        esac

        # Wait for server to start (models take 3-5 minutes to load)
        print_color $YELLOW "⏳ Waiting for $server_name server to start (model loading takes 3-5 minutes)..."
        local attempts=0
        local max_attempts=150  # 5 minutes at 2-second intervals
        local last_gpu_check=0

        while [ $attempts -lt $max_attempts ]; do
            if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
                print_color $GREEN "✅ $server_name server is ready!"
                return 0
            fi

            sleep 2
            attempts=$((attempts + 1))
        done
        print_color $RED "❌ $server_name server failed to start after $((max_attempts * 2)) seconds"
        print_color $YELLOW "   Try manually starting: ./${model}.sh"
        exit 1
    else
        print_color $RED "❌ $server_name server not running on port $port"
        print_color $YELLOW "   Start with: ./${model}.sh"
        exit 1
    fi
}

# Function to run all complex dataset evaluations
run_all_complex_evaluations() {
    local architecture=$1

    # Only 5 core complex query types (3,022 total queries with comprehensive datasets)
    local query_types=(
        "organ_specific:945:Organ-Specific Queries"
        "severity_filtered:877:Severity-Filtered Queries"
        "drug_comparison:147:Drug Comparison Queries"
        "reverse_lookup:600:Reverse Lookup Queries"
        "combination:453:Combination Queries"
    )

    print_color $PURPLE "   Running evaluation on 5 complex query types (3,022 total queries)..."
    echo ""

    cd experiments

    # Use batch processing if enabled
    if [ "$USE_BATCH_PROCESSING" = true ]; then
        print_color $CYAN "   🚀 Using batch processing with 4 GPUs (batch_size=$BATCH_SIZE)"

        # Extract base architecture name and model
        local base_arch="${architecture%_*}"
        local model="${architecture##*_}"

        # Use batch evaluation script
        time python evaluate_complex_queries_batch.py \
            --architectures $base_arch \
            --models $model \
            --query_types organ_specific severity_filtered drug_comparison reverse_lookup combination \
            --batch_size $BATCH_SIZE \
            --limit $TEST_SIZE_COMPLEX

    # Use enhanced evaluation for new architectures or when flag is set
    elif [[ "$architecture" == enhanced_graphrag_* ]] || [[ "$architecture" == advanced_rag_format_b_* ]] || [ "$USE_ENHANCED_EVAL" = true ]; then
        print_color $YELLOW "   Using enhanced evaluation with Chain-of-Thought and semantic metrics"

        # Extract base architecture name and model
        local base_arch="${architecture%_*}"
        local model="${architecture##*_}"

        # Run enhanced evaluation for all 5 query types at once
        time python evaluate_enhanced_complex_queries.py \
            --architectures $base_arch \
            --models $model \
            --query_types organ_specific severity_filtered drug_comparison reverse_lookup combination \
            --queries_per_type all
    else
        # Run standard evaluation for each query type
        for query_info in "${query_types[@]}"; do
            IFS=':' read -r query_type query_count description <<< "$query_info"
            print_color $YELLOW "   📂 $description ($query_count queries)"

            time python evaluate_complex_queries.py \
                --architecture $architecture \
                --query_type $query_type \
                --test_size $query_count
            echo ""
        done
    fi

    cd ..
}

# Function to get strategy architecture names
get_architectures() {
    local strategy=$1
    local model=$2
    local query_type=$3

    case $strategy in
        pure)
            echo "pure_llm_$model"
            ;;
        format_a)
            echo "format_a_$model"
            ;;
        format_b)
            echo "format_b_$model"
            ;;
        graphrag)
            echo "graphrag_$model"
            ;;
        enhanced_b)
            echo "enhanced_format_b_$model"
            ;;
        enhanced_graphrag)
            echo "enhanced_graphrag_$model"
            ;;
        advanced_rag_b)
            echo "advanced_rag_format_b_$model"
            ;;
        all)
            # For binary queries, only use basic architectures
            if [[ "$query_type" == "binary" ]]; then
                echo "pure_llm_$model format_a_$model format_b_$model graphrag_$model"
            else
                # For complex queries, only use enhanced architectures
                echo "enhanced_format_b_$model enhanced_graphrag_$model advanced_rag_format_b_$model"
            fi
            ;;
    esac
}

# Function to run evaluation for a specific architecture
run_evaluation() {
    local architecture=$1
    local arch_display=$2
    local query_type=$3

    print_color $PURPLE "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_color $PURPLE "                              $arch_display"
    print_color $PURPLE "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    case $query_type in
        binary|both)
            print_color $CYAN "🎯 Binary Queries (Ground Truth Validation)"
            print_color $CYAN "   Testing against evaluation_dataset.csv with exact Cypher queries"
            cd experiments
            time python evaluate_vllm.py --test_size $TEST_SIZE_BINARY --architecture $architecture
            cd ..
            echo ""
            ;;
    esac

    case $query_type in
        complex|both)
            print_color $CYAN "🧠 Complex Queries (Multi-step Reasoning)"

            if [ "$RUN_ALL_COMPLEX" = true ]; then
                print_color $CYAN "   Testing ALL complex query datasets"
                run_all_complex_evaluations "$architecture"
            else
                print_color $CYAN "   Testing dataset: $COMPLEX_DATASET"
                cd experiments

                # Use enhanced evaluation for new architectures or when flag is set
                if [[ "$architecture" == enhanced_graphrag_* ]] || [[ "$architecture" == advanced_rag_format_b_* ]] || [ "$USE_ENHANCED_EVAL" = true ]; then
                    print_color $YELLOW "   Using enhanced evaluation with Chain-of-Thought and semantic metrics"
                    # Extract base architecture name and model
                    local base_arch="${architecture%_*}"
                    local model="${architecture##*_}"

                    # If using specific dataset
                    if [ "$COMPLEX_DATASET" != "complex_query_dataset.csv" ]; then
                        time python evaluate_complex_queries.py --test_size $TEST_SIZE_COMPLEX --architecture $architecture \
                            --dataset ../data/processed/$COMPLEX_DATASET
                    else
                        time python evaluate_enhanced_complex_queries.py --architectures $base_arch --models $model \
                            --query_types organ_specific drug_comparison severity_filtered \
                            --queries_per_type $TEST_SIZE_COMPLEX
                    fi
                else
                    if [ "$COMPLEX_DATASET" != "complex_query_dataset.csv" ]; then
                        time python evaluate_complex_queries.py --test_size $TEST_SIZE_COMPLEX --architecture $architecture \
                            --dataset ../data/processed/$COMPLEX_DATASET
                    else
                        time python evaluate_complex_queries.py --test_size $TEST_SIZE_COMPLEX --architecture $architecture
                    fi
                fi
                cd ..
                echo ""
            fi
            ;;
    esac
}

# Main execution function
main() {
    echo "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        ENHANCED DRUGRAG EVALUATION                               ║"
    echo "║                     Flexible Parameter-Based Testing                            ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Parse and validate arguments
    parse_arguments "$@"
    validate_arguments

    # Show configuration
    print_color $BLUE "🚀 EVALUATION CONFIGURATION"
    print_color $BLUE "   LLM Model(s): $LLM"
    print_color $BLUE "   Query Type: $QUERY_TYPE"
    print_color $BLUE "   Strategy: $STRATEGY"
    print_color $BLUE "   Binary Test Size: $TEST_SIZE_BINARY"
    print_color $BLUE "   Complex Test Size: $TEST_SIZE_COMPLEX"
    if [ "$RUN_ALL_COMPLEX" = true ]; then
        print_color $BLUE "   Complex Query Types: ALL (5 types, 2,905 queries)"
    else
        print_color $BLUE "   Complex Dataset: $COMPLEX_DATASET"
    fi
    print_color $BLUE "   Auto-start servers: $AUTO_START_SERVER"
    print_color $BLUE "   Enhanced evaluation: $USE_ENHANCED_EVAL"
    echo ""

    # Determine which models to test
    local models_to_test=""
    case $LLM in
        qwen)
            models_to_test="qwen"
            ;;
        llama3)
            models_to_test="llama3"
            ;;
        both)
            models_to_test="qwen llama3"
            ;;
    esac

    # Ensure required servers are running
    for model in $models_to_test; do
        case $model in
            qwen)
                ensure_server_running "qwen" "8002" "Qwen"
                ;;
            llama3)
                ensure_server_running "llama3" "8003" "LLAMA3"
                ;;
        esac
    done

    echo ""
    print_color $GREEN "═══════════════════════════════════════════════════════════════════════════════════"
    print_color $GREEN "                              STARTING EVALUATION"
    print_color $GREEN "═══════════════════════════════════════════════════════════════════════════════════"
    echo ""

    # Run evaluations
    for model in $models_to_test; do
        local architectures=$(get_architectures "$STRATEGY" "$model" "$QUERY_TYPE")

        for arch in $architectures; do
            local arch_display=""
            case $arch in
                pure_llm_*)
                    arch_display="Pure LLM ($(echo $arch | cut -d'_' -f3 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                format_a_*)
                    arch_display="RAG Format A ($(echo $arch | cut -d'_' -f3 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                format_b_*)
                    arch_display="RAG Format B ($(echo $arch | cut -d'_' -f3 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                graphrag_*)
                    arch_display="GraphRAG ($(echo $arch | cut -d'_' -f2 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                enhanced_format_b_*)
                    arch_display="Enhanced Format B ($(echo $arch | cut -d'_' -f4 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                enhanced_graphrag_*)
                    arch_display="Enhanced GraphRAG with CoT ($(echo $arch | cut -d'_' -f3 | tr '[:lower:]' '[:upper:]'))"
                    ;;
                advanced_rag_format_b_*)
                    arch_display="Advanced RAG Format B ($(echo $arch | cut -d'_' -f5 | tr '[:lower:]' '[:upper:]'))"
                    ;;
            esac

            run_evaluation "$arch" "$arch_display" "$QUERY_TYPE"
        done
    done

    echo ""
    print_color $GREEN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_color $GREEN "                             EVALUATION COMPLETE"
    print_color $GREEN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    print_color $CYAN "📊 RESULTS SUMMARY"
    print_color $CYAN "   LLM Model(s) tested: $LLM"
    print_color $CYAN "   Query types evaluated: $QUERY_TYPE"
    print_color $CYAN "   Architecture strategy: $STRATEGY"
    print_color $CYAN "   Result files generated in experiments/ directory"
    echo ""

    print_color $YELLOW "💡 Next Steps:"
    print_color $YELLOW "   • Check experiments/ directory for result files"
    print_color $YELLOW "   • Analyze JSON output files for metrics"
    print_color $YELLOW "   • Compare performance across architectures"
    echo ""
}

# Run main function with all arguments
main "$@"