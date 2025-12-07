#!/bin/bash
# =============================================================================
# DGX Spark Fine-Tuning Run Script
# =============================================================================
# Quick-start script for common fine-tuning scenarios
#
# Usage: ./run_finetune.sh <preset> [options]
#
# Presets:
#   quick     - SmolLM 360M, full fine-tuning, fast experimentation
#   small     - Qwen 3B with LoRA, balanced speed/quality
#   medium    - Llama 8B with LoRA, high quality
#   large     - Llama 8B with QLoRA, memory efficient
#   custom    - Use your own settings via environment variables
# =============================================================================

set -e

# Activate virtual environment if not active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
fi

# Default values
PRESET="${1:-quick}"
DATASET="${DATASET:-data/sample_data.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
EPOCHS="${EPOCHS:-3}"

# Parse preset
case "$PRESET" in
    quick)
        echo "=== Quick Experimentation Preset ==="
        echo "Model: SmolLM 360M | Method: Full | Epochs: $EPOCHS"
        MODEL="smollm-360m"
        METHOD="full"
        BATCH_SIZE=32
        ;;
    
    small)
        echo "=== Small Model Preset ==="
        echo "Model: Qwen 2.5 3B | Method: LoRA | Epochs: $EPOCHS"
        MODEL="qwen2.5-3b"
        METHOD="lora"
        BATCH_SIZE=16
        ;;
    
    medium)
        echo "=== Medium Model Preset ==="
        echo "Model: Llama 3.1 8B | Method: LoRA | Epochs: $EPOCHS"
        MODEL="llama-3.1-8b"
        METHOD="lora"
        BATCH_SIZE=8
        ;;
    
    large)
        echo "=== Large Model Preset (Memory Efficient) ==="
        echo "Model: Llama 3.1 8B | Method: QLoRA | Epochs: $EPOCHS"
        MODEL="llama-3.1-8b"
        METHOD="qlora"
        BATCH_SIZE=16
        ;;
    
    custom)
        echo "=== Custom Configuration ==="
        MODEL="${MODEL:-qwen2.5-3b}"
        METHOD="${METHOD:-lora}"
        BATCH_SIZE="${BATCH_SIZE:-8}"
        echo "Model: $MODEL | Method: $METHOD | Epochs: $EPOCHS"
        ;;
    
    *)
        echo "Unknown preset: $PRESET"
        echo ""
        echo "Available presets:"
        echo "  quick   - SmolLM 360M, full fine-tuning"
        echo "  small   - Qwen 3B with LoRA"
        echo "  medium  - Llama 8B with LoRA"
        echo "  large   - Llama 8B with QLoRA"
        echo "  custom  - Set MODEL, METHOD, BATCH_SIZE env vars"
        exit 1
        ;;
esac

echo ""
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR/${MODEL}_${METHOD}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python scripts/finetune_dgx_spark.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR/${MODEL}_${METHOD}"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $OUTPUT_DIR/${MODEL}_${METHOD}"
echo ""
echo "To test the model:"
echo "python scripts/finetune_dgx_spark.py --inference --model-path $OUTPUT_DIR/${MODEL}_${METHOD}/merged_model"
