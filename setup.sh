#!/bin/bash
# =============================================================================
# DGX Spark Fine-Tuning Setup Script
# =============================================================================
# This script sets up the environment for fine-tuning on NVIDIA DGX Spark
#
# Usage: ./setup.sh [--full]
#   --full: Install all optional dependencies including Flash Attention
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "DGX Spark Fine-Tuning Environment Setup"
echo "=============================================="
echo ""

# Check if running on DGX Spark (Grace Blackwell)
check_hardware() {
    echo "Checking hardware..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            echo "  GPU detected: $GPU_INFO"
        fi
    fi
    
    # Check total memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "  System memory: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -ge 120 ]; then
        echo "  ✓ Sufficient memory for large model fine-tuning"
    fi
    echo ""
}

# Create virtual environment
setup_venv() {
    echo "Setting up Python environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "  Created virtual environment"
    else
        echo "  Using existing virtual environment"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
    echo ""
}

# Install dependencies
install_deps() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Install Flash Attention if requested
    if [ "$1" == "--full" ]; then
        echo ""
        echo "Installing Flash Attention 2..."
        pip install flash-attn --no-build-isolation
    fi
    echo ""
}

# Create directory structure
create_dirs() {
    echo "Creating directory structure..."
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p output/models
    mkdir -p output/logs
    mkdir -p configs
    echo "  ✓ Directories created"
    echo ""
}

# Create sample configuration
create_sample_config() {
    echo "Creating sample configuration..."
    cat > configs/sample_config.yaml << 'EOF'
# DGX Spark Fine-Tuning Configuration
# ====================================

# Model Configuration
model:
  name: "qwen2.5-3b"           # Model shorthand or HuggingFace path
  method: "lora"               # full, lora, or qlora
  use_flash_attention: true

# LoRA Configuration (if method is lora or qlora)
lora:
  r: 16                        # LoRA rank
  alpha: 32                    # LoRA alpha
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# Dataset Configuration
dataset:
  path: "data/sample_data.json"
  format: "alpaca"             # alpaca, sharegpt, or text
  max_length: 2048
  max_samples: null            # null for all samples

# Training Configuration
training:
  epochs: 3
  batch_size: 8                # Per-device batch size
  gradient_accumulation: 4     # Effective batch = 8 * 4 = 32
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Optimization
  bf16: true
  tf32: true
  gradient_checkpointing: true
  
  # Logging
  logging_steps: 10
  save_steps: 100
  save_total_limit: 3

# Output Configuration
output:
  dir: "output/my_finetuned_model"
  tensorboard: true
EOF
    echo "  ✓ Created configs/sample_config.yaml"
    echo ""
}

# Create sample data
create_sample_data() {
    echo "Creating sample training data..."
    python3 scripts/prepare_dataset.py --create-sample --output data/sample_data.json
    echo ""
}

# Print usage instructions
print_usage() {
    echo "=============================================="
    echo "Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Quick Start:"
    echo ""
    echo "1. Activate the environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run fine-tuning with sample data:"
    echo "   python scripts/finetune_dgx_spark.py \\"
    echo "       --model qwen2.5-3b \\"
    echo "       --method lora \\"
    echo "       --dataset data/sample_data.json"
    echo ""
    echo "3. Or use the run script:"
    echo "   ./run_finetune.sh"
    echo ""
    echo "4. Monitor training with TensorBoard:"
    echo "   tensorboard --logdir output/logs"
    echo ""
    echo "For more options:"
    echo "   python scripts/finetune_dgx_spark.py --help"
    echo ""
}

# Main
check_hardware
setup_venv
install_deps "$1"
create_dirs
create_sample_config
create_sample_data
print_usage
