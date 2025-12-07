# Fine-Tuning Language Models on NVIDIA DGX Spark

Complete toolkit for fine-tuning open-source language models on the NVIDIA DGX Spark personal AI supercomputer.

## 🚀 Quick Start

```bash
# 1. Setup environment
chmod +x setup.sh && ./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run fine-tuning with a preset
./run_finetune.sh small
```

## 📋 Contents

```
finetune-guide/
├── scripts/
│   ├── finetune_dgx_spark.py   # Main fine-tuning script
│   └── prepare_dataset.py       # Dataset preparation utilities
├── requirements.txt              # Python dependencies
├── setup.sh                      # Environment setup script
├── run_finetune.sh              # Quick-start run script
└── README.md                     # This file
```

## 🎯 Supported Models

| Model | Parameters | HuggingFace Path |
|-------|------------|------------------|
| SmolLM | 135M-1.7B | `HuggingFaceTB/SmolLM-*` |
| Qwen 2.5 | 1.5B-7B | `Qwen/Qwen2.5-*` |
| Llama 3.2 | 1B-3B | `meta-llama/Llama-3.2-*` |
| Llama 3.1 | 8B | `meta-llama/Llama-3.1-8B` |
| Mistral | 7B-12B | `mistralai/Mistral-*` |
| Phi-3 | 3.8B-8B | `microsoft/Phi-3-*` |
| Gemma 2 | 2B-9B | `google/gemma-2-*` |

## 🔧 Fine-Tuning Methods

### Full Fine-Tuning
Best for small models (<3B). Updates all parameters.
```bash
python scripts/finetune_dgx_spark.py --model smollm-360m --method full --dataset data.json
```

### LoRA (Recommended)
Best for medium models (3-13B). Trains only adapter layers.
```bash
python scripts/finetune_dgx_spark.py --model qwen2.5-3b --method lora --dataset data.json
```

### QLoRA
Best for large models (13B+). Uses 4-bit quantization.
```bash
python scripts/finetune_dgx_spark.py --model llama-3.1-8b --method qlora --dataset data.json
```

## 📊 Dataset Format

### Alpaca Format (Recommended)
```json
[
  {
    "instruction": "Explain machine learning",
    "input": "",
    "output": "Machine learning is..."
  }
]
```

### Create Sample Data
```bash
python scripts/prepare_dataset.py --create-sample
```

## ⚡ DGX Spark Advantages

- **128GB Unified Memory**: Train 70B models locally with QLoRA
- **900 GB/s NVLink-C2C**: Fast CPU-GPU communication
- **Local Execution**: Complete privacy, no cloud costs
- **1 PFLOPS Performance**: Enterprise-grade AI capabilities

## 📖 Full Documentation

See `DGX_Spark_Fine_Tuning_Guide.docx` for comprehensive instructions.

## 🔄 Run Presets

```bash
./run_finetune.sh quick   # SmolLM 360M, full fine-tuning
./run_finetune.sh small   # Qwen 3B, LoRA
./run_finetune.sh medium  # Llama 8B, LoRA  
./run_finetune.sh large   # Llama 8B, QLoRA
```

## 📈 Monitoring Training

```bash
# TensorBoard
tensorboard --logdir output/logs

# Check GPU usage
nvidia-smi -l 1
```

## 🧪 Test Your Model

```bash
python scripts/finetune_dgx_spark.py \
    --inference \
    --model-path output/merged_model \
    --prompt "Your test prompt here"
```

## 📝 License

MIT License - Use freely for any purpose.
