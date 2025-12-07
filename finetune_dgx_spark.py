#!/usr/bin/env python3
"""
DGX Spark Fine-Tuning Script
=============================
Optimized for NVIDIA DGX Spark's 128GB unified memory architecture.

Supports multiple fine-tuning strategies:
- Full fine-tuning (for smaller models, leverages full memory)
- LoRA (Low-Rank Adaptation - memory efficient)
- QLoRA (Quantized LoRA - maximum memory efficiency)

Author: DGX Spark AI Development Series
License: MIT
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DGX SPARK CONFIGURATION
# =============================================================================

class DGXSparkConfig:
    """
    Hardware-optimized configuration for NVIDIA DGX Spark.
    
    The DGX Spark features:
    - NVIDIA Grace Blackwell architecture
    - 128GB unified memory (CPU+GPU shared)
    - NVLink-C2C interconnect (900 GB/s bandwidth)
    - Up to 1 PFLOPS AI performance (FP4)
    """
    
    # Memory configuration (128GB unified)
    TOTAL_MEMORY_GB = 128
    RECOMMENDED_MODEL_MEMORY_GB = 80  # Leave headroom for activations
    
    # Optimal batch sizes for different model sizes
    BATCH_SIZE_RECOMMENDATIONS = {
        "1B": {"full": 16, "lora": 32, "qlora": 64},
        "3B": {"full": 8, "lora": 16, "qlora": 32},
        "7B": {"full": 4, "lora": 8, "qlora": 16},
        "13B": {"full": 2, "lora": 4, "qlora": 8},
        "70B": {"full": 1, "lora": 2, "qlora": 4},
    }
    
    # Gradient accumulation to simulate larger batches
    GRADIENT_ACCUMULATION = {
        "1B": 1, "3B": 2, "7B": 4, "13B": 8, "70B": 16
    }
    
    @classmethod
    def get_optimal_settings(cls, model_size: str, method: str) -> Dict[str, int]:
        """Get optimal training settings for model size and method."""
        size_key = cls._get_size_key(model_size)
        return {
            "batch_size": cls.BATCH_SIZE_RECOMMENDATIONS.get(size_key, {}).get(method, 4),
            "gradient_accumulation": cls.GRADIENT_ACCUMULATION.get(size_key, 4)
        }
    
    @staticmethod
    def _get_size_key(model_size: str) -> str:
        """Map model size string to configuration key."""
        size_lower = model_size.lower()
        if "70b" in size_lower:
            return "70B"
        elif "13b" in size_lower:
            return "13B"
        elif "7b" in size_lower or "8b" in size_lower:
            return "7B"
        elif "3b" in size_lower:
            return "3B"
        else:
            return "1B"


# =============================================================================
# MODEL REGISTRY
# =============================================================================

SUPPORTED_MODELS = {
    # Llama family
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    
    # Mistral family
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    
    # Qwen family
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    
    # Phi family (Microsoft)
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    
    # Gemma family (Google)
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    
    # SmolLM (Hugging Face) - great for experimentation
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B",
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

class DatasetPreparator:
    """Prepare datasets for fine-tuning."""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_instruction_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        instruction_col: str = "instruction",
        input_col: str = "input",
        output_col: str = "output",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """
        Prepare an instruction-following dataset.
        
        Supports common formats:
        - Alpaca-style: instruction, input, output
        - ShareGPT-style: conversations
        - Custom: specify column names
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_name, split="train")
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        # Format into instruction template
        def format_instruction(example):
            instruction = example.get(instruction_col, "")
            inp = example.get(input_col, "")
            output = example.get(output_col, "")
            
            if inp:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            return {"text": text}
        
        dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
        return self._tokenize_dataset(dataset)
    
    def prepare_conversational_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        conversations_col: str = "conversations",
        num_samples: Optional[int] = None
    ) -> Dataset:
        """Prepare a conversational/chat dataset."""
        logger.info(f"Loading conversational dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        def format_conversation(example):
            conversations = example.get(conversations_col, [])
            text_parts = []
            
            for msg in conversations:
                role = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                
                if role in ["user", "human"]:
                    text_parts.append(f"### User:\n{content}")
                else:
                    text_parts.append(f"### Assistant:\n{content}")
            
            return {"text": "\n\n".join(text_parts)}
        
        dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        return self._tokenize_dataset(dataset)
    
    def prepare_custom_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """Prepare a custom dataset from a list of text examples."""
        dataset = Dataset.from_list([{"text": item["text"]} for item in data])
        return self._tokenize_dataset(dataset)
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )


# =============================================================================
# FINE-TUNING STRATEGIES
# =============================================================================

class FineTuner:
    """
    Fine-tuning orchestrator for DGX Spark.
    
    Supports:
    - Full fine-tuning (best quality, highest memory)
    - LoRA (good quality, moderate memory)
    - QLoRA (good quality, lowest memory)
    """
    
    def __init__(
        self,
        model_name: str,
        method: str = "lora",
        output_dir: str = "./output",
        use_flash_attention: bool = True
    ):
        self.model_name = self._resolve_model_name(model_name)
        self.method = method.lower()
        self.output_dir = Path(output_dir)
        self.use_flash_attention = use_flash_attention
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Initializing FineTuner")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _resolve_model_name(self, name: str) -> str:
        """Resolve model shorthand to full HuggingFace path."""
        return SUPPORTED_MODELS.get(name.lower(), name)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model with method: {self.method}")
        
        # Common model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        # Enable Flash Attention 2 if available
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e}")
        
        if self.method == "full":
            self._load_full_precision_model(model_kwargs)
        elif self.method == "lora":
            self._load_lora_model(model_kwargs)
        elif self.method == "qlora":
            self._load_qlora_model(model_kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Log memory usage
        self._log_memory_usage()
    
    def _load_full_precision_model(self, model_kwargs: Dict):
        """Load model for full fine-tuning."""
        model_kwargs["torch_dtype"] = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        logger.info("Loaded model for full fine-tuning with gradient checkpointing")
    
    def _load_lora_model(self, model_kwargs: Dict):
        """Load model with LoRA adapters."""
        model_kwargs["torch_dtype"] = torch.bfloat16
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # LoRA configuration optimized for instruction-following
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                          # LoRA rank
            lora_alpha=32,                 # Alpha scaling factor
            lora_dropout=0.05,             # Dropout for regularization
            target_modules=[               # Target attention layers
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("Loaded model with LoRA adapters")
    
    def _load_qlora_model(self, model_kwargs: Dict):
        """Load quantized model with LoRA adapters (QLoRA)."""
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,      # Nested quantization
        )
        
        model_kwargs["quantization_config"] = bnb_config
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare for k-bit training
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True
        )
        
        # QLoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,                          # Higher rank for QLoRA
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("Loaded quantized model with QLoRA adapters")
    
    def _log_memory_usage(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        warmup_ratio: float = 0.03,
        save_steps: int = 100,
        logging_steps: int = 10,
        max_grad_norm: float = 1.0,
    ):
        """
        Execute fine-tuning training.
        
        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Per-device batch size (auto-detected if None)
            gradient_accumulation_steps: Gradient accumulation (auto-detected if None)
            warmup_ratio: Warmup ratio for learning rate scheduler
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        # Auto-detect optimal settings if not provided
        if batch_size is None or gradient_accumulation_steps is None:
            settings = DGXSparkConfig.get_optimal_settings(
                self.model_name, self.method
            )
            batch_size = batch_size or settings["batch_size"]
            gradient_accumulation_steps = gradient_accumulation_steps or settings["gradient_accumulation"]
        
        logger.info(f"Training Configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective Batch Size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Learning Rate: {learning_rate}")
        
        # Training arguments optimized for DGX Spark
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            bf16=True,                          # Use BF16 on DGX Spark
            tf32=True,                          # Enable TF32 for faster matmuls
            gradient_checkpointing=True,
            max_grad_norm=max_grad_norm,
            report_to=["tensorboard"],
            logging_dir=str(self.output_dir / "logs"),
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            optim="adamw_torch_fused",          # Fused optimizer for speed
        )
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[MemoryMonitorCallback()],
        )
        
        # Start training
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        self.save_model()
        
        # Log training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """Save the fine-tuned model."""
        save_path = Path(path) if path else self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        if self.method in ["lora", "qlora"]:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Also save merged model for inference
            merged_path = save_path.parent / "merged_model"
            logger.info(f"Merging and saving full model to {merged_path}")
            
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            self.tokenizer.save_pretrained(merged_path)
        else:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        # Save training config
        config = {
            "model_name": self.model_name,
            "method": self.method,
            "timestamp": datetime.now().isoformat(),
        }
        with open(save_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Model saved successfully")


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor GPU memory during training."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"Step {state.global_step}: GPU Memory = {allocated:.2f} GB")


# =============================================================================
# INFERENCE
# =============================================================================

class InferenceEngine:
    """Run inference with fine-tuned models."""
    
    def __init__(self, model_path: str, use_flash_attention: bool = True):
        self.model_path = Path(model_path)
        
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        self.model.eval()
        logger.info("Model loaded for inference")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def chat(self, instruction: str, context: str = "") -> str:
        """Generate a response to an instruction."""
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return self.generate(prompt)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models on NVIDIA DGX Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune Llama 3.2 1B with LoRA
  python finetune_dgx_spark.py --model llama-3.2-1b --method lora --dataset alpaca

  # Fine-tune Qwen 2.5 3B with QLoRA
  python finetune_dgx_spark.py --model qwen2.5-3b --method qlora --dataset tatsu-lab/alpaca

  # Full fine-tuning of SmolLM (small model)
  python finetune_dgx_spark.py --model smollm-360m --method full --dataset your_data.json

Supported models:
  llama-3.2-1b, llama-3.2-3b, llama-3.1-8b
  mistral-7b, mistral-nemo-12b
  qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b
  phi-3-mini, phi-3-small
  gemma-2-2b, gemma-2-9b
  smollm-135m, smollm-360m, smollm-1.7b
        """
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (shorthand or HuggingFace path)")
    parser.add_argument("--method", type=str, default="lora",
                       choices=["full", "lora", "qlora"],
                       help="Fine-tuning method")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (HuggingFace or local JSON file)")
    parser.add_argument("--dataset-format", type=str, default="instruction",
                       choices=["instruction", "conversation", "text"],
                       help="Dataset format type")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of training samples")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (auto-detected if not set)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory for model and logs")
    
    # Hardware options
    parser.add_argument("--no-flash-attention", action="store_true",
                       help="Disable Flash Attention 2")
    
    # Inference mode
    parser.add_argument("--inference", action="store_true",
                       help="Run in inference mode (requires --model-path)")
    parser.add_argument("--model-path", type=str,
                       help="Path to fine-tuned model for inference")
    parser.add_argument("--prompt", type=str,
                       help="Prompt for inference mode")
    
    args = parser.parse_args()
    
    # Inference mode
    if args.inference:
        if not args.model_path:
            parser.error("--model-path required for inference mode")
        
        engine = InferenceEngine(
            args.model_path,
            use_flash_attention=not args.no_flash_attention
        )
        
        if args.prompt:
            response = engine.chat(args.prompt)
            print(f"\nResponse:\n{response}")
        else:
            # Interactive mode
            print("\nInteractive mode (type 'quit' to exit)")
            while True:
                prompt = input("\nYou: ").strip()
                if prompt.lower() == 'quit':
                    break
                response = engine.chat(prompt)
                print(f"\nAssistant: {response}")
        
        return
    
    # Training mode
    print("=" * 60)
    print("DGX Spark Fine-Tuning")
    print("=" * 60)
    
    # Initialize fine-tuner
    finetuner = FineTuner(
        model_name=args.model,
        method=args.method,
        output_dir=args.output_dir,
        use_flash_attention=not args.no_flash_attention
    )
    
    # Load model
    finetuner.load_model_and_tokenizer()
    
    # Prepare dataset
    preparator = DatasetPreparator(
        finetuner.tokenizer,
        max_length=args.max_length
    )
    
    if args.dataset_format == "instruction":
        train_dataset = preparator.prepare_instruction_dataset(
            args.dataset,
            num_samples=args.max_samples
        )
    elif args.dataset_format == "conversation":
        train_dataset = preparator.prepare_conversational_dataset(
            args.dataset,
            num_samples=args.max_samples
        )
    else:
        raise ValueError(f"Unsupported dataset format: {args.dataset_format}")
    
    # Train
    metrics = finetuner.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
