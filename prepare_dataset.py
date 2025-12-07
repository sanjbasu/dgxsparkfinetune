#!/usr/bin/env python3
"""
Dataset Preparation Utilities for DGX Spark Fine-Tuning
========================================================

This module provides utilities for preparing custom datasets
for fine-tuning language models on the NVIDIA DGX Spark.

Supports:
- JSON/JSONL files
- CSV files
- Text files
- HuggingFace datasets
- Custom data pipelines
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

from datasets import Dataset, DatasetDict, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset preparation."""
    max_length: int = 2048
    train_split: float = 0.9
    seed: int = 42
    shuffle: bool = True


@dataclass  
class InstructionTemplate:
    """Template for formatting instruction data."""
    
    # Default Alpaca-style template
    template: str = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    template_no_input: str = """### Instruction:
{instruction}

### Response:
{output}"""
    
    def format(self, instruction: str, output: str, input_text: str = "") -> str:
        """Format a single example."""
        if input_text:
            return self.template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        return self.template_no_input.format(
            instruction=instruction,
            output=output
        )


@dataclass
class ChatTemplate:
    """Template for formatting chat/conversation data."""
    
    system_template: str = "### System:\n{content}\n\n"
    user_template: str = "### User:\n{content}\n\n"
    assistant_template: str = "### Assistant:\n{content}"
    
    def format(self, messages: List[Dict[str, str]]) -> str:
        """Format a conversation into a single string."""
        formatted_parts = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(self.system_template.format(content=content))
            elif role in ["user", "human"]:
                formatted_parts.append(self.user_template.format(content=content))
            elif role in ["assistant", "gpt", "bot"]:
                formatted_parts.append(self.assistant_template.format(content=content))
        
        return "".join(formatted_parts)


class DatasetBuilder:
    """
    Build datasets from various sources.
    
    Examples:
        # From JSON file
        builder = DatasetBuilder()
        dataset = builder.from_json("data.json", format="alpaca")
        
        # From CSV
        dataset = builder.from_csv("data.csv", 
            instruction_col="question",
            output_col="answer"
        )
        
        # From custom data
        data = [
            {"instruction": "Say hello", "output": "Hello!"},
            {"instruction": "Count to 3", "output": "1, 2, 3"}
        ]
        dataset = builder.from_list(data)
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.instruction_template = InstructionTemplate()
        self.chat_template = ChatTemplate()
    
    def from_json(
        self,
        path: str,
        format: str = "alpaca",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        conversations_key: str = "conversations",
    ) -> Dataset:
        """
        Load dataset from JSON or JSONL file.
        
        Args:
            path: Path to JSON/JSONL file
            format: Data format - "alpaca", "sharegpt", or "text"
            instruction_key: Key for instruction field
            input_key: Key for input field
            output_key: Key for output field
            conversations_key: Key for conversations (ShareGPT format)
        
        Returns:
            Formatted Dataset
        """
        path = Path(path)
        logger.info(f"Loading data from {path}")
        
        # Load JSON/JSONL
        if path.suffix == ".jsonl":
            data = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        logger.info(f"Loaded {len(data)} examples")
        
        # Format based on type
        if format == "alpaca":
            return self._format_alpaca(data, instruction_key, input_key, output_key)
        elif format == "sharegpt":
            return self._format_sharegpt(data, conversations_key)
        elif format == "text":
            return self._format_text(data)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def from_csv(
        self,
        path: str,
        instruction_col: str = "instruction",
        input_col: Optional[str] = None,
        output_col: str = "output",
    ) -> Dataset:
        """Load dataset from CSV file."""
        path = Path(path)
        logger.info(f"Loading CSV from {path}")
        
        data = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                example = {
                    "instruction": row[instruction_col],
                    "input": row.get(input_col, "") if input_col else "",
                    "output": row[output_col],
                }
                data.append(example)
        
        logger.info(f"Loaded {len(data)} examples from CSV")
        return self._format_alpaca(data, "instruction", "input", "output")
    
    def from_list(
        self,
        data: List[Dict[str, str]],
        format: str = "alpaca"
    ) -> Dataset:
        """
        Create dataset from list of dictionaries.
        
        For alpaca format, each dict should have:
            - instruction: str
            - output: str
            - input: str (optional)
        
        For text format, each dict should have:
            - text: str
        """
        logger.info(f"Creating dataset from {len(data)} examples")
        
        if format == "alpaca":
            return self._format_alpaca(data, "instruction", "input", "output")
        elif format == "text":
            return self._format_text(data)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        format: str = "alpaca",
        **kwargs
    ) -> Dataset:
        """Load dataset from HuggingFace Hub."""
        logger.info(f"Loading {dataset_name} from HuggingFace")
        
        dataset = load_dataset(dataset_name, split=split)
        data = [dict(row) for row in dataset]
        
        return self.from_list(data, format=format)
    
    def _format_alpaca(
        self,
        data: List[Dict],
        instruction_key: str,
        input_key: str,
        output_key: str
    ) -> Dataset:
        """Format data as Alpaca-style instruction-following."""
        formatted = []
        
        for example in data:
            instruction = example.get(instruction_key, "")
            input_text = example.get(input_key, "")
            output = example.get(output_key, "")
            
            text = self.instruction_template.format(instruction, output, input_text)
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def _format_sharegpt(
        self,
        data: List[Dict],
        conversations_key: str
    ) -> Dataset:
        """Format ShareGPT-style conversation data."""
        formatted = []
        
        for example in data:
            conversations = example.get(conversations_key, [])
            text = self.chat_template.format(conversations)
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def _format_text(self, data: List[Dict]) -> Dataset:
        """Format raw text data."""
        formatted = [{"text": item.get("text", str(item))} for item in data]
        return Dataset.from_list(formatted)
    
    def train_test_split(
        self,
        dataset: Dataset,
        test_size: float = 0.1
    ) -> DatasetDict:
        """Split dataset into train and test sets."""
        split = dataset.train_test_split(
            test_size=test_size,
            seed=self.config.seed,
            shuffle=self.config.shuffle
        )
        return split


def create_sample_dataset(output_path: str = "sample_data.json"):
    """Create a sample dataset for testing."""
    
    sample_data = [
        {
            "instruction": "Explain what machine learning is in simple terms.",
            "input": "",
            "output": "Machine learning is a type of artificial intelligence where computers learn patterns from data instead of being explicitly programmed. Think of it like teaching a child to recognize dogs - you show them many pictures of dogs, and eventually they learn to identify dogs on their own. Similarly, machine learning systems analyze lots of examples to learn how to make predictions or decisions."
        },
        {
            "instruction": "Write a Python function that calculates the factorial of a number.",
            "input": "",
            "output": """def factorial(n):
    \"\"\"Calculate the factorial of a non-negative integer.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Example usage:
# factorial(5) returns 120"""
        },
        {
            "instruction": "Summarize the following text in one sentence.",
            "input": "The NVIDIA DGX Spark is a personal AI supercomputer featuring the Grace Blackwell architecture. It includes 128GB of unified memory shared between the CPU and GPU, connected via NVLink-C2C with 900 GB/s bandwidth. This enables running large language models locally with performance that rivals cloud infrastructure.",
            "output": "The NVIDIA DGX Spark is a personal AI supercomputer with 128GB unified memory and Grace Blackwell architecture that enables local execution of large language models with cloud-competitive performance."
        },
        {
            "instruction": "What are the benefits of fine-tuning a language model?",
            "input": "",
            "output": "Fine-tuning offers several key benefits: First, it customizes a pre-trained model for specific tasks or domains, significantly improving performance on targeted use cases. Second, it's more efficient than training from scratch, requiring less data and compute. Third, it allows you to incorporate proprietary knowledge while preserving the model's general capabilities. Finally, fine-tuned models often respond in more consistent, predictable ways aligned with your requirements."
        },
        {
            "instruction": "Convert this temperature from Celsius to Fahrenheit.",
            "input": "25°C",
            "output": "25°C is equal to 77°F. The formula is: F = (C × 9/5) + 32, so (25 × 9/5) + 32 = 45 + 32 = 77°F."
        },
        {
            "instruction": "Explain the difference between LoRA and full fine-tuning.",
            "input": "",
            "output": "Full fine-tuning updates all model parameters during training, offering maximum customization but requiring significant memory and compute. LoRA (Low-Rank Adaptation) only trains small adapter matrices inserted into the model, typically less than 1% of parameters. This dramatically reduces memory requirements and training time while achieving comparable quality for most tasks. LoRA adapters can also be easily swapped or combined, making them ideal for multi-task scenarios."
        },
        {
            "instruction": "Write a haiku about artificial intelligence.",
            "input": "",
            "output": "Silicon minds dream,\nPatterns emerge from the void—\nData becomes thought."
        },
        {
            "instruction": "Debug this code and explain the issue.",
            "input": "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
            "output": "The code will raise a ZeroDivisionError because dividing by zero is undefined. Here's the fixed version:\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError(\"Cannot divide by zero\")\n    return a / b\n\n# Or return a default value:\ndef safe_divide(a, b, default=None):\n    return a / b if b != 0 else default"
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample dataset at {output_path}")
    return output_path


def create_domain_dataset(
    domain: str,
    num_examples: int = 100,
    output_path: Optional[str] = None
) -> str:
    """
    Create a domain-specific dataset template.
    
    This creates a template with placeholder examples that you can
    fill in with your domain-specific data.
    
    Args:
        domain: Domain name (e.g., "medical", "legal", "coding")
        num_examples: Number of placeholder examples to generate
        output_path: Output file path
    
    Returns:
        Path to created file
    """
    if output_path is None:
        output_path = f"{domain}_dataset_template.json"
    
    template_examples = []
    for i in range(num_examples):
        template_examples.append({
            "instruction": f"[{domain.upper()} INSTRUCTION {i+1}] - Replace with your instruction",
            "input": f"[Optional input/context for example {i+1}]",
            "output": f"[Expected response for example {i+1}]"
        })
    
    with open(output_path, 'w') as f:
        json.dump(template_examples, f, indent=2)
    
    logger.info(f"Created {domain} dataset template at {output_path}")
    return output_path


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create a sample dataset")
    parser.add_argument("--create-template", type=str, metavar="DOMAIN",
                       help="Create a domain-specific template")
    parser.add_argument("--num-examples", type=int, default=100,
                       help="Number of examples for template")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--convert", type=str, metavar="INPUT_FILE",
                       help="Convert file to training format")
    parser.add_argument("--format", type=str, default="alpaca",
                       choices=["alpaca", "sharegpt", "text"],
                       help="Input format for conversion")
    
    args = parser.parse_args()
    
    if args.create_sample:
        path = create_sample_dataset(args.output or "sample_data.json")
        print(f"Created sample dataset: {path}")
    
    elif args.create_template:
        path = create_domain_dataset(
            args.create_template,
            args.num_examples,
            args.output
        )
        print(f"Created template: {path}")
    
    elif args.convert:
        builder = DatasetBuilder()
        dataset = builder.from_json(args.convert, format=args.format)
        
        output = args.output or args.convert.replace(".json", "_processed.json")
        
        # Save processed dataset
        data_list = [{"text": row["text"]} for row in dataset]
        with open(output, 'w') as f:
            json.dump(data_list, f, indent=2)
        
        print(f"Converted {len(data_list)} examples to {output}")
    
    else:
        parser.print_help()
