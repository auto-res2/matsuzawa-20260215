#!/usr/bin/env python3
"""
Dataset loading and preprocessing for arithmetic word problem benchmarks.
Supports GSM8K and SVAMP datasets.
"""

import re
from typing import List, Dict, Any
from pathlib import Path

from datasets import load_dataset as hf_load_dataset


def extract_answer_number(answer_text: str) -> float:
    """
    Extract numeric answer from answer text.
    Handles GSM8K format: '#### 42' or similar patterns.
    """
    # Try GSM8K format first
    if "####" in answer_text:
        match = re.search(r"####\s*([+-]?\d+\.?\d*)", answer_text)
        if match:
            return float(match.group(1))
    
    # Try general number extraction
    # Look for last number in text
    numbers = re.findall(r"([+-]?\d+\.?\d*)", answer_text)
    if numbers:
        return float(numbers[-1])
    
    raise ValueError(f"Could not extract answer from: {answer_text}")


def load_gsm8k(split: str = "test", num_samples: int = None, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """Load GSM8K dataset."""
    print(f"Loading GSM8K dataset (split={split})...")
    
    # Load from HuggingFace
    dataset = hf_load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    # Convert to list of dicts
    examples = []
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]
        
        try:
            answer_number = extract_answer_number(answer_text)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        
        examples.append({
            "question": question,
            "answer": answer_number,
            "answer_text": answer_text,
        })
    
    # Limit samples if specified
    if num_samples is not None:
        examples = examples[:num_samples]
    
    print(f"Loaded {len(examples)} examples from GSM8K")
    return examples


def load_svamp(split: str = "test", num_samples: int = None, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """Load SVAMP dataset."""
    print(f"Loading SVAMP dataset (split={split})...")
    
    # SVAMP typically uses a single split
    # Load from HuggingFace (ChilleD/SVAMP or similar)
    try:
        dataset = hf_load_dataset("ChilleD/SVAMP", split="test", cache_dir=cache_dir)
    except Exception as e:
        print(f"Warning: Could not load SVAMP from HuggingFace: {e}")
        print("Attempting alternative dataset source...")
        # Alternative: try direct dataset
        dataset = hf_load_dataset("svamp", split="test", cache_dir=cache_dir)
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: All outputs invalid - model says "The problem does not provide enough information"
    # [CAUSE]: SVAMP dataset has "Body" (context) and "Question" fields, but only "Question" was extracted, missing the problem setup
    # [FIX]: Concatenate "Body" and "Question" fields to form complete problem text
    #
    # [OLD CODE]:
    # if "Question" in item:
    #     question = item["Question"]
    #     answer_number = float(item.get("Answer", item.get("answer", 0)))
    # elif "question" in item:
    #     question = item["question"]
    #     answer_number = float(item.get("answer", 0))
    #
    # [NEW CODE]:
    # Convert to list of dicts
    examples = []
    for item in dataset:
        # SVAMP format varies by source - typically has "Body" (context) and "Question" fields
        if "Question" in item:
            # Combine Body and Question for complete problem text
            body = item.get("Body", "")
            question_text = item["Question"]
            # Create full question by concatenating body and question
            question = f"{body} {question_text}".strip() if body else question_text
            answer_number = float(item.get("Answer", item.get("answer", 0)))
        elif "question" in item:
            # Handle lowercase variant (if any)
            body = item.get("body", "")
            question_text = item["question"]
            question = f"{body} {question_text}".strip() if body else question_text
            answer_number = float(item.get("answer", 0))
        else:
            # Skip items with unexpected format
            continue
        
        examples.append({
            "question": question,
            "answer": answer_number,
        })
    
    # Limit samples if specified
    if num_samples is not None:
        examples = examples[:num_samples]
    
    print(f"Loaded {len(examples)} examples from SVAMP")
    return examples


def load_dataset(cfg) -> List[Dict[str, Any]]:
    """
    Load dataset based on config.
    
    Args:
        cfg: Dataset config with fields: name, split, num_samples, cache_dir
    
    Returns:
        List of examples with 'question' and 'answer' fields
    """
    dataset_name = cfg.name.lower()
    split = cfg.get("split", "test")
    num_samples = cfg.get("num_samples", None)
    cache_dir = cfg.get("cache_dir", ".cache")
    
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if dataset_name == "gsm8k":
        return load_gsm8k(split=split, num_samples=num_samples, cache_dir=cache_dir)
    elif dataset_name == "svamp":
        return load_svamp(split=split, num_samples=num_samples, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing GSM8K...")
    gsm8k_examples = load_gsm8k(split="test", num_samples=5)
    print(f"Sample: {gsm8k_examples[0]}")
    
    print("\nTesting SVAMP...")
    svamp_examples = load_svamp(split="test", num_samples=5)
    print(f"Sample: {svamp_examples[0]}")
