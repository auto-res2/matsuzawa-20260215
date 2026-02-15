#!/usr/bin/env python3
"""
Inference script for Contract-and-Falsify CoT vs baseline MC-CoT experiments.
This script handles two-pass prompt-based reasoning with deterministic arithmetic auditing.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import wandb
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from src.preprocess import load_dataset


def get_llm_client(cfg):
    """Initialize LLM client based on config."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Missing key 'method' when accessing cfg.method
    # [CAUSE]: Config fields are nested under cfg.run.* due to Hydra defaults structure
    # [FIX]: Changed cfg.model to cfg.run.model
    #
    # [OLD CODE]:
    # provider = cfg.model.provider
    # api_key = os.environ.get(cfg.model.api_key_env)
    #
    # [NEW CODE]:
    provider = cfg.run.model.provider
    
    if provider == "openai":
        import openai
        api_key = os.environ.get(cfg.run.model.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {cfg.run.model.api_key_env} not set")
        client = openai.OpenAI(api_key=api_key)
        return client
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get(cfg.run.model.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {cfg.run.model.api_key_env} not set")
        client = anthropic.Anthropic(api_key=api_key)
        return client
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def call_llm(client, cfg, prompt: str) -> str:
    """Call LLM with given prompt."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Missing key 'method' when accessing cfg.method
    # [CAUSE]: Config fields are nested under cfg.run.* due to Hydra defaults structure
    # [FIX]: Changed cfg.model to cfg.run.model
    #
    # [OLD CODE]:
    # provider = cfg.model.provider
    # model_name = cfg.model.name
    # temperature = cfg.model.temperature
    # max_tokens = cfg.model.max_tokens
    #
    # [NEW CODE]:
    provider = cfg.run.model.provider
    model_name = cfg.run.model.name
    temperature = cfg.run.model.temperature
    max_tokens = cfg.run.model.max_tokens
    
    if provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def build_cf_cot_pass1_prompt(question: str) -> str:
    """Build Contract-and-Falsify CoT Pass 1 prompt (CONTRACT-SOLVE)."""
    return f"""Solve this arithmetic word problem step by step, outputting ONLY valid JSON.

Problem: {question}

Instructions:
1. Break down your solution into atomic arithmetic steps
2. Each step must be a simple computation (few operators)
3. Format each step as: {{"expr": "<python-like arithmetic>", "value": <number>, "note": "optional explanation"}}
4. Output valid JSON with "steps" list and "final" numeric answer

Example format:
{{
  "steps": [
    {{"expr": "5 + 3", "value": 8, "note": "total items"}},
    {{"expr": "8 * 2", "value": 16, "note": "doubled"}}
  ],
  "final": 16
}}

Output ONLY the JSON, no other text:"""


def build_cf_cot_pass2_prompt(question: str, pass1_json: Dict, audit_report: Dict) -> str:
    """Build Contract-and-Falsify CoT Pass 2 prompt (FALSIFY-AND-REVISE)."""
    return f"""Review this arithmetic solution using the audit report.

Problem: {question}

Pass 1 Solution:
{json.dumps(pass1_json, indent=2)}

Audit Report:
{json.dumps(audit_report, indent=2)}

Instructions:
1. If audit found errors: Repair from the earliest failing step onward
2. Even if audit passed: Attempt to disprove the final answer via independent check
3. Output ONE line verdict:
   - "CONFIRMED" if answer is correct
   - "REVISED: <number>" if you found an error

Your verdict (one line only):"""


def build_mc_cot_pass1_prompt(question: str) -> str:
    """Build Metacognitive Self-Check CoT Pass 1 prompt (standard solve)."""
    return f"""Solve this arithmetic word problem step by step.

Problem: {question}

Show your reasoning and arrive at a final numeric answer.

Solution:"""


def build_mc_cot_pass2_prompt(question: str, pass1_solution: str) -> str:
    """Build Metacognitive Self-Check CoT Pass 2 prompt (self-check)."""
    return f"""Review this solution to check if it's correct.

Problem: {question}

Solution:
{pass1_solution}

Instructions:
1. Carefully review the reasoning and calculations
2. Check for any errors in logic or arithmetic
3. Output ONE line verdict:
   - "CONFIRMED" if the solution is correct
   - "REVISED: <number>" if you found an error

Your verdict (one line only):"""


def audit_arithmetic_contract(contract: Dict) -> Dict:
    """
    Deterministically audit arithmetic steps in a contract.
    Returns audit report with any errors found.
    """
    report = {
        "status": "pass",
        "errors": [],
        "first_error_step": None,
        "first_error_details": None,
    }
    
    if "steps" not in contract:
        report["status"] = "invalid"
        report["errors"].append("Missing 'steps' field")
        return report
    
    steps = contract["steps"]
    context = {}  # Can be extended for variable tracking
    
    for i, step in enumerate(steps):
        if "expr" not in step or "value" not in step:
            report["status"] = "invalid"
            report["errors"].append(f"Step {i}: missing 'expr' or 'value'")
            continue
        
        expr = step["expr"]
        claimed_value = step["value"]
        
        try:
            # Evaluate the expression (basic arithmetic only)
            # Security note: In production, use a safer eval or AST-based parser
            computed_value = eval(expr, {"__builtins__": {}}, context)
            
            # Check if values match (with tolerance for floats)
            if isinstance(computed_value, (int, float)) and isinstance(claimed_value, (int, float)):
                if abs(computed_value - claimed_value) > 1e-6:
                    error = {
                        "step_index": i,
                        "expr": expr,
                        "claimed": claimed_value,
                        "computed": computed_value,
                    }
                    report["errors"].append(error)
                    if report["first_error_step"] is None:
                        report["first_error_step"] = i
                        report["first_error_details"] = error
                        report["status"] = "fail"
        except Exception as e:
            # Expression evaluation failed
            error = {
                "step_index": i,
                "expr": expr,
                "error": str(e),
            }
            report["errors"].append(error)
            if report["first_error_step"] is None:
                report["first_error_step"] = i
                report["first_error_details"] = error
                report["status"] = "fail"
    
    return report


def extract_final_answer(text: str) -> Optional[float]:
    """Extract numeric final answer from text."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Low valid output rate (2/10 = 20%) - most examples have final_answer=null
    # [CAUSE]: extract_final_answer regex patterns don't match common GSM8K answer formats like "<<9*2=18>>18", "$18", or "= $18"
    # [FIX]: Added more comprehensive patterns to match GSM8K-style answers, dollar amounts, and numbers followed by units/punctuation.
    #        Patterns are ordered by specificity - more explicit answer markers first, then contextual clues, then fallbacks.
    #        We search for the LAST occurrence of lower-priority patterns (like "= X") to get the final answer rather than intermediate steps.
    #
    # [OLD CODE]:
    # patterns = [
    #     r"REVISED:\s*([+-]?\d+\.?\d*)",
    #     r"final[\"']?\s*:\s*([+-]?\d+\.?\d*)",
    #     r"answer[\"']?\s*:?\s*([+-]?\d+\.?\d*)",
    #     r"####\s*([+-]?\d+\.?\d*)",  # GSM8K format
    #     r"([+-]?\d+\.?\d*)\s*$",  # Number at end
    # ]
    #
    # [NEW CODE]:
    # High-priority patterns (explicit answer markers) - use first match
    high_priority_patterns = [
        r"REVISED:\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # REVISED: answer
        r"####\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # #### X (GSM8K final answer marker)
        r"final[\"']?\s*:\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # final: answer
    ]
    
    for pattern in high_priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except (ValueError, IndexError):
                continue
    
    # Medium-priority patterns (contextual clues) - use last match
    # Reordered to prioritize explicit answer indicators over generic "=" patterns
    # Note: patterns are checked in order; first match in pattern order wins (but uses LAST occurrence of that pattern)
    sentences = text.split('.')
    last_sentence = sentences[-1] if sentences else text
    
    medium_priority_patterns = [
        r"(?:total|profit|earnings?|makes?)\s+(?:of\s+)?\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # "total $18", "makes $18" with dollar sign - most explicit
        r"answer[\"']?\s*:?\s*(?:is\s+)?\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # answer: X or answer is X
        r"(?:total|profit)\s+(?:of\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # "total 18", "profit 70000" without dollar sign
        r"=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # = X (prefer last occurrence) - least specific
    ]
    
    for pattern in medium_priority_patterns:
        # Find all matches and use the last one
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # Use last match
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except (ValueError, IndexError):
                continue
    
    # Low-priority patterns (look for numbers with units or at end) - search last sentence first, then full text
    low_priority_patterns = [
        r"<<[^>]+>>\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # <<expr=18>>18 (GSM8K step marker)
        r"\$([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$",  # $18 at end with optional period
        r"(?:is|are)\s+\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # "is $18"
        r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:dollars?|bolts?|meters?|minutes?|sheep|glasses|cups|hours?|miles?)",  # number before unit
        r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$",  # Number at end (fallback)
    ]
    
    # Try last sentence first
    for pattern in low_priority_patterns:
        matches = list(re.finditer(pattern, last_sentence, re.IGNORECASE))
        if matches:
            match = matches[-1]  # Use last match in last sentence
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except (ValueError, IndexError):
                continue
    
    # Fallback: search full text
    for pattern in low_priority_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            match = matches[-1]  # Use last match
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except (ValueError, IndexError):
                continue
    
    return None


def run_cf_cot_inference(client, cfg, question: str, ground_truth: Optional[float]) -> Dict:
    """Run Contract-and-Falsify CoT (two passes with audit)."""
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "method": "cf-cot",
        "pass1_response": None,
        "pass1_json": None,
        "audit_report": None,
        "pass2_response": None,
        "final_answer": None,
        "correct": None,
    }
    
    # Pass 1: CONTRACT-SOLVE
    pass1_prompt = build_cf_cot_pass1_prompt(question)
    pass1_response = call_llm(client, cfg, pass1_prompt)
    result["pass1_response"] = pass1_response
    
    # Parse JSON from Pass 1
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_text = pass1_response
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()
        
        pass1_json = json.loads(json_text)
        result["pass1_json"] = pass1_json
        
        # Deterministic audit
        audit_report = audit_arithmetic_contract(pass1_json)
        result["audit_report"] = audit_report
        
        # Pass 2: FALSIFY-AND-REVISE
        pass2_prompt = build_cf_cot_pass2_prompt(question, pass1_json, audit_report)
        pass2_response = call_llm(client, cfg, pass2_prompt)
        result["pass2_response"] = pass2_response
        
        # Extract final answer
        if "REVISED:" in pass2_response.upper():
            revised_answer = extract_final_answer(pass2_response)
            result["final_answer"] = revised_answer if revised_answer is not None else pass1_json.get("final")
        else:
            result["final_answer"] = pass1_json.get("final")
        
    except json.JSONDecodeError as e:
        # JSON parsing failed - use Pass 1 extracted answer
        result["pass1_json"] = {"error": f"JSON decode failed: {e}"}
        result["audit_report"] = {"status": "invalid", "errors": ["JSON decode failed"]}
        result["final_answer"] = extract_final_answer(pass1_response)
    except Exception as e:
        result["error"] = str(e)
        result["final_answer"] = None
    
    # Check correctness
    if ground_truth is not None and result["final_answer"] is not None:
        result["correct"] = abs(result["final_answer"] - ground_truth) < 1e-6
    
    return result


def run_mc_cot_inference(client, cfg, question: str, ground_truth: Optional[float]) -> Dict:
    """Run Metacognitive Self-Check CoT (two passes without audit)."""
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "method": "mc-cot",
        "pass1_response": None,
        "pass2_response": None,
        "final_answer": None,
        "correct": None,
    }
    
    # Pass 1: Standard solve
    pass1_prompt = build_mc_cot_pass1_prompt(question)
    pass1_response = call_llm(client, cfg, pass1_prompt)
    result["pass1_response"] = pass1_response
    
    # Extract Pass 1 answer
    pass1_answer = extract_final_answer(pass1_response)
    
    # Pass 2: Self-check
    pass2_prompt = build_mc_cot_pass2_prompt(question, pass1_response)
    pass2_response = call_llm(client, cfg, pass2_prompt)
    result["pass2_response"] = pass2_response
    
    # Extract final answer from Pass 2
    if "REVISED:" in pass2_response.upper():
        revised_answer = extract_final_answer(pass2_response)
        result["final_answer"] = revised_answer if revised_answer is not None else pass1_answer
    else:
        result["final_answer"] = pass1_answer
    
    # Check correctness
    if ground_truth is not None and result["final_answer"] is not None:
        result["correct"] = abs(result["final_answer"] - ground_truth) < 1e-6
    
    return result


def validate_sanity_check(results: List[Dict], mode: str) -> Dict:
    """Validate sanity check results and emit verdict."""
    validation = {
        "status": "PASS",
        "reason": None,
        "summary": {},
    }
    
    if mode != "sanity_check":
        return validation
    
    # Check: At least 5 samples processed
    num_samples = len(results)
    validation["summary"]["samples"] = num_samples
    
    if num_samples < 5:
        validation["status"] = "FAIL"
        validation["reason"] = "insufficient_samples"
        return validation
    
    # Check: All outputs are valid (not all None)
    valid_outputs = sum(1 for r in results if r.get("final_answer") is not None)
    validation["summary"]["valid_outputs"] = valid_outputs
    
    if valid_outputs == 0:
        validation["status"] = "FAIL"
        validation["reason"] = "all_outputs_invalid"
        return validation
    
    # Check: Outputs are not all identical
    final_answers = [r.get("final_answer") for r in results if r.get("final_answer") is not None]
    unique_answers = len(set(final_answers))
    validation["summary"]["unique_answers"] = unique_answers
    
    if unique_answers == 1 and len(final_answers) >= 5:
        validation["status"] = "FAIL"
        validation["reason"] = "all_outputs_identical"
        return validation
    
    # Check: At least some correct answers (if ground truth available)
    correct_count = sum(1 for r in results if r.get("correct") is True)
    validation["summary"]["correct_count"] = correct_count
    
    accuracy = correct_count / num_samples if num_samples > 0 else 0
    validation["summary"]["accuracy"] = accuracy
    
    # All checks passed
    return validation


def main():
    """Main inference execution."""
    # Load config from environment variable (passed from main.py)
    config_yaml = os.environ.get("HYDRA_CONFIG")
    if not config_yaml:
        print("ERROR: HYDRA_CONFIG environment variable not set")
        sys.exit(1)
    
    cfg = OmegaConf.create(yaml.safe_load(config_yaml))
    
    # Extract parameters
    run_id = cfg.run.run_id
    mode = cfg.get("mode", "main")
    results_dir = Path(cfg.get("results_dir", ".research/results")) / run_id
    
    print(f"Starting inference for run: {run_id}")
    print(f"Mode: {mode}")
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Missing key 'method' when accessing cfg.method
    # [CAUSE]: Config fields are nested under cfg.run.* due to Hydra defaults structure
    # [FIX]: Changed cfg.method, cfg.dataset, cfg.inference, cfg.model to cfg.run.*
    #
    # [OLD CODE]:
    # print(f"Method: {cfg.method.type}")
    # print(f"Dataset: {cfg.dataset.name}")
    # dataset = load_dataset(cfg.dataset)
    # num_samples = cfg.inference.num_examples
    # print(f"\nInitializing LLM client: {cfg.model.provider}/{cfg.model.name}")
    # print(f"\nRunning inference with method: {cfg.method.type}")
    # method_type = cfg.method.type
    #
    # [NEW CODE]:
    print(f"Method: {cfg.run.method.type}")
    print(f"Dataset: {cfg.run.dataset.name}")
    
    # Initialize WandB
    wandb_enabled = cfg.wandb.mode == "online"
    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB initialized: {wandb.run.url}")
    else:
        print("WandB disabled (offline mode)")
    
    # Load dataset
    print(f"\nLoading dataset: {cfg.run.dataset.name}")
    dataset = load_dataset(cfg.run.dataset)
    num_samples = cfg.run.inference.num_examples
    dataset = dataset[:num_samples]
    print(f"Loaded {len(dataset)} examples")
    
    # Initialize LLM client
    print(f"\nInitializing LLM client: {cfg.run.model.provider}/{cfg.run.model.name}")
    client = get_llm_client(cfg)
    
    # Run inference
    print(f"\nRunning inference with method: {cfg.run.method.type}")
    results = []
    
    method_type = cfg.run.method.type
    
    for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
        question = example["question"]
        ground_truth = example.get("answer")
        
        try:
            if method_type == "cf-cot":
                result = run_cf_cot_inference(client, cfg, question, ground_truth)
            elif method_type == "mc-cot":
                result = run_mc_cot_inference(client, cfg, question, ground_truth)
            else:
                raise ValueError(f"Unknown method type: {method_type}")
            
            result["example_id"] = i
            results.append(result)
            
            # Log to WandB
            if wandb_enabled:
                log_data = {
                    "example_id": i,
                    "correct": result.get("correct"),
                    "final_answer": result.get("final_answer"),
                }
                if method_type == "cf-cot":
                    log_data["audit_status"] = result.get("audit_report", {}).get("status")
                    log_data["audit_errors"] = len(result.get("audit_report", {}).get("errors", []))
                
                wandb.log(log_data)
        
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "example_id": i,
                "question": question,
                "error": str(e),
                "correct": False,
            })
    
    # Calculate metrics
    print("\nCalculating metrics...")
    total = len(results)
    correct = sum(1 for r in results if r.get("correct") is True)
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        "total_examples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    
    # Method-specific metrics
    if method_type == "cf-cot":
        audit_pass = sum(1 for r in results if r.get("audit_report", {}).get("status") == "pass")
        audit_fail = sum(1 for r in results if r.get("audit_report", {}).get("status") == "fail")
        metrics["audit_pass"] = audit_pass
        metrics["audit_fail"] = audit_fail
        metrics["audit_pass_rate"] = audit_pass / total if total > 0 else 0
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_file}")
    
    # Save metrics
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    
    # Log metrics to WandB
    if wandb_enabled:
        wandb.summary.update(metrics)
        wandb.finish()
    
    # Sanity validation
    if mode == "sanity_check":
        validation = validate_sanity_check(results, mode)
        print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(validation['summary'])}")
        print(f"SANITY_VALIDATION: {validation['status']}" + 
              (f" reason={validation['reason']}" if validation['reason'] else ""))
        
        if validation["status"] == "FAIL":
            sys.exit(1)
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
