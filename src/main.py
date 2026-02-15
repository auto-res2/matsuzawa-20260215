#!/usr/bin/env python3
"""
Main orchestrator for CoT experiments.
Invokes inference.py as a subprocess for inference-only tasks.
"""

import os
import sys
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main orchestrator for a single run."""
    
    # Print config for debugging
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Determine task type from config
    # This is an inference-only task (no training)
    task_type = "inference"
    
    # Apply mode-specific overrides
    mode = cfg.get("mode", "main")
    
    if mode == "sanity_check":
        # Override settings for sanity check
        print(f"\n[MODE: {mode}] Applying sanity_check overrides...")
        
        # Reduce samples for quick validation
        if "inference" in cfg and "num_examples" in cfg.inference:
            original_num = cfg.inference.num_examples
            cfg.inference.num_examples = min(10, original_num)
            print(f"  - inference.num_examples: {original_num} -> {cfg.inference.num_examples}")
        
        if "dataset" in cfg and "num_samples" in cfg.dataset:
            original_num = cfg.dataset.num_samples
            cfg.dataset.num_samples = min(10, original_num)
            print(f"  - dataset.num_samples: {original_num} -> {cfg.dataset.num_samples}")
        
        # Use separate W&B project for sanity checks
        if "wandb" in cfg and "project" in cfg.wandb:
            original_project = cfg.wandb.project
            cfg.wandb.project = f"{original_project}-sanity"
            print(f"  - wandb.project: {original_project} -> {cfg.wandb.project}")
        
        # Keep wandb.mode as online for sanity checks
        if "wandb" in cfg:
            cfg.wandb.mode = "online"
            print(f"  - wandb.mode: online")
    
    elif mode == "main":
        print(f"\n[MODE: {mode}] Running full experiment...")
        # Ensure wandb is online for main runs
        if "wandb" in cfg:
            cfg.wandb.mode = "online"
    
    # Create results directory
    results_dir = Path(cfg.get("results_dir", ".research/results"))
    run_results_dir = results_dir / cfg.run.run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {run_results_dir}")
    
    # Save resolved config
    config_path = run_results_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved config to: {config_path}")
    
    # Invoke inference.py as subprocess
    print("\n" + "=" * 80)
    print(f"Starting {task_type} task...")
    print("=" * 80 + "\n")
    
    # Build command
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
        f"run_id={cfg.run.run_id}",
        f"results_dir={run_results_dir}",
        f"mode={mode}",
    ]
    
    print(f"Executing: {' '.join(cmd)}\n")
    
    # Run subprocess
    env = os.environ.copy()
    
    # Pass config as JSON via environment variable
    env["HYDRA_CONFIG"] = OmegaConf.to_yaml(cfg)
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd=Path.cwd(),
    )
    
    if result.returncode != 0:
        print(f"\n{'=' * 80}")
        print(f"ERROR: {task_type} failed with exit code {result.returncode}")
        print("=" * 80)
        sys.exit(result.returncode)
    
    print(f"\n{'=' * 80}")
    print(f"SUCCESS: {task_type} completed successfully")
    print(f"Results saved to: {run_results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
