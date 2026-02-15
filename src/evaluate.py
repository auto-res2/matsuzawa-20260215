#!/usr/bin/env python3
"""
Evaluation script for aggregating and comparing metrics across runs.
Fetches data from WandB and generates comparison figures.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs to compare")
    parser.add_argument("--wandb_entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="2026-02-15", help="WandB project")
    return parser.parse_args()


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB API."""
    api = wandb.Api()
    
    try:
        # Get run
        run_path = f"{entity}/{project}/{run_id}"
        run = api.run(run_path)
        
        # Extract data
        data = {
            "run_id": run_id,
            "config": run.config,
            "summary": dict(run.summary),
            "history": run.history(pandas=True),
        }
        
        return data
    except Exception as e:
        print(f"Warning: Could not fetch WandB run {run_id}: {e}")
        return {
            "run_id": run_id,
            "config": {},
            "summary": {},
            "history": pd.DataFrame(),
        }


def load_local_metrics(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Load metrics from local results directory as fallback."""
    metrics_file = results_dir / run_id / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    return {}


def generate_per_run_figures(results_dir: Path, run_id: str, run_data: Dict) -> List[str]:
    """Generate per-run figures (e.g., accuracy over examples)."""
    generated_files = []
    
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = run_data.get("history")
    if history is None or len(history) == 0:
        return generated_files
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot accuracy over examples (if available)
    if "correct" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate rolling accuracy
        window_size = min(20, len(history))
        rolling_acc = history["correct"].rolling(window=window_size).mean()
        
        ax.plot(history.index, rolling_acc, linewidth=2)
        ax.set_xlabel("Example Index")
        ax.set_ylabel(f"Rolling Accuracy (window={window_size})")
        ax.set_title(f"Accuracy Over Time - {run_id}")
        ax.set_ylim([0, 1])
        
        output_path = run_dir / "rolling_accuracy.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        generated_files.append(str(output_path))
        print(f"Generated: {output_path}")
    
    return generated_files


def generate_comparison_figures(results_dir: Path, all_runs_data: List[Dict]) -> List[str]:
    """Generate comparison figures across all runs."""
    generated_files = []
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Bar plot: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = []
    accuracies = []
    
    for run_data in all_runs_data:
        run_id = run_data["run_id"]
        summary = run_data.get("summary", {})
        accuracy = summary.get("accuracy", 0)
        
        run_ids.append(run_id)
        accuracies.append(accuracy)
    
    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.set_title("Accuracy Comparison Across Runs")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    output_path = comparison_dir / "comparison_accuracy.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    generated_files.append(str(output_path))
    print(f"Generated: {output_path}")
    
    # 2. Line plot: Accuracy over examples (if history available)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    has_history = False
    for run_data in all_runs_data:
        history = run_data.get("history")
        if history is not None and len(history) > 0 and "correct" in history.columns:
            has_history = True
            run_id = run_data["run_id"]
            
            # Calculate rolling accuracy
            window_size = min(20, len(history))
            rolling_acc = history["correct"].rolling(window=window_size).mean()
            
            linestyle = "-" if "proposed" in run_id else "--"
            ax.plot(history.index, rolling_acc, label=run_id, linewidth=2, linestyle=linestyle)
    
    if has_history:
        ax.set_xlabel("Example Index")
        ax.set_ylabel(f"Rolling Accuracy")
        ax.set_title("Accuracy Over Examples - All Runs")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = comparison_dir / "comparison_accuracy_over_time.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        generated_files.append(str(output_path))
        print(f"Generated: {output_path}")
    else:
        plt.close()
    
    # 3. Method-specific metrics (for CF-CoT)
    cf_cot_runs = [r for r in all_runs_data if "proposed" in r["run_id"]]
    if cf_cot_runs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        run_ids_cf = []
        audit_pass_rates = []
        
        for run_data in cf_cot_runs:
            run_id = run_data["run_id"]
            summary = run_data.get("summary", {})
            audit_pass_rate = summary.get("audit_pass_rate", 0)
            
            run_ids_cf.append(run_id)
            audit_pass_rates.append(audit_pass_rate)
        
        bars = ax.bar(range(len(run_ids_cf)), audit_pass_rates, color="#9b59b6")
        ax.set_xticks(range(len(run_ids_cf)))
        ax.set_xticklabels(run_ids_cf, rotation=45, ha="right")
        ax.set_ylabel("Audit Pass Rate")
        ax.set_ylim([0, 1])
        ax.set_title("CF-CoT: Audit Pass Rate (No Arithmetic Errors)")
        
        # Add value labels
        for bar, rate in zip(bars, audit_pass_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        output_path = comparison_dir / "cf_cot_audit_pass_rate.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        generated_files.append(str(output_path))
        print(f"Generated: {output_path}")
    
    return generated_files


def main():
    """Main evaluation execution."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Fetch data from WandB for each run
    all_runs_data = []
    
    for run_id in run_ids:
        print(f"\nFetching data for run: {run_id}")
        
        # Try WandB first
        run_data = fetch_wandb_run(args.wandb_entity, args.wandb_project, run_id)
        
        # Fallback to local metrics if WandB fails
        if not run_data.get("summary"):
            local_metrics = load_local_metrics(results_dir, run_id)
            if local_metrics:
                run_data["summary"] = local_metrics
                print(f"  Loaded local metrics for {run_id}")
        
        all_runs_data.append(run_data)
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_export_path = run_dir / "metrics.json"
        with open(metrics_export_path, "w") as f:
            json.dump(run_data["summary"], f, indent=2)
        print(f"  Exported metrics to: {metrics_export_path}")
        
        # Generate per-run figures
        per_run_figures = generate_per_run_figures(results_dir, run_id, run_data)
        if per_run_figures:
            print(f"  Generated {len(per_run_figures)} per-run figures")
    
    # Generate comparison figures
    print("\nGenerating comparison figures...")
    comparison_figures = generate_comparison_figures(results_dir, all_runs_data)
    print(f"Generated {len(comparison_figures)} comparison figures")
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": {},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }
    
    proposed_accuracies = []
    baseline_accuracies = []
    
    for run_data in all_runs_data:
        run_id = run_data["run_id"]
        summary = run_data.get("summary", {})
        
        metrics_dict = {
            "accuracy": summary.get("accuracy", 0),
            "total_examples": summary.get("total_examples", 0),
            "correct": summary.get("correct", 0),
        }
        
        # Add CF-CoT specific metrics
        if "audit_pass_rate" in summary:
            metrics_dict["audit_pass_rate"] = summary["audit_pass_rate"]
            metrics_dict["audit_pass"] = summary.get("audit_pass", 0)
            metrics_dict["audit_fail"] = summary.get("audit_fail", 0)
        
        aggregated["metrics"][run_id] = metrics_dict
        
        # Track best runs
        accuracy = summary.get("accuracy", 0)
        if "proposed" in run_id:
            proposed_accuracies.append(accuracy)
        else:
            baseline_accuracies.append(accuracy)
    
    # Calculate best runs and gap
    if proposed_accuracies:
        aggregated["best_proposed"] = max(proposed_accuracies)
    if baseline_accuracies:
        aggregated["best_baseline"] = max(baseline_accuracies)
    
    if aggregated["best_proposed"] is not None and aggregated["best_baseline"] is not None:
        aggregated["gap"] = aggregated["best_proposed"] - aggregated["best_baseline"]
    
    # Export aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nExported aggregated metrics to: {aggregated_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"\nBest proposed: {aggregated['best_proposed']:.4f}" if aggregated['best_proposed'] else "Best proposed: N/A")
    print(f"Best baseline: {aggregated['best_baseline']:.4f}" if aggregated['best_baseline'] else "Best baseline: N/A")
    print(f"Gap: {aggregated['gap']:.4f}" if aggregated['gap'] else "Gap: N/A")
    print("\nPer-run metrics:")
    for run_id, metrics in aggregated["metrics"].items():
        print(f"  {run_id}: accuracy={metrics['accuracy']:.4f}")
    print("=" * 80)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
