#!/usr/bin/env python
"""
Run all PADDLE examples to demonstrate capabilities.
"""

import subprocess
import sys
import os


def run_example(script_path):
    """Run a single example script."""
    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80 + "\n")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\nWarning: {script_path} exited with code {result.returncode}")
    
    return result.returncode


def main():
    """Run all example scripts."""
    examples = [
        "examples/churn_prediction_example.py",
        "examples/sales_forecast_example.py",
        "examples/fraud_detection_example.py",
        "examples/pricing_optimization_example.py",
        "examples/recommendation_system_example.py",
        "examples/inventory_optimization_example.py",
    ]
    
    print("=" * 80)
    print("PADDLE - Predictive Analytics and Data-driven Decision Learning Engine")
    print("Running All Examples")
    print("=" * 80)
    
    results = {}
    for example in examples:
        if os.path.exists(example):
            results[example] = run_example(example)
        else:
            print(f"Warning: {example} not found")
            results[example] = -1
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    for example, code in results.items():
        status = "✓ SUCCESS" if code == 0 else "✗ FAILED"
        print(f"{status}: {example}")
    
    successful = sum(1 for code in results.values() if code == 0)
    total = len(results)
    print(f"\nTotal: {successful}/{total} examples ran successfully")
    print("=" * 80)
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
