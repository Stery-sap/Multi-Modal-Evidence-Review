import os
import sys
import logging
from typing import Dict, Any, List

# Resolve import name collision by removing script's directory and inserting code/ directory first
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if script_dir in sys.path:
    sys.path.remove(script_dir)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from main import ClaimVerifier
from utils import load_csv, load_user_history, load_evidence_requirements

# Restore script dir to sys.path
sys.path.append(script_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_predictions(predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compares predictions against ground truth and calculates key accuracy metrics."""
    total = len(ground_truth)
    if total == 0:
        return {}
        
    metrics = {
        "total_claims": total,
        "status_match": 0,
        "part_match": 0,
        "issue_match": 0,
        "evidence_std_match": 0,
        "valid_image_match": 0,
        "severity_match": 0,
        "risk_flags_match": 0
    }
    
    for pred, gt in zip(predictions, ground_truth):
        # 1. claim_status
        if pred["claim_status"] == gt["claim_status"]:
            metrics["status_match"] += 1
            
        # 2. object_part
        if pred["object_part"] == gt["object_part"]:
            metrics["part_match"] += 1
            
        # 3. issue_type
        if pred["issue_type"] == gt["issue_type"]:
            metrics["issue_match"] += 1
            
        # 4. evidence_standard_met
        pred_std = str(pred["evidence_standard_met"]).lower()
        gt_std = str(gt["evidence_standard_met"]).lower()
        if pred_std == gt_std:
            metrics["evidence_std_match"] += 1
            
        # 5. valid_image
        pred_valid = str(pred["valid_image"]).lower()
        gt_valid = str(gt["valid_image"]).lower()
        if pred_valid == gt_valid:
            metrics["valid_image_match"] += 1
            
        # 6. severity
        if pred["severity"] == gt["severity"]:
            metrics["severity_match"] += 1
            
        # 7. risk_flags (compare sets)
        pred_flags = set(f.strip() for f in pred["risk_flags"].split(";") if f.strip())
        gt_flags = set(f.strip() for f in gt["risk_flags"].split(";") if f.strip())
        if pred_flags == gt_flags:
            metrics["risk_flags_match"] += 1

    # Convert counts to percentages
    accuracies = {
        "total_claims": total,
        "claim_status_accuracy": (metrics["status_match"] / total) * 100,
        "object_part_accuracy": (metrics["part_match"] / total) * 100,
        "issue_type_accuracy": (metrics["issue_match"] / total) * 100,
        "evidence_standard_met_accuracy": (metrics["evidence_std_match"] / total) * 100,
        "valid_image_accuracy": (metrics["valid_image_match"] / total) * 100,
        "severity_accuracy": (metrics["severity_match"] / total) * 100,
        "risk_flags_exact_match_accuracy": (metrics["risk_flags_match"] / total) * 100
    }
    return accuracies

def run_evaluation():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "dataset", "sample_claims.csv")
    history_path = os.path.join(project_root, "dataset", "user_history.csv")
    requirements_path = os.path.join(project_root, "dataset", "evidence_requirements.csv")
    
    # Load data
    samples = load_csv(sample_path)
    user_history = load_user_history(history_path)
    requirements = load_evidence_requirements(requirements_path)
    
    logger.info(f"Loaded {len(samples)} sample claims for evaluation.")
    
    # Configuration 1: Default Strategy
    logger.info("Evaluating Strategy A (Default configuration)...")
    verifier_a = ClaimVerifier(use_model="gemini-2.5-flash")
    preds_a = []
    for claim in samples:
        res = verifier_a.process_claim(claim, user_history, requirements, samples)
        preds_a.append(res)
    metrics_a = evaluate_predictions(preds_a, samples)
    
    # Configuration 2: Alternative Strategy (e.g. OpenAI configuration or prompt variant)
    logger.info("Evaluating Strategy B (Alternative configuration / OpenAI fallback)...")
    verifier_b = ClaimVerifier(use_model="gpt-4o-mini")
    preds_b = []
    for claim in samples:
        res = verifier_b.process_claim(claim, user_history, requirements, samples)
        preds_b.append(res)
    metrics_b = evaluate_predictions(preds_b, samples)
    
    # Print results summary
    print("\n==================================================")
    print("           EVALUATION COMPARISON REPORT           ")
    print("==================================================")
    print(f"Total evaluated claims: {len(samples)}")
    print("\nMetric                           Strategy A (%)  Strategy B (%)")
    print("----------------------------------------------------------------")
    for metric in metrics_a.keys():
        if metric == "total_claims":
            continue
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]
        print(f"{metric:<32} {val_a:>12.2f}% {val_b:>12.2f}%")
    print("==================================================")
    
    # Log telemetry summaries
    print("\nStrategy A Telemetry:")
    summary_a = verifier_a.tracker.get_summary()
    for k, v in summary_a.items():
        print(f"  {k}: {v}")
        
    print("\nStrategy B Telemetry:")
    summary_b = verifier_b.tracker.get_summary()
    for k, v in summary_b.items():
        print(f"  {k}: {v}")
        
    return metrics_a, metrics_b

if __name__ == "__main__":
    run_evaluation()
