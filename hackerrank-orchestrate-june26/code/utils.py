import csv
import os
import re
from typing import List, Dict, Any

def load_csv(file_path: str) -> List[Dict[str, str]]:
    """Loads a CSV file and returns a list of dictionaries (keys are header names)."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def write_output_csv(file_path: str, rows: List[Dict[str, Any]]):
    """Writes output rows in the exact order and column schema specified in the problem statement."""
    headers = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
    ]
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            # Prepare row matching headers exactly
            row_to_write = {}
            for h in headers:
                val = r.get(h, "")
                if isinstance(val, bool):
                    val = "true" if val else "false"
                elif val is None:
                    val = ""
                row_to_write[h] = str(val)
            writer.writerow(row_to_write)

def parse_image_paths(image_paths_str: str) -> List[str]:
    """Splits semicolon-separated image paths."""
    if not image_paths_str:
        return []
    return [p.strip() for p in image_paths_str.split(";") if p.strip()]

def get_image_id(path_str: str) -> str:
    """Extracts filename without extension (e.g., 'images/sample/case_001/img_1.jpg' -> 'img_1')."""
    base = os.path.basename(path_str)
    name, _ = os.path.splitext(base)
    return name

def load_user_history(history_path: str) -> Dict[str, Dict[str, str]]:
    """Returns a lookup mapping user_id -> user history details."""
    rows = load_csv(history_path)
    return {r["user_id"]: r for r in rows}

def load_evidence_requirements(req_path: str) -> List[Dict[str, str]]:
    """Returns the list of minimum evidence requirements."""
    return load_csv(req_path)

def get_matching_requirements(claim_object: str, user_claim: str, requirements: List[Dict[str, str]]) -> str:
    """Matches requirements from evidence_requirements.csv to the claim object and issue type."""
    matched = []
    
    # Identify issue type keywords in user claim to map specific guidelines
    claim_lower = user_claim.lower()
    issue_type_keywords = {
        "dent": ["dent", "bump", "panel", "bumper", "crushed", "press"],
        "scratch": ["scratch", "scrape", "mark", "stain", "oily"],
        "crack": ["crack", "shatter", "broken", "glass", "split"],
        "missing": ["missing", "lost", "not inside", "cannot find", "fell off", "keycap"],
        "water": ["water", "wet", "spill", "coffee", "liquid", "rain", "stain"]
    }
    
    inferred_issue_families = []
    for family, keywords in issue_type_keywords.items():
        if any(kw in claim_lower for kw in keywords):
            inferred_issue_families.append(family)
            
    for req in requirements:
        req_obj = req.get("claim_object", "").lower()
        applies_to = req.get("applies_to", "").lower()
        min_evidence = req.get("minimum_image_evidence", "")
        req_id = req.get("requirement_id", "")
        
        # 'all' applies to general claim review, reviewability, multi-image rows, etc.
        if req_obj == "all" or req_obj == claim_object.lower():
            # Check if this rule applies to the issue
            # If applies_to is 'general claim review', 'reviewability', or 'multi-image rows', always include
            if applies_to in ["general claim review", "reviewability", "multi-image rows"]:
                matched.append(f"- **{req_id} ({applies_to})**: {min_evidence}")
                continue
                
            # Otherwise match by keyword overlap
            family_matches = False
            for family in inferred_issue_families:
                # e.g., req applies to 'dent or scratch' matches 'dent' or 'scratch'
                family_words = re.split(r'\s+or\s+|\s*,\s*', applies_to)
                if any(f in family for f in family_words) or any(family in f for f in family_words):
                    family_matches = True
                    break
                    
            if family_matches or not inferred_issue_families:
                matched.append(f"- **{req_id} ({applies_to})**: {min_evidence}")
                
    if not matched:
        return "No specific guidelines matched. Check general object readability and verification rules."
        
    return "\n".join(matched)
