import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# AGE_CATEGORIES: List[str] = ["0-29", "30-39", "40-59", "60+"]
AGE_CATEGORIES: List[str] = ["18-34", "35-59", "60-79", "80+"]
# AGE_CATEGORIES: List[str] = ["0-29", "30-59", "60+"]

symptom_clusters = {"smell_taste_disorder": "chemoreceptive_deficits",
                    "fatigue": "fatigue",
                    "shortness_of_breath": "exercise_intolerance",
                    "low_performance": "exercise_intolerance",
                    "rapid_exhaustion": "exercise_intolerance",
                    "joint_muscle_pain": "joint_muscle_pain",
                    "back_pain": "joint_muscle_pain",
                    "sore_throat": "ent",
                    "cold": "ent",
                    "symptoms_ear_pain": "ent",
                    "cough": "cough",
                    "chestpain": "chest_pain",
                    "chest_pain": "chest_pain",
                    "abdomen_pain": "gastrointestinal",
                    "diarrhea": "gastrointestinal",
                    "nausea": "gastrointestinal",
                    "circulatory_problems": "neurological",
                    "vertigo": "neurological",
                    "headache": "neurological",
                    "cognitive_impairment": "neurological",
                    "concentration_problems": "neurological",
                    "memory_impairment": "neurological",
                    "muscle_twitching": "neurological",
                    "sensory_disturbances": "neurological",
                    "muscle_weakness": "neurological",
                    "fainting_spell": "neurological",
                    "fever": "infection",
                    "lymph_node_swelling": "infection",
                    "sleep_disorder": "sleep_disturbances",
                    "night_sweats": "sleep_disturbances",
                    "anxiety": "psychological",
                    "depression": "psychological",
                    "eye_conjunctivitis": "other",
                    "arrhythmia": "other",
                    "pms": "other",
                    "pain_intercourse": "other",
                    "light_sensitivity": "other"}


def parse_range(s: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(s):
        return None, None
    s = str(s).strip()
    if not s:
        return None, None
    if s.endswith('+'):
        try:
            lo = int(s[:-1])
            return lo, None
        except ValueError:
            return None, None
    if '-' in s:
        parts = s.split('-')
        try:
            lo = int(parts[0])
            hi = int(parts[1])
            return lo, hi
        except ValueError:
            return None, None
    if s.isdigit():
        a = int(s)
        return a, a
    return None, None


def overlaps(r_lo: Optional[int], r_hi: Optional[int], c_lo: Optional[int], c_hi: Optional[int]) -> bool:
    if r_lo is None:
        return False
    r_hi = float('inf') if r_hi is None else r_hi
    c_hi = float('inf') if c_hi is None else c_hi
    return not (r_hi < c_lo or c_hi < r_lo)


def compute_cat_ranges(age_categories: List[str]) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    return {cat: parse_range(cat) for cat in age_categories}


def add_age_category_flags(df: pd.DataFrame, age_categories: List[str] = AGE_CATEGORIES, col: str = "basis_age_cat") -> pd.DataFrame:
    cat_ranges = compute_cat_ranges(age_categories)
    for cat, (c_lo, c_hi) in cat_ranges.items():
        def flag(v, c_lo=c_lo, c_hi=c_hi, cat=cat):
            r_lo, r_hi = parse_range(v)
            if r_lo is None:
                # fallback to exact string match (case-insensitive)
                return int(str(v).strip().lower() == cat.lower())
            return int(overlaps(r_lo, r_hi, c_lo, c_hi))
        df[cat] = df[col].apply(flag)
    return df

def merge_age_categories(df: pd.DataFrame, merge_cateogories: List[str]) -> pd.DataFrame:
    minimum_age = min(parse_range(cat)[0] for cat in merge_cateogories)
    maximum_age = max(parse_range(cat)[1] for cat in merge_cateogories)
    df[f"{minimum_age}-{maximum_age}"] = df[merge_cateogories].max(axis=1)
    return df

def add_female_column(df: pd.DataFrame, col: str = "basis_geschl", female_value: str = "weiblich", out_col: str = "female") -> pd.DataFrame:
    df[out_col] = (df[col] == female_value).astype(int)
    return df

def add_comorbidity_column(df: pd.DataFrame, col: str = "CCI_Score", out_col: str = "comorbidity") -> pd.DataFrame:
    df[out_col] = (df[col] > 1).astype(int) 
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    replace_column_names = {
        "ih_acute_sym": "pcc_0months_sym",
        "ih_4weeks_sym": "pcc_1months_sym",
        "pcc_sym_12weeks": "pcc_3months_sym",
        "pcc_lcr_1": "pcc_6months",
        "pcc_lcr_2": "pcc_12months",
        "pcc_lcr_3": "pcc_18months",
        "pcc_lcr_4": "pcc_24months",
        "pcc_lcr_5": "pcc_30months",
    }
    
    rename_mapping = {}
    for old, new in replace_column_names.items():
        for col in df.columns:
            if old in col:
                rename_mapping[col] = col.replace(old, new)

    return df.rename(columns=rename_mapping)

def is_healed(df: pd.DataFrame) -> pd.DataFrame:

    for followup in range(1, 5):
        date_col = f"pcc_lcr_{followup}_submitdate"
        pcc_col = f"pcc_lcr_{followup}_ongoing_PCC"

        healed_mask = (
            df[date_col].notna()
            & (df[pcc_col] == "Nein")
        )
        if healed_mask.any():
            for f in range(followup, 6):
                cols = [c for c in df.columns if c.startswith(f"pcc_lcr_{f}_sym_")]
                if cols:
                    df.loc[healed_mask, cols] = "Beschwerde liegt nicht vor"

    for followup in range(1, 6):
        followup_sym_cols = [c for c in df.columns if c.startswith(f"pcc_lcr_{followup}_sym_")]
        if followup_sym_cols:
            df[followup_sym_cols] = df[followup_sym_cols].fillna(-1)
    months3_sym_cols = [c for c in df.columns if c.startswith(f"pcc_sym_12weeks")]
    if months3_sym_cols:
        df[months3_sym_cols] = df[months3_sym_cols].fillna(-1)

    return df


def map_symptoms_severity_to_binary(df: pd.DataFrame) -> pd.DataFrame: 
    symptom_cols = [c for c in df.columns if "_sym_" in c]
    severity_mapping = {
        "Beschwerde liegt nicht vor": 0,
        "Nicht COVID bezogen": 0,
        "Wenig beeinträchtigt": 1,
        "Stark beeinträchtigt": 1,
        "Weiß nicht": -1,
        "Gar nicht": 0,
        "Sehr schwach": 0,
        "Schwach": 1,
        "Mittelmäßig": 1,
        "Schwer": 1,
        "Sehr schwer": 1,
        -1: -1,
        "Beschwerde liegt vor und steht im Zusammenhang mit COVID, stark beeinträchtigt": 1,
        "Beschwerde liegt vor und steht im Zusammenhang mit COVID, wenig beeinträchtigt": 1,
        "Beschwerde liegt vor, steht aber nicht im Zusammenhang mit COVID": 0,
        "Beschwerde liegt vor, nicht beeinträchtigt": 0
    }
    for col in symptom_cols:
        df[col] = df[col].fillna(-1).map(severity_mapping)
    return df


def preprocess_file(in_path: Path, out_path: Path, age_categories: List[str] = AGE_CATEGORIES) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    df = is_healed(df)
    df = rename_columns(df)
    add_age_category_flags(df, age_categories=age_categories)
    add_female_column(df)
    add_comorbidity_column(df)
    df = map_symptoms_severity_to_binary(df)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DigiHero symptom CSV")
    parser.add_argument("infile", nargs="?", default="data/DigiHero/2026-07-01_data_symptoms_DigiHero_Bonn.csv")
    parser.add_argument("outfile", nargs="?", default="data/DigiHero/2026-07-01_data_symptoms_DigiHero_Bonn_preprocessed.csv")
    args = parser.parse_args()
    preprocess_file(Path(args.infile), Path(args.outfile))


if __name__ == "__main__":
    main()