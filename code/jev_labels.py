"""Swap the platform classifier's label columns for the frozen JEV labels.

Domain labels join on `full_domain`, URL labels on `url`; anything the relabel missed
keeps its original label and the fallback rate is asserted to be negligible.
"""
import pandas as pd

from design import DATA


def swap(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    for task, key, col in (("domains", "full_domain", "domain_classification"),
                           ("urls", "url", "url_classification")):
        if key not in m.columns or col not in m.columns:
            continue
        lab = pd.read_parquet(DATA / f"jev_labels_{task}.parquet")
        mapped = m[key].map(dict(zip(lab[key], lab["jev_cls"])))
        miss = mapped.isna() & m[col].notna()
        print(f"  jev {task}: {len(lab)} labels, {int(miss.sum())} rows fell back to the original label")
        assert miss.mean() < 0.01, f"{col}: {miss.mean():.3f} of rows unlabelled by JEV"
        m[col] = mapped.fillna(m[col])
    return m
