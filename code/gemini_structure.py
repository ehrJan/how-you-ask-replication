"""Break test on Gemini answer structure (answer fields that do not depend on source parsing).

Daily series over the SUCCEEDED gemini-ui chats of data/<org>/chats_structure.parquet
(answer-structure export): mean message length (characters), share of answers with a TABLE
element, with a SOURCES element, with no element. Each series gets model_regime's
sup-|t| single-split test (permutation p over day labels, MAD-z single-day flags).
Study org only (Gemini claims rest on this study's data). Output:
out/revised/gemini_structure_breaks.csv
"""
import numpy as np
import pandas as pd

from design import DATA
from model_regime import sup_t_break
from revised_analysis import OUT, SEED

PATH = DATA / "chats_structure.parquet"
SERIES = ["message_length", "has_table", "has_sources", "no_element"]


def daily(chats):
    """Per-day Gemini means of the four structure outcomes (date index, SERIES columns)."""
    g = chats[chats["model_id"] == "gemini-ui"]
    el = g["element_types"].map(list)
    d = pd.DataFrame({"date": pd.to_datetime(g["date"]), "message_length": g["message_length"],
                      "has_table": el.map(lambda e: "TABLE" in e), "has_sources": el.map(lambda e: "SOURCES" in e),
                      "no_element": el.map(len).eq(0)})
    return d.groupby("date")[SERIES].mean().sort_index()


def breaks(series, rng):
    return pd.DataFrame([dict(outcome=c, **sup_t_break(series.index.to_numpy(), series[c].to_numpy(), rng))
                         for c in SERIES])


def main():
    if not PATH.exists():
        print(f"{PATH} missing. Nothing written.")
        return
    out = breaks(daily(pd.read_parquet(PATH)), np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "gemini_structure_breaks.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
