"""Study 2 run universe: engine canonicalisation and the mechanical exclusion rules.

Design: 25 scenarios (5 topics x 5) x 12 prompts = 300, US.
Cell: phrasing (NEU/ADV) x FEA(+/-) x PRC(+/-) = 8 factorial cells,
plus SEED (natural seed prompt, off-factorial). Rewordings: second prompt in
NEU, ADV, ADV+FEA+PRC per scenario. The design table ships as data/prompt_design.parquet.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DERIVED = DATA  # the replication data ship prompt_design.parquet next to the run tables

# Engine canonicalisation. `sonar` = Perplexity served through the API while the
# consumer-UI scrape was down (2026-09-12..15). Same vendor, different collection
# channel: API answers carry citations on ~99.6% of chats vs ~43% for the UI scrape,
# so channel is kept as a column and the primary analysis uses channel == "ui".
ENGINE_MAP = {"sonar": "perplexity-ui"}
# Mechanical exclusion rules: one rule set for all data.
# Evaluated per canonical engine x channel on the SUCCEEDED chats of the full universe
# passed in (before any window filter); any-source = chat has >= 1 normalized source URL.
#   1. engine-day with < RUNS_PER_ENGINE_DAY SUCCEEDED runs -> drop the engine-day.
#   2. provider-day any-source rate < PROVIDER_FAIL while every other provider of that
#      engine-day (>= 1 required) is >= PROVIDER_OK -> drop that provider's chats that day.
#   3. engine-day any-source rate (after rule 2) >= ENGINE_DROP below the median of the
#      engine's previous BASELINE_DAYS complete engine-days -> drop if the next calendar day
#      fails rule 1 or has no runs (collection failure), else keep (engine change).
RUNS_PER_ENGINE_DAY = 300
# rule 2 = near-total collapse (Section 3.2): the real failure, ChatGPT BD, captured 0-4%;
# Gemini SL/DF at .49-.63 on 09-05..08 (CL ~1.0 until 09-09) is the Gemini change reaching
# providers at different times. At .50 rule 2 would also cut Gemini SL 09-05 (checked when the rules were set).
PROVIDER_FAIL, PROVIDER_OK = 0.25, 0.85
ENGINE_DROP, BASELINE_DAYS = 0.30, 7


def canonical_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Add `channel` ('ui' / 'api') and map API engine ids onto the UI engine."""
    df = df.copy()
    df["channel"] = df["model_id"].map(lambda m: "api" if m == "sonar" else "ui")
    df["model_id"] = df["model_id"].replace(ENGINE_MAP)
    return df


def source_chat_ids(path=DATA / "sources.parquet") -> set:
    """Chat ids with >= 1 normalized source URL (= revised_analysis.build_runs' any_source)."""
    m = pd.read_parquet(path, columns=["chat_id", "url_normalized"])
    return set(m["chat_id"][m["url_normalized"].fillna("").ne("")])


def exclusions(universe: pd.DataFrame, has_source=None) -> pd.DataFrame:
    """Chats dropped by the mechanical rules (canonical engine ids) with the `rule` (1/2/3).

    `has_source`: chat ids with >= 1 normalized source; default reads the metrics table.
    A flagged rule-3 day on the last loaded date counts as a collection failure (its next
    day has no runs); load one look-ahead day and cut it with build_runs(end=...).
    """
    u = canonical_engine(universe)
    has_source = source_chat_ids() if has_source is None else has_source
    ed, edp = ["model_id", "channel", "day"], ["model_id", "channel", "day", "provider_id"]
    u["day"], u["src"], u["rule"] = u["date"].astype(str).str[:10], u["chat_id"].isin(has_source), 0
    ok = u["status"].eq("SUCCEEDED")
    n = u[ok].groupby(ed).size()
    full = set(n[n >= RUNS_PER_ENGINE_DAY].index)
    u.loc[[k not in full for k in zip(*map(u.get, ed))], "rule"] = 1
    r = u[ok & u["rule"].eq(0)].groupby(edp)["src"].mean()
    bad = {k for _, s in r.groupby(level=ed) for k, v in s.items()
           if v < PROVIDER_FAIL and len(s) > 1 and s.drop(k).min() >= PROVIDER_OK}
    u.loc[[k in bad for k in zip(*map(u.get, edp))], "rule"] = 2
    rate = u[ok & u["rule"].eq(0)].groupby(ed)["src"].mean()
    for (m, c), s in rate.groupby(level=["model_id", "channel"]):
        base = s.shift(1).rolling(BASELINE_DAYS, min_periods=1).median()
        for day in s.index.get_level_values("day")[(base - s >= ENGINE_DROP - 1e-9).to_numpy()]:
            if (m, c, (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")) not in full:
                u.loc[u["model_id"].eq(m) & u["channel"].eq(c) & u["day"].eq(day) & u["rule"].eq(0), "rule"] = 3
    return u[u["rule"] > 0].drop(columns=["day", "src"])


def exclusion_audit(universe: pd.DataFrame, has_source=None) -> pd.DataFrame:
    """Label-independent audit: engine, channel, date, provider, rule, n_chats."""
    x = exclusions(universe, has_source).assign(date=lambda d: d["date"].astype(str).str[:10])
    return (x.groupby(["model_id", "channel", "date", "provider_id", "rule"]).size()
            .rename("n_chats").reset_index().rename(columns={"model_id": "engine", "provider_id": "provider"}))


def complete_engine_days(universe: pd.DataFrame, channel: str = "ui", has_source=None) -> pd.DataFrame:
    """Canonicalise the run universe and drop the chats excluded by the mechanical rules.

    Rules are evaluated on the whole universe passed in (both channels); `channel="ui"`
    then restricts to the scraped consumer UI (primary), `channel='all'` keeps the API
    fallback days as well.
    """
    u = canonical_engine(universe)
    u = u[~u["chat_id"].isin(exclusions(universe, has_source)["chat_id"])]
    return u if channel == "all" else u[u["channel"] == channel]
