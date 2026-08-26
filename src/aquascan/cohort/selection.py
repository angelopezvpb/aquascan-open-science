"""Gate 4: Deterministic cohort selection (step 2).

Select exactly 500 IGME records (100 per domain), stratified across:
- Candidate P1/P2/P3/P4 tiers
- depth ranges
- construction periods
- point nature
- QA status

Selection must be deterministic and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DOMAINS = [
    "Santa Eulalia",
    "San Carlos",
    "San Antonio",
    "Migjorn",
    "Región de los arenales",
]

PER_DOMAIN = 100

SEED = 42


def _stable_int(s: str, modulus: int = 1_000_000) -> int:
    """Deterministic integer hash of a string (independent of PYTHONHASHSEED)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % modulus


def _build_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add stratification columns to a domain subset."""
    df = df.copy()

    # Depth categories
    df["_depth_cat"] = pd.cut(
        df["total_depth_m"].fillna(-1),
        bins=[-2, 30, 100, 300, 1_000_000],
        labels=["unknown", "shallow", "medium", "deep"],
    )

    # Construction period categories
    def period(date_str: str | None) -> str:
        if date_str is None:
            return "unknown"
        try:
            year = int(str(date_str)[:4])
        except (ValueError, TypeError):
            return "unknown"
        if year < 1980:
            return "pre_1980"
        if year < 2000:
            return "1980_2000"
        return "post_2000"

    df["_period"] = df["construction_date"].apply(period)

    # QA status
    df["_has_qa"] = (df.get("qa_flag_count", 0)) > 0

    # Strata key (order-stable)
    def strata_key(row: pd.Series) -> str:
        return (
            f"{row.get('candidate_tier', 'P4')}|"
            f"{row['_depth_cat']}|{row['_period']}|"
            f"{str(row.get('point_nature', 'unknown'))[:30]}|"
            f"qa:{int(row['_has_qa'])}"
        )

    df["_strata"] = df.apply(strata_key, axis=1)
    return df


def select_domain_cohort(
    df: pd.DataFrame,
    domain: str,
    n: int = PER_DOMAIN,
    seed: int = SEED,
) -> list[dict]:
    """Select exactly n records for a domain, stratified deterministically.

    Args:
        df: Full candidate DataFrame.
        domain: Hydrogeological unit domain name.
        n: Target number of records.
        seed: Random seed for reproducibility.

    Returns:
        List of selected record dicts with selection metadata.
    """
    pool = df[df["hydrogeological_unit"] == domain].copy()
    if len(pool) == 0:
        logger.warning("Domain '%s' has 0 records — skipping", domain)
        return []

    pool = _build_strata(pool)

    # If the domain has fewer than n records, take all of them
    if len(pool) <= n:
        selected: list[dict] = []
        for _idx, row in pool.iterrows():
            rec = row.to_dict()
            rec["selection_domain"] = domain
            rec["selection_strata"] = row["_strata"]
            rec["selection_reason"] = f"all_records (n={len(pool)} <= {n})"
            rec["selection_run"] = SEED
            selected.append(rec)
        return selected

    # Deterministic proportional quota across strata, min 1 per non-empty stratum
    strata_groups = pool.groupby("_strata", sort=True)
    sizes = strata_groups.size()
    strata_names = sorted(sizes.index)

    quotas: dict[str, int] = {}
    for s in strata_names:
        q = max(1, int(round(n * sizes[s] / len(pool))))
        quotas[s] = q

    # Adjust quotas to sum exactly to n
    total_q = sum(quotas.values())
    if total_q > n:
        for s in sorted(quotas, key=lambda x: (-quotas[x], x)):
            while quotas[s] > 1 and total_q > n:
                quotas[s] -= 1
                total_q -= 1
    elif total_q < n:
        for s in sorted(quotas):
            while quotas[s] < sizes[s] and total_q < n:
                quotas[s] += 1
                total_q += 1

    # Deterministic selection within each stratum
    selected: list[dict] = []
    for s in sorted(quotas):
        stratum = strata_groups.get_group(s)
        q = quotas[s]
        if q >= len(stratum):
            pick = stratum
        else:
            # Deterministic per-stratum RNG
            rng_seed = seed + _stable_int(s)
            pick = stratum.sample(n=q, random_state=rng_seed)

        for _, row in pick.iterrows():
            rec = row.to_dict()
            rec["selection_domain"] = domain
            rec["selection_strata"] = s
            rec["selection_reason"] = f"strata={s}, quota={q}/{len(stratum)}"
            rec["selection_run"] = SEED
            selected.append(rec)

    # Final safety: exactly n records
    return selected[:n]


def select_cohort_500(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly 500 records (100 per domain) deterministically.

    If a domain cannot supply 100 eligible records, report this and
    redistribute only with explicit justification.

    Returns:
        DataFrame with selected records and selection metadata.
    """
    results: list[dict] = []
    deficits: dict[str, int] = {}

    for domain in DOMAINS:
        pool = df[df["hydrogeological_unit"] == domain]
        n_avail = len(pool)
        target = PER_DOMAIN
        logger.info(
            "Domain '%s': %d total, target %d",
            domain,
            n_avail,
            target,
        )

        if n_avail < target:
            deficits[domain] = target - n_avail
            logger.warning(
                "Domain '%s' has only %d records (deficit: %d)",
                domain,
                n_avail,
                deficits[domain],
            )

        cohort = select_domain_cohort(df, domain, min(target, n_avail))
        results.extend(cohort)

    # Redistribution if any domain has a deficit
    expected = PER_DOMAIN * len(DOMAINS)
    surplus = expected - len(results)
    if surplus > 0:
        logger.warning(
            "Redistributing %d surplus slots across domains (deficits: %s)",
            surplus,
            deficits,
        )
        already_selected = {r["igme_id"] for r in results}
        # Redistribute to domains that can still supply more, in deterministic order
        for domain in DOMAINS:
            if surplus <= 0:
                break
            pool = df[df["hydrogeological_unit"] == domain]
            already_in_domain = sum(1 for r in results if r["selection_domain"] == domain)
            available = len(pool) - already_in_domain
            if available <= 0:
                continue
            extra_n = min(surplus, available)
            extra_pool = pool[~pool["igme_id"].isin(already_selected)]
            extra = extra_pool.sample(n=extra_n, random_state=SEED + 999_000)
            for _, row in extra.iterrows():
                rec = row.to_dict()
                rec["selection_domain"] = domain
                rec["selection_strata"] = "redistribution"
                rec["selection_reason"] = f"redistribution to fill deficit in {list(deficits)}"
                rec["selection_run"] = SEED
                results.append(rec)
            surplus -= extra_n
            logger.info(
                "  Redistributed %d extra to '%s'",
                extra_n,
                domain,
            )

    result_df = pd.DataFrame(results)
    logger.info(
        "Final cohort: %d records across %d domains (target: %d)",
        len(result_df),
        result_df["selection_domain"].nunique(),
        expected,
    )

    for domain in DOMAINS:
        dom = result_df[result_df["selection_domain"] == domain]
        logger.info(
            "  %s: %d records, tiers=%s",
            domain,
            len(dom),
            dom["candidate_tier"].value_counts().to_dict(),
        )

    return result_df


PRIORITY_DOMAINS = [
    "San Antonio",
    "Santa Eulalia",
    "Migjorn",
    "San Carlos",
]
CONTRAST_DOMAIN = "Región de los arenales"
TOTAL_TARGET = 5_000


def select_cohort_5000(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly 5,000 records for Gate 5 scale-up.

    Strategy:
    - Take ALL available records from the 4 priority domains
    - Fill remaining slots from Región de los arenales (contrast domain)
    - Stratified selection within each domain

    Domain inventory:
      San Antonio: 173, Santa Eulalia: 271,
      Migjorn: 537, San Carlos: 204
      => 1,185 priority + 3,815 contrast = 5,000
    """
    results: list[dict] = []

    # Take ALL from priority domains
    for domain in PRIORITY_DOMAINS:
        pool = df[df["hydrogeological_unit"] == domain]
        n_avail = len(pool)
        if n_avail == 0:
            logger.warning("Priority domain '%s' has 0 records", domain)
            continue
        cohort = select_domain_cohort(df, domain, n_avail)
        results.extend(cohort)
        logger.info("  %s: took all %d records", domain, n_avail)

    n_priority = len(results)
    n_contrast = TOTAL_TARGET - n_priority
    logger.info(
        "Priority domains: %d records, need %d from contrast",
        n_priority,
        n_contrast,
    )

    # Fill from contrast domain
    contrast_pool = df[df["hydrogeological_unit"] == CONTRAST_DOMAIN]
    if len(contrast_pool) < n_contrast:
        logger.warning(
            "Contrast domain has only %d (need %d)",
            len(contrast_pool),
            n_contrast,
        )
        n_contrast = len(contrast_pool)

    contrast_cohort = select_domain_cohort(df, CONTRAST_DOMAIN, n_contrast)
    results.extend(contrast_cohort)
    logger.info("  %s: took %d records", CONTRAST_DOMAIN, len(contrast_cohort))

    result_df = pd.DataFrame(results)
    logger.info(
        "Final cohort: %d records across %d domains",
        len(result_df),
        result_df["selection_domain"].nunique(),
    )

    for domain in PRIORITY_DOMAINS + [CONTRAST_DOMAIN]:
        dom = result_df[result_df["selection_domain"] == domain]
        if len(dom) > 0:
            logger.info(
                "  %s: %d records, tiers=%s",
                domain,
                len(dom),
                dom["candidate_tier"].value_counts().to_dict(),
            )

    return result_df


def save_cohort(cohort_df: pd.DataFrame, path: Path) -> None:
    """ "Save cohort with selection metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(path, index=False)
    id_path = path.with_suffix(".ids.json")
    id_data = {
        "cohort_size": len(cohort_df),
        "domains": {},
        "all_ids": sorted(cohort_df["igme_id"].tolist()),
    }
    for domain in DOMAINS:
        dom_ids = sorted(cohort_df[cohort_df["selection_domain"] == domain]["igme_id"].tolist())
        id_data["domains"][domain] = {"count": len(dom_ids), "ids": dom_ids}
    with open(id_path, "w") as f:
        json.dump(id_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved cohort to %s", path)
    logger.info("Saved IDs to %s", id_path)
